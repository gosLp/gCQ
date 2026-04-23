// rt_compaction.cpp
// ============================================================================
// BASELINE: Double-Buffer with Stream Compaction
// ============================================================================
// This implements the standard wavefront raytracing approach used by:
// - PBRT-4 GPU path tracer (Pharr et al., 2023)
// - "Megakernels Considered Harmful" (Laine et al., SIGGRAPH 2013)
// - Most production GPU renderers (Arnold GPU, Cycles, etc.)
//
// Algorithm:
// 1. Generate all primary rays into contiguous buffer
// 2. Trace kernel: each thread processes rays[thread_id], writes results
// 3. Mark active rays (those needing continuation, e.g., reflections)
// 4. Stream compaction: parallel prefix sum to pack active rays contiguously
// 5. Repeat until no active rays remain
//
// Compile:
//   hipcc -O3 rt_compaction.cpp -o rt_compact
//
// ============================================================================

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <fstream>

#define METHOD_NAME "COMPACT"

#define HIP_CHECK(call) do {                                 \
  hipError_t _e = (call);                                    \
  if (_e != hipSuccess) {                                    \
    fprintf(stderr, "HIP error %s:%d: %s\n",                 \
            __FILE__, __LINE__, hipGetErrorString(_e));      \
    exit(1);                                                 \
  }                                                          \
} while(0)

#define HIP_LAUNCH_CHECK() do {                              \
  HIP_CHECK(hipGetLastError());                              \
  HIP_CHECK(hipDeviceSynchronize());                         \
} while(0)

// ============================================================================
// Math Helpers (same as queue version for fair comparison)
// ============================================================================
__host__ __device__ inline float3 f3(float x, float y, float z) {
    return make_float3(x, y, z);
}
__host__ __device__ inline float3 add(const float3& a, const float3& b) {
    return f3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__host__ __device__ inline float3 sub(const float3& a, const float3& b) {
    return f3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__host__ __device__ inline float3 mul(const float3& a, float s) {
    return f3(a.x * s, a.y * s, a.z * s);
}
__host__ __device__ inline float dot3(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__host__ __device__ inline float3 normalize3(const float3& v) {
    float d = sqrtf(dot3(v, v));
    return d > 0 ? mul(v, 1.0f / d) : v;
}
__host__ __device__ inline float3 clamp01(const float3& c) {
    return f3(fminf(fmaxf(c.x, 0.f), 1.f),
              fminf(fmaxf(c.y, 0.f), 1.f),
              fminf(fmaxf(c.z, 0.f), 1.f));
}

// debug macros 
#define HIP_CHECK(call) do {                                \
  hipError_t _e = (call);                                   \
  if (_e != hipSuccess) {                                   \
    fprintf(stderr, "HIP error %s:%d: %s\n",                \
            __FILE__, __LINE__, hipGetErrorString(_e));     \
    exit(1);                                                \
  }                                                         \
} while(0)

#define HIP_LAUNCH_CHECK() do {                             \
  HIP_CHECK(hipGetLastError());                             \
  HIP_CHECK(hipDeviceSynchronize());                        \
} while(0)


// ============================================================================
// Scene Structures (identical to queue version)
// ============================================================================
struct Sphere {
    float3 center;
    float radius;
    float3 albedo;
    float reflectivity;
};

struct Plane {
    float3 normal;
    float d;
    float3 albedo;
    float reflectivity;
};

// Ray structure - slightly different: we store pixel coords and accumulated color
struct Ray {
    float3 origin;
    float3 dir;
    int px, py;           // Pixel coordinates for writeback
    int depth;            // Bounce depth
    int active;           // 1 if ray needs processing, 0 if terminated
};

struct Hit {
    float t;
    float3 normal;
    float3 point;
    float3 albedo;
    float reflectivity;
    bool hit;
};

struct Pixel {
    float r, g, b;
};

// ============================================================================
// Intersection Tests (identical to queue version)
// ============================================================================
__device__ Hit intersect_sphere(const Ray& ray, const Sphere& s) {
    Hit h;
    h.hit = false;
    h.t = 1e30f;

    float3 oc = sub(ray.origin, s.center);
    float b = dot3(oc, ray.dir);
    float c = dot3(oc, oc) - s.radius * s.radius;
    float disc = b * b - c;

    if (disc >= 0.f) {
        float t = -b - sqrtf(disc);
        if (t > 1e-4f && t < h.t) {
            h.t = t;
            h.hit = true;
            h.point = add(ray.origin, mul(ray.dir, t));
            h.normal = normalize3(sub(h.point, s.center));
            h.albedo = s.albedo;
            h.reflectivity = s.reflectivity;
        }
    }
    return h;
}

__device__ Hit intersect_plane(const Ray& ray, const Plane& pl) {
    Hit h;
    h.hit = false;
    h.t = 1e30f;

    float denom = dot3(pl.normal, ray.dir);
    if (fabsf(denom) > 1e-5f) {
        float t = -(dot3(pl.normal, ray.origin) + pl.d) / denom;
        if (t > 1e-4f && t < h.t) {
            h.t = t;
            h.hit = true;
            h.point = add(ray.origin, mul(ray.dir, t));
            h.normal = pl.normal;
            h.albedo = pl.albedo;
            h.reflectivity = pl.reflectivity;
        }
    }
    return h;
}

__device__ Hit trace_scene(const Ray& r,
                           const Sphere* spheres, int num_spheres,
                           const Plane* planes, int num_planes) {
    Hit best;
    best.hit = false;
    best.t = 1e30f;

    for (int i = 0; i < num_spheres; i++) {
        Hit h = intersect_sphere(r, spheres[i]);
        if (h.hit && h.t < best.t) best = h;
    }
    for (int i = 0; i < num_planes; i++) {
        Hit h = intersect_plane(r, planes[i]);
        if (h.hit && h.t < best.t) best = h;
    }
    return best;
}

// ============================================================================
// Image Helpers
// ============================================================================
__device__ inline void write_pixel(Pixel* img, int w, int x, int y, const float3& c) {
    int idx = y * w + x;
    img[idx].r = c.x;
    img[idx].g = c.y;
    img[idx].b = c.z;
}

__device__ inline void add_pixel(Pixel* img, int w, int x, int y, const float3& c) {
    int idx = y * w + x;
    atomicAdd(&img[idx].r, c.x);
    atomicAdd(&img[idx].g, c.y);
    atomicAdd(&img[idx].b, c.z);
}

// ============================================================================
// KERNEL: Generate Primary Rays
// ============================================================================
// Generates all W*H primary rays into a contiguous buffer
// This is O(pixels) work, embarrassingly parallel
__global__ void kernel_generate_rays(
    Ray* rays,
    int img_w, int img_h,
    float3 cam_origin,
    float3 cam_fwd,
    float3 cam_right,
    float3 cam_up)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = img_w * img_h;
    if (idx >= total_pixels) return;

    int px = idx % img_w;
    int py = idx / img_w;

    float aspect = (float)img_w / (float)img_h;
    float u = ((px + 0.5f) / img_w - 0.5f) * 2.0f * aspect;
    float v = -((py + 0.5f) / img_h - 0.5f) * 2.0f;

    Ray r;
    r.origin = cam_origin;
    r.dir = normalize3(add(cam_fwd, add(mul(cam_right, u), mul(cam_up, v))));
    r.px = px;
    r.py = py;
    r.depth = 0;
    r.active = 1;

    rays[idx] = r;
}

// ============================================================================
// KERNEL: Trace Rays (Single Pass)
// ============================================================================
// Each thread traces one ray from rays_in[idx]
// - Writes color to image
// - If reflection needed and depth < max_depth:
//   - Writes reflection ray to rays_out[idx]
//   - Sets needs_continuation[idx] = 1
// - Else sets needs_continuation[idx] = 0
__global__ void kernel_trace_rays(
    const Ray* __restrict__ rays_in,
    Ray* __restrict__ rays_out,
    int* __restrict__ needs_continuation,
    int num_rays,
    const Sphere* __restrict__ spheres, int num_spheres,
    const Plane* __restrict__ planes, int num_planes,
    Pixel* __restrict__ img,
    int img_w, int img_h,
    int max_depth,
    float depth_weight)  // weight for this bounce (1.0 for primary, 0.5 for secondary, etc.)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rays) return;

    Ray r = rays_in[idx];
    if (!r.active) {
        needs_continuation[idx] = 0;
        return;
    }

    const float3 light_dir = normalize3(f3(-0.5f, 1.0f, 0.2f));
    const float3 ambient = f3(0.05f, 0.05f, 0.05f);

    Hit hit = trace_scene(r, spheres, num_spheres, planes, num_planes);
    float3 color = f3(0, 0, 0);
    int wants_reflection = 0;

    if (hit.hit) {
        float NdotL = fmaxf(0.f, dot3(hit.normal, light_dir));
        color = add(ambient, mul(hit.albedo, NdotL));

        // Generate reflection ray if surface is reflective and we haven't hit max depth
        if (hit.reflectivity > 0.0f && r.depth < max_depth) {
            Ray refl;
            refl.origin = add(hit.point, mul(hit.normal, 1e-4f));
            refl.dir = normalize3(sub(r.dir, mul(hit.normal, 2.f * dot3(r.dir, hit.normal))));
            refl.px = r.px;
            refl.py = r.py;
            refl.depth = r.depth + 1;
            refl.active = 1;
            rays_out[idx] = refl;
            wants_reflection = 1;
        }
    } else {
        // Sky gradient
        float t = 0.5f * (r.dir.y + 1.0f);
        color = add(mul(f3(1, 1, 1), 1.0f - t), mul(f3(0.5f, 0.7f, 1.0f), t));
    }

    // Write color to image (with depth weight for blending bounces)
    color = mul(clamp01(color), depth_weight);
    if (r.depth == 0) {
        write_pixel(img, img_w, r.px, r.py, color);
    } else {
        add_pixel(img, img_w, r.px, r.py, color);
    }

    needs_continuation[idx] = wants_reflection;
}

// ============================================================================
// STREAM COMPACTION: Parallel Prefix Sum + Scatter
// ============================================================================
// This is the key difference from queue-based approaches.
// We use a simple two-phase approach:
// 1. Prefix sum to compute output indices
// 2. Scatter active rays to their new positions
//
// For production, you'd use rocPRIM or similar, but we implement a basic
// version to show the algorithm and avoid external dependencies.

// Prefix sum using shared memory (Blelloch scan)
// This is a work-efficient parallel prefix sum
#define BLOCK_SIZE 256

__global__ void kernel_prefix_sum_local(
    const int* __restrict__ input,
    int* __restrict__ output,
    int* __restrict__ block_sums,
    int n)
{
    __shared__ int temp[BLOCK_SIZE * 2];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x * 2 + tid;
    
    // Load input into shared memory
    temp[tid] = (gid < n) ? input[gid] : 0;
    temp[tid + BLOCK_SIZE] = (gid + BLOCK_SIZE < n) ? input[gid + BLOCK_SIZE] : 0;
    
    // Build sum in place (up-sweep)
    int offset = 1;
    for (int d = BLOCK_SIZE; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    
    // Store block sum and clear last element
    if (tid == 0) {
        if (block_sums) block_sums[blockIdx.x] = temp[BLOCK_SIZE * 2 - 1];
        temp[BLOCK_SIZE * 2 - 1] = 0;
    }
    
    // Down-sweep
    for (int d = 1; d < BLOCK_SIZE * 2; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    
    __syncthreads();
    
    // Write results
    if (gid < n) output[gid] = temp[tid];
    if (gid + BLOCK_SIZE < n) output[gid + BLOCK_SIZE] = temp[tid + BLOCK_SIZE];
}

__global__ void kernel_add_block_sums(
    int* __restrict__ output,
    const int* __restrict__ block_sums,
    int n)
{
    int gid = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    if (blockIdx.x > 0) {
        int add_val = block_sums[blockIdx.x];
        if (gid < n) output[gid] += add_val;
        if (gid + BLOCK_SIZE < n) output[gid + BLOCK_SIZE] += add_val;
    }
}

// Simple recursive prefix sum for larger arrays
void prefix_sum_gpu(const int* d_input, int* d_output, int n, 
                    int* d_temp1, int* d_temp2) {
    int num_blocks = (n + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
    
    if (num_blocks == 1) {
        // Single block - simple case
        kernel_prefix_sum_local<<<1, BLOCK_SIZE>>>(d_input, d_output, nullptr, n);
    } else {
        // Multi-block - need to handle block sums
        int* d_block_sums = d_temp1;
        int* d_block_sums_scanned = d_temp2;
        
        // Local prefix sums
        kernel_prefix_sum_local<<<num_blocks, BLOCK_SIZE>>>(
            d_input, d_output, d_block_sums, n);
        
        // Recursively scan block sums (for simplicity, we'll do it on host for small counts)
        // In production, this would be recursive GPU calls
        if (num_blocks <= BLOCK_SIZE * 2) {
            kernel_prefix_sum_local<<<1, BLOCK_SIZE>>>(
                d_block_sums, d_block_sums_scanned, nullptr, num_blocks);
        } else {
            // For very large arrays, need more recursion levels
            // This is sufficient for our image sizes
            std::vector<int> h_sums(num_blocks);
            hipMemcpy(h_sums.data(), d_block_sums, num_blocks * sizeof(int), 
                     hipMemcpyDeviceToHost);
            int acc = 0;
            for (int i = 0; i < num_blocks; i++) {
                int t = h_sums[i];
                h_sums[i] = acc;
                acc += t;
            }
            hipMemcpy(d_block_sums_scanned, h_sums.data(), num_blocks * sizeof(int),
                     hipMemcpyHostToDevice);
        }
        
        // Add block sums back
        kernel_add_block_sums<<<num_blocks, BLOCK_SIZE>>>(
            d_output, d_block_sums_scanned, n);
    }
    hipDeviceSynchronize();
}

// Scatter kernel: compact active rays using prefix sum results
__global__ void kernel_scatter_rays(
    const Ray* __restrict__ rays_in,
    Ray* __restrict__ rays_out,
    const int* __restrict__ flags,       // 1 if ray should be kept
    const int* __restrict__ scatter_idx, // prefix sum of flags (output position)
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    if (flags[idx]) {
        int out_idx = scatter_idx[idx];
        rays_out[out_idx] = rays_in[idx];
    }
}

// Get final count (last prefix sum value + last flag)
__global__ void kernel_get_count(
    const int* __restrict__ flags,
    const int* __restrict__ prefix_sum,
    int n,
    int* __restrict__ count)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *count = prefix_sum[n - 1] + flags[n - 1];
    }
}

// ============================================================================
// PPM Writer
// ============================================================================
void save_ppm(const std::string& path, const std::vector<Pixel>& img, int w, int h) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { perror("save_ppm"); return; }
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    for (int i = 0; i < w * h; i++) {
        unsigned char r = (unsigned char)std::min(255.f, std::max(0.f, img[i].r * 255.f));
        unsigned char g = (unsigned char)std::min(255.f, std::max(0.f, img[i].g * 255.f));
        unsigned char b = (unsigned char)std::min(255.f, std::max(0.f, img[i].b * 255.f));
        fwrite(&r, 1, 1, f);
        fwrite(&g, 1, 1, f);
        fwrite(&b, 1, 1, f);
    }
    fclose(f);
}

std::string getGpuShort(const char* name) {
    std::string full(name);
    size_t pos = full.find_last_of(" ");
    if (pos != std::string::npos && pos + 1 < full.size()) {
        return full.substr(pos + 1);
    }
    return full;
}

static void build_scene_complex(std::vector<Sphere>& sph, std::vector<Plane>& pln) {
    // Floor
    pln.push_back({f3(0,1,0), 1.0f, f3(0.75f,0.75f,0.75f), 0.05f});

    // 10x10 sphere grid
    const int gx = 10, gz = 10;
    const float spacing = 0.55f;
    const float base_y  = 0.25f;
    const float radius  = 0.22f;

    for (int iz = 0; iz < gz; iz++) {
        for (int ix = 0; ix < gx; ix++) {
            float x = (ix - (gx - 1) * 0.5f) * spacing;
            float z = -2.0f - iz * spacing;

            int m = (ix + iz * 7) % 7;
            float refl =
                (m == 0) ? 0.9f :
                (m == 1) ? 0.6f :
                (m == 2) ? 0.3f :
                (m == 3) ? 0.1f :
                           0.0f;

            float3 alb =
                (m == 0) ? f3(0.9f, 0.9f, 0.9f) :
                (m == 1) ? f3(0.9f, 0.5f, 0.2f) :
                (m == 2) ? f3(0.2f, 0.8f, 0.9f) :
                (m == 3) ? f3(0.7f, 0.2f, 0.8f) :
                (m == 4) ? f3(0.2f, 0.9f, 0.3f) :
                (m == 5) ? f3(0.9f, 0.2f, 0.2f) :
                           f3(0.8f, 0.8f, 0.3f);

            sph.push_back({f3(x, base_y, z), radius, alb, refl});
        }
    }

    // Hero spheres
    sph.push_back({f3(-1.2f, 0.6f, -3.5f), 0.6f, f3(0.9f,0.3f,0.3f), 0.7f});
    sph.push_back({f3( 1.4f, 0.9f, -4.2f), 0.9f, f3(0.3f,0.9f,0.4f), 0.2f});
    sph.push_back({f3( 0.0f, 1.3f, -5.5f), 1.3f, f3(0.4f,0.6f,0.95f), 0.0f});
}

static void build_scene_cornell(std::vector<Sphere>& sph, std::vector<Plane>& pln) {
    sph.push_back({f3(-0.6f,0.5f,-2.5f), 0.5f, f3(0.9f,0.3f,0.3f),   0.85f});
    sph.push_back({f3( 0.6f,0.6f,-2.8f), 0.6f, f3(0.95f,0.95f,0.95f), 0.95f});

    pln.push_back({f3( 0, 1,0),  1.0f, f3(0.9f,0.9f,0.9f),    0.3f});  // floor
    pln.push_back({f3( 0, 0,1),  5.0f, f3(0.6f,0.65f,0.8f),   0.7f});  // back
    pln.push_back({f3( 1, 0,0),  2.0f, f3(0.8f,0.2f,0.2f),    0.6f});  // left
    pln.push_back({f3(-1, 0,0),  2.0f, f3(0.2f,0.8f,0.3f),    0.6f});  // right
    pln.push_back({f3( 0,-1,0),  2.5f, f3(0.95f,0.95f,0.95f), 0.2f});  // ceiling
}

static void build_scene_queue_friendly(std::vector<Sphere>& sph, std::vector<Plane>& pln) {
    // ------------------------------------------------------------------------
    // Queue-friendly "mirror tunnel" scene:
    //
    // Goal:
    //   - keep a large fraction of rays alive for many bounces
    //   - but not uniformly: some rays die early on matte absorbers
    //   - create a long-lived, spatially skewed active set
    //
    // This is much more favorable to a persistent queue than Scene 0.
    // ------------------------------------------------------------------------

    // Reflective tunnel / chamber
    pln.push_back({f3( 1, 0, 0), 2.0f,  f3(0.95f, 0.95f, 0.95f), 0.985f}); // left   x = -2.0
    pln.push_back({f3(-1, 0, 0), 2.0f,  f3(0.95f, 0.95f, 0.95f), 0.985f}); // right  x =  2.0
    pln.push_back({f3( 0, 1, 0), 1.2f,  f3(0.95f, 0.95f, 0.95f), 0.985f}); // floor  y = -1.2
    pln.push_back({f3( 0,-1, 0), 1.2f,  f3(0.95f, 0.95f, 0.95f), 0.985f}); // ceil   y =  1.2
    pln.push_back({f3( 0, 0, 1), 20.0f, f3(0.95f, 0.95f, 0.95f), 0.985f}); // back   z = -20
    pln.push_back({f3( 0, 0,-1), 2.5f,  f3(0.95f, 0.95f, 0.95f), 0.985f}); // front  z =  2.5

    // Matte / low-reflective absorbers:
    // These kill some rays early, so the continuation pattern is not uniform.
    sph.push_back({f3(-1.45f, -0.25f, -4.2f), 0.70f, f3(0.85f, 0.20f, 0.20f), 0.00f});
    sph.push_back({f3( 1.35f,  0.20f, -5.3f), 0.80f, f3(0.20f, 0.85f, 0.25f), 0.00f});
    sph.push_back({f3( 0.00f, -0.75f, -8.5f), 0.40f, f3(0.85f, 0.85f, 0.20f), 0.00f});

    // Reflective chain through the tunnel:
    // Center-ish rays can keep bouncing around these for many depths.
    sph.push_back({f3( 0.00f,  0.00f, -4.8f), 0.90f, f3(0.95f, 0.95f, 0.95f), 0.995f});
    sph.push_back({f3(-0.75f,  0.35f, -7.2f), 0.85f, f3(0.95f, 0.85f, 0.85f), 0.995f});
    sph.push_back({f3( 0.80f, -0.25f, -9.8f), 0.90f, f3(0.85f, 0.95f, 0.85f), 0.995f});
    sph.push_back({f3(-0.55f,  0.30f, -12.2f),0.95f, f3(0.85f, 0.85f, 0.95f), 0.995f});
    sph.push_back({f3( 0.70f, -0.20f, -14.7f),1.00f, f3(0.95f, 0.95f, 0.80f), 0.995f});
    sph.push_back({f3( 0.00f,  0.10f, -17.2f),1.05f, f3(0.90f, 0.90f, 0.95f), 0.995f});
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char* argv[]) {
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    std::cout << "Wavefront Raytracer (Double-Buffer + Stream Compaction) on " 
              << prop.name << "\n";
    
    int scene_id = (argc > 1) ? atoi(argv[1]) : 0; // 0 = simple, 1 = complex
    int max_bounces = (argc > 2) ? atoi(argv[2]) : 4; // Max reflection depth

    std::cout << "Scene " << scene_id << " ("
          << (scene_id == 0 ? "Complex Spheres" :
              scene_id == 1 ? "Cornell Box" :
                              "Mirror Tunnel")
          << ")\n";
    std::cout << "Max bounces: " << max_bounces << "\n";

    // Image dimensions (same as queue version for fair comparison)
    const int W = 1280, H = 720;
    const int TOTAL_PIXELS = W * H;
    const int MAX_DEPTH = max_bounces; // 0 = primary only, 1 = one reflection bounce

    const int TPB = 1024;
    const int NUM_BLOCKS = (TOTAL_PIXELS + TPB - 1) / TPB;

    std::cout << "Resolution: " << W << "x" << H << " = " << TOTAL_PIXELS << " pixels\n";
    std::cout << "Max bounce depth: " << MAX_DEPTH << "\n";

    // ========================================================================
    // Allocate Buffers
    // ========================================================================
    
    // Double buffer for rays (ping-pong)
    Ray* d_rays[2];
    hipMalloc(&d_rays[0], TOTAL_PIXELS * sizeof(Ray));
    hipMalloc(&d_rays[1], TOTAL_PIXELS * sizeof(Ray));

    // Continuation flags and prefix sum
    int* d_flags;
    int* d_prefix_sum;
    int* d_temp1;  // For prefix sum block sums
    int* d_temp2;
    int* d_count;
    
    hipMalloc(&d_flags, TOTAL_PIXELS * sizeof(int));
    hipMalloc(&d_prefix_sum, TOTAL_PIXELS * sizeof(int));
    
    int num_blocks_prefix = (TOTAL_PIXELS + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
    hipMalloc(&d_temp1, num_blocks_prefix * sizeof(int));
    hipMalloc(&d_temp2, num_blocks_prefix * sizeof(int));
    hipMalloc(&d_count, sizeof(int));

    // Image buffer
    Pixel* d_img;
    hipMalloc(&d_img, TOTAL_PIXELS * sizeof(Pixel));
    hipMemset(d_img, 0, TOTAL_PIXELS * sizeof(Pixel));

    // ========================================================================
    // Scene Setup (identical to queue version)
    // ========================================================================
    // Sphere h_spheres[2];
    // h_spheres[0] = {f3(-0.7f, 0.5f, -3.0f), 0.5f, f3(0.9f, 0.3f, 0.3f), 0.4f};
    // h_spheres[1] = {f3(0.8f, 0.7f, -2.5f), 0.7f, f3(0.3f, 0.9f, 0.4f), 0.0f};

    // Plane h_planes[1];
    // h_planes[0] = {f3(0, 1, 0), 0.9f, f3(0.8f, 0.8f, 0.8f), 0.2f};

    // Sphere* d_spheres;
    // Plane* d_planes;
    // hipMalloc(&d_spheres, sizeof(h_spheres));
    // hipMalloc(&d_planes, sizeof(h_planes));
    // hipMemcpy(d_spheres, h_spheres, sizeof(h_spheres), hipMemcpyHostToDevice);
    // hipMemcpy(d_planes, h_planes, sizeof(h_planes), hipMemcpyHostToDevice);

      // ---------------- Scene selection ----------------
    std::vector<Sphere> h_spheres;
    std::vector<Plane>  h_planes;

    if (scene_id == 0) {
        build_scene_complex(h_spheres, h_planes);
    } else if (scene_id == 1) {
        build_scene_cornell(h_spheres, h_planes);
    } else {
        build_scene_queue_friendly(h_spheres, h_planes);
    }

    Sphere* d_spheres = nullptr;
    Plane*  d_planes  = nullptr;

    HIP_CHECK(hipMalloc(&d_spheres, h_spheres.size() * sizeof(Sphere)));
    HIP_CHECK(hipMalloc(&d_planes,  h_planes.size()  * sizeof(Plane)));
    HIP_CHECK(hipMemcpy(d_spheres, h_spheres.data(),
                        h_spheres.size() * sizeof(Sphere), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_planes,  h_planes.data(),
                        h_planes.size() * sizeof(Plane),  hipMemcpyHostToDevice));

    int ns = (int)h_spheres.size();
    int np = (int)h_planes.size();

    std::cout << "Spheres: " << ns << ", Planes: " << np << "\n";

    // Camera
    float3 cam_origin = f3(0, 0.3f, 0.5f);
    float3 cam_fwd = normalize3(f3(0, -0.2f, -1.0f));
    float3 cam_right = normalize3(f3(1, 0, 0));
    float3 cam_up = normalize3(f3(0, 1, 0));

    if (scene_id == 0) {
        cam_origin = f3(0, 0.3f, 0.5f);
        cam_fwd    = normalize3(f3(0, -0.2f, -1.0f));
    } else if (scene_id == 1) {
        cam_origin = f3(0, 0.5f, 1.5f);
        cam_fwd    = normalize3(f3(0, -0.1f, -1.0f));
    } else {
        cam_origin = f3(0.0f, 0.0f, 1.6f);
        cam_fwd    = normalize3(f3(0.0f, 0.0f, -1.0f));
    }
    cam_right = normalize3(f3(1, 0, 0));
    cam_up    = normalize3(f3(0, 1, 0));

    // ========================================================================
    // Timing
    // ========================================================================
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    float ms_total = 0.0f;
    unsigned long long total_rays = 0;

    // ========================================================================
    // PASS 0: Generate Primary Rays
    // ========================================================================
    std::cout << "\nPASS 0: Generating " << TOTAL_PIXELS << " primary rays\n";

    hipEventRecord(start);
    kernel_generate_rays<<<NUM_BLOCKS, TPB>>>(
        d_rays[0], W, H, cam_origin, cam_fwd, cam_right, cam_up);
    HIP_LAUNCH_CHECK();
    hipDeviceSynchronize();
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float ms_gen;
    hipEventElapsedTime(&ms_gen, start, stop);
    ms_total += ms_gen;
    total_rays += TOTAL_PIXELS;

    std::cout << "  Generation time: " << ms_gen << " ms\n";

    // ========================================================================
    // Wavefront Loop: Trace -> Compact -> Repeat
    // ========================================================================
    int current_buf = 0;
    int num_active_rays = TOTAL_PIXELS;
    int depth = 0;
    int prev_count = TOTAL_PIXELS;  // Track count before compaction for scatter

    while (num_active_rays > 0 && depth <= MAX_DEPTH) {
        std::cout << "\nPASS " << (depth + 1) << ": Tracing " << num_active_rays << " rays (depth=" << depth << ")\n";

        int next_buf = 1 - current_buf;
        int trace_blocks = (num_active_rays + TPB - 1) / TPB;

        // Weight for color accumulation (primary = 1.0, reflections = 0.5)
        float weight = (depth == 0) ? 1.0f : 0.5f;

        // ---------------------------------------------------------------------
        // Trace rays
        // ---------------------------------------------------------------------
        hipEventRecord(start);
        kernel_trace_rays<<<trace_blocks, TPB>>>(
            d_rays[current_buf],
            d_rays[next_buf],
            d_flags,
            num_active_rays,
            d_spheres, ns,
            d_planes, np,
            d_img, W, H,
            MAX_DEPTH,
            weight);
        HIP_LAUNCH_CHECK();
        hipEventRecord(stop);
        hipEventSynchronize(stop);

        float ms_trace;
        hipEventElapsedTime(&ms_trace, start, stop);
        ms_total += ms_trace;
        std::cout << "  Trace time: " << ms_trace << " ms\n";

        // Check if we've reached max depth
        if (depth >= MAX_DEPTH) {
            std::cout << "  Max depth reached, terminating\n";
            break;
        }

        // ---------------------------------------------------------------------
        // Stream compaction: prefix sum on flags, then scatter
        // ---------------------------------------------------------------------
        hipEventRecord(start);

        // Prefix sum on continuation flags
        prefix_sum_gpu(d_flags, d_prefix_sum, num_active_rays, d_temp1, d_temp2);

        // Get count of active rays for next iteration
        kernel_get_count<<<1, 1>>>(d_flags, d_prefix_sum, num_active_rays, d_count);
        HIP_LAUNCH_CHECK();
        hipDeviceSynchronize();
        
        prev_count = num_active_rays;  // Save for scatter
        hipMemcpy(&num_active_rays, d_count, sizeof(int), hipMemcpyDeviceToHost);

        std::cout << "  Rays wanting continuation: " << num_active_rays << "\n";

        if (num_active_rays > 0) {
            // Scatter: compact reflection rays from next_buf to current_buf
            // We use the prefix sum to pack them contiguously into current_buf
            int scatter_blocks = (prev_count + TPB - 1) / TPB;
            kernel_scatter_rays<<<scatter_blocks, TPB>>>(
                d_rays[next_buf],      // Source: reflection rays (at parent indices)
                d_rays[current_buf],   // Dest: compacted buffer for next iteration
                d_flags,
                d_prefix_sum,
                prev_count);           // Number of elements to check
            hipDeviceSynchronize();
        }

        hipEventRecord(stop);
        hipEventSynchronize(stop);

        float ms_compact;
        hipEventElapsedTime(&ms_compact, start, stop);
        ms_total += ms_compact;

        std::cout << "  Compaction time: " << ms_compact << " ms\n";
        std::cout << "  Active rays for next pass: " << num_active_rays << "\n";

        total_rays += num_active_rays;
        depth++;

        // Note: After compaction, rays are in d_rays[current_buf], ready for next trace
        // We DON'T swap buffers - current_buf now has compacted rays
    }

    // ========================================================================
    // Results
    // ========================================================================
    double mrays = (total_rays / 1.0e6) / (ms_total / 1.0e3);

    std::cout << "\n========================================\n";
    std::cout << "RESULTS (Double-Buffer + Stream Compaction)\n";
    std::cout << "========================================\n";
    std::cout << "Total rays traced: " << total_rays << "\n";
    std::cout << "Total time: " << ms_total << " ms\n";
    std::cout << "Throughput: " << mrays << " MRays/s\n";

    // Save to CSV for comparison
    const char* csv_path = "rt_performance.csv";
    std::string device = getGpuShort(prop.name);

    bool file_exists = static_cast<bool>(std::ifstream(csv_path));
    std::ofstream csv(csv_path, std::ios::app);
    if (csv) {
        if (!file_exists) {
            csv << "DEVICE,QUEUE,THREADS,TOTAL_RAYS,TOTAL_TIME_MS,TOTAL_MRAYS\n";
        }
        csv << device << ","
            << METHOD_NAME << ","
            << TPB << ","          // "threads" - not really applicable, use block size
            << total_rays << ","
            << ms_total << ","
            << mrays << "\n";
    }

    // ========================================================================
    // Save Image
    // ========================================================================
    std::vector<Pixel> h_img(TOTAL_PIXELS);
    hipMemcpy(h_img.data(), d_img, TOTAL_PIXELS * sizeof(Pixel), hipMemcpyDeviceToHost);

    // std::string scene_str = (scene_id == 0) ? "complex" : "cornell";
    std::string scene_str =
    (scene_id == 0) ? "complex" :
    (scene_id == 1) ? "cornell" :
                      "mirror_tunnel";
    std::string filename = "rt_" + std::string(METHOD_NAME) + "_" + scene_str + ".ppm";
    save_ppm(filename, h_img, W, H);
    std::cout << "\nImage saved: " << filename << "\n";

    // ========================================================================
    // Cleanup
    // ========================================================================
    hipFree(d_rays[0]);
    hipFree(d_rays[1]);
    hipFree(d_flags);
    hipFree(d_prefix_sum);
    hipFree(d_temp1);
    hipFree(d_temp2);
    hipFree(d_count);
    hipFree(d_img);
    hipFree(d_spheres);
    hipFree(d_planes);
    hipEventDestroy(start);
    hipEventDestroy(stop);

    return 0;
}