// // raytrace_final.cpp - Final ray tracing benchmark with two scenes
// // Scene 0: Complex sphere field (100+ spheres with varied materials)
// // Scene 1: Cornell Box style - 2 spheres + floor + 3 reflective walls, multiple bounces
// //
// // Compile examples:
// //   hipcc -O3 raytrace_final.cpp -o rt_wfq
// //   hipcc -O3 raytrace_final.cpp -DUSE_CAWFQ -o rt_cawfq
// //   hipcc -O3 raytrace_final.cpp -DUSE_BWFQ -o rt_bwfq
// //   hipcc -O3 raytrace_final.cpp -DUSE_SFQ -o rt_sfq
// //
// // Usage: ./rt_wfq [threads] [scene] [max_bounces]
// //   threads: number of threads per tile (default: 1024)
// //   scene: 0 = complex spheres, 1 = cornell box (default: 0)
// //   max_bounces: reflection depth for scene 1 (default: 4)

// #include <hip/hip_runtime.h>
// #include <cstdio>
// #include <cstdlib>
// #include <cstdint>
// #include <cmath>
// #include <vector>
// #include <string>
// #include <iostream>
// #include <algorithm>
// #include <fstream>

// #define HIP_CHECK(call) do {                                 \
//   hipError_t _e = (call);                                    \
//   if (_e != hipSuccess) {                                    \
//     fprintf(stderr, "HIP error %s:%d: %s\n",                 \
//             __FILE__, __LINE__, hipGetErrorString(_e));      \
//     exit(1);                                                 \
//   }                                                          \
// } while(0)

// #define HIP_LAUNCH_CHECK() do {                              \
//   HIP_CHECK(hipGetLastError());                              \
//   HIP_CHECK(hipDeviceSynchronize());                         \
// } while(0)

// // ============================================================================
// // Queue Sizing Knobs (benchmark-focused)
// // ============================================================================
// // Keep queue sizing closer to per-tile demand to reduce init/memory overhead.
// // Override at compile time if needed.
// #ifndef RT_QUEUE_RING_CAPACITY
// #define RT_QUEUE_RING_CAPACITY 32768ULL
// #endif

// #ifndef RT_QUEUE_SEGMENT_SAFETY
// #define RT_QUEUE_SEGMENT_SAFETY 4
// #endif

// // ============================================================================
// // Queue Selection
// // ============================================================================
// #ifdef USE_CAWFQ
//   #include "../wfqueue_cawfq.hpp"
//   #include "../wfqueue_cawfq.cpp"
//   #define QUEUE_TYPE    wf_queue
//   #define HANDLE_TYPE   wf_handle
//   #define QUEUE_EMPTY   WF_EMPTY
//   #define QUEUE_NAME    "CAWFQ"
//   #define HAS_LOCAL_BUF 1
// #elif defined(USE_BLFQ)
//   #ifndef CAWFQ_PREALLOC_OPS_PER_THREAD
//     // Bounded-ring backend: keep capacity close to tile-level working set.
//     #define CAWFQ_PREALLOC_OPS_PER_THREAD 32
//   #endif
//   #ifndef CAWFQ_SEGMENT_SAFETY
//     #define CAWFQ_SEGMENT_SAFETY RT_QUEUE_SEGMENT_SAFETY
//   #endif
//   #include "../wfqueue_cawfq2.hpp"
//   // #include "../wfqueue_cawfq2.cpp"
//   #define QUEUE_TYPE    wf_queue
//   #define HANDLE_TYPE   wf_handle
//   #define QUEUE_EMPTY   WF_EMPTY
//   #define QUEUE_NAME    "BLFQ-WarpBatch"
//   #define HAS_LOCAL_BUF 0
// #elif defined(USE_BWFQ)
//   #ifndef WF_RING_CAPACITY
//     #define WF_RING_CAPACITY RT_QUEUE_RING_CAPACITY
//   #endif
//   #include "../wfqueue_bounded.hpp"
//   #define QUEUE_TYPE    wf_queue
//   #define HANDLE_TYPE   wf_handle
//   #define QUEUE_EMPTY   WF_EMPTY
//   #define QUEUE_NAME    "BWFQ"
//   #define HAS_LOCAL_BUF 0
//   #define NEEDS_RECORDS 1
// #elif defined(USE_SFQ)
//   #include "../sfqueue_hip.hpp"
//   #include "../sfqueue_hip.cpp"
//   #define QUEUE_TYPE    sfq_queue
//   #define HANDLE_TYPE   sfq_handle
//   #define QUEUE_EMPTY   SFQ_EMPTY
//   #define QUEUE_NAME    "SFQ"
//   #define HAS_LOCAL_BUF 0
//   #define wf_enqueue    sfq_enqueue
//   #define wf_dequeue    sfq_dequeue
//   #define wf_queue_host_init sfq_queue_host_init
// #elif defined(USE_WLQ)
//   // #include "worklist_mpmc_hip.hpp"
//   // #include "worklist_mpmc_hip.cpp"

//   // #define QUEUE_TYPE    wlq_queue
//   // #define HANDLE_TYPE   wlq_handle
//   // #define QUEUE_EMPTY   WLQ_EMPTY
//   // #define QUEUE_NAME    "WLQ_MPMC_RING"
//   // #define HAS_LOCAL_BUF 0
//   // #define wf_enqueue wlq_enqueue
//   // #define wf_dequeue wlq_dequeue
//   #include "rt_qshim.hpp"
// #else
//   // Default: WFQ
//   #include "../wfqueue_hip_opt.hpp"
//   #define QUEUE_TYPE    wf_queue
//   #define HANDLE_TYPE   wf_handle
//   #define QUEUE_EMPTY   WF_EMPTY
//   #define QUEUE_NAME    "WFQ"
//   #define HAS_LOCAL_BUF 0
// #endif

// // Token encoding: never enqueue 0
// #define ENC_IDX(i)  ((uint64_t)((i) + 1))
// #define DEC_TOK(t)  ((int)((t) - 1))

// // ============================================================================
// // Math Helpers
// // ============================================================================
// __host__ __device__ inline float3 f3(float x, float y, float z) { return make_float3(x, y, z); }
// __host__ __device__ inline float3 add(const float3& a, const float3& b) { return f3(a.x+b.x, a.y+b.y, a.z+b.z); }
// __host__ __device__ inline float3 sub(const float3& a, const float3& b) { return f3(a.x-b.x, a.y-b.y, a.z-b.z); }
// __host__ __device__ inline float3 mul(const float3& a, float s) { return f3(a.x*s, a.y*s, a.z*s); }
// __host__ __device__ inline float3 mul3(const float3& a, const float3& b) { return f3(a.x*b.x, a.y*b.y, a.z*b.z); }
// __host__ __device__ inline float  dot3(const float3& a, const float3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
// __host__ __device__ inline float3 normalize3(const float3& v) {
//   float d = sqrtf(dot3(v, v));
//   return d > 0 ? mul(v, 1.0f/d) : v;
// }
// __host__ __device__ inline float3 reflect3(const float3& v, const float3& n) {
//   return sub(v, mul(n, 2.f * dot3(v, n)));
// }
// __host__ __device__ inline float3 clamp01(const float3& c) {
//   return f3(fminf(fmaxf(c.x, 0.f), 1.f),
//             fminf(fmaxf(c.y, 0.f), 1.f),
//             fminf(fmaxf(c.z, 0.f), 1.f));
// }

// // ============================================================================
// // Scene Structures
// // ============================================================================
// struct Sphere { float3 c; float r; float3 albedo; float reflect; };
// struct Plane  { float3 n; float d; float3 albedo; float reflect; };

// struct Ray {
//   float3 o, d;
//   int px, py;
//   int depth;
//   float3 attenuation;  // accumulated color multiplier for multi-bounce
// };

// struct Hit {
//   float t;
//   float3 n;
//   float3 p;
//   float3 albedo;
//   float reflect;
//   bool hit;
// };

// struct Pixel { float r, g, b; };

// // ============================================================================
// // Intersection Tests
// // ============================================================================
// __device__ Hit hit_sphere(const Ray& ray, const Sphere& s) {
//   float3 oc = sub(ray.o, s.c);
//   float b = dot3(oc, ray.d);
//   float c = dot3(oc, oc) - s.r * s.r;
//   float disc = b * b - c;
//   Hit h; h.hit = false; h.t = 1e30f;
//   if (disc >= 0.f) {
//     float t = -b - sqrtf(disc);
//     if (t > 1e-4f && t < h.t) {
//       h.t = t; h.hit = true;
//       h.p = add(ray.o, mul(ray.d, t));
//       h.n = normalize3(sub(h.p, s.c));
//       h.albedo = s.albedo;
//       h.reflect = s.reflect;
//     }
//   }
//   return h;
// }

// __device__ Hit hit_plane(const Ray& ray, const Plane& pl) {
//   Hit h; h.hit = false; h.t = 1e30f;
//   float denom = dot3(pl.n, ray.d);
//   if (fabsf(denom) > 1e-5f) {
//     float t = -(dot3(pl.n, ray.o) + pl.d) / denom;
//     if (t > 1e-4f && t < h.t) {
//       h.t = t; h.hit = true;
//       h.p = add(ray.o, mul(ray.d, t));
//       h.n = pl.n;
//       h.albedo = pl.albedo;
//       h.reflect = pl.reflect;
//     }
//   }
//   return h;
// }

// __device__ Hit scene_intersect(const Ray& r, const Sphere* spheres, int ns,
//                                 const Plane* planes, int np) {
//   Hit best; best.hit = false; best.t = 1e30f;
//   for (int i = 0; i < ns; i++) {
//     Hit h = hit_sphere(r, spheres[i]);
//     if (h.hit && h.t < best.t) best = h;
//   }
//   for (int i = 0; i < np; i++) {
//     Hit h = hit_plane(r, planes[i]);
//     if (h.hit && h.t < best.t) best = h;
//   }
//   return best;
// }

// // ============================================================================
// // Image Buffer Operations
// // ============================================================================
// __device__ inline void write_pixel(Pixel* img, int w, int x, int y, const float3& c) {
//   int idx = y * w + x;
//   img[idx].r = c.x; img[idx].g = c.y; img[idx].b = c.z;
// }

// __device__ inline void add_pixel(Pixel* img, int w, int x, int y, const float3& c) {
//   int idx = y * w + x;
//   // In this pipeline each pixel index is processed once per bounce pass.
//   // Using plain adds avoids heavy global atomic contention.
//   img[idx].r += c.x;
//   img[idx].g += c.y;
//   img[idx].b += c.z;
// }

// // ============================================================================
// // Tiling Constants
// // ============================================================================
// __constant__ int d_tiles_x, d_tiles_y, d_tile_w, d_tile_h;

// // ============================================================================
// // Kernels
// // ============================================================================

// // Generate primary rays (2D grid: blockIdx.x = tile_id, blockIdx.y = thread block within tile)
// __global__ void k_generate_primaries(
//     QUEUE_TYPE** qs, HANDLE_TYPE** hs,
//     Ray* rays, int img_w, int img_h,
//     float3 cam_o, float3 cam_f, float3 cam_r, float3 cam_u,
//     unsigned long long* enq_counts,
//     int num_threads_per_tile)
// {
//   int tile_id = (int)blockIdx.x;
//   int global_tid = (int)blockIdx.y * (int)blockDim.x + (int)threadIdx.x;
//   if (global_tid >= num_threads_per_tile) return;

//   QUEUE_TYPE* q = qs[tile_id];
//   HANDLE_TYPE* handles = hs[tile_id];
//   HANDLE_TYPE* h = &handles[global_tid];

//   int tx = tile_id % d_tiles_x;
//   int ty = tile_id / d_tiles_x;
//   int x0 = tx * d_tile_w;
//   int y0 = ty * d_tile_h;
//   int tw = min(d_tile_w, img_w - x0);
//   int th = min(d_tile_h, img_h - y0);
//   int tile_pixels = tw * th;

//   float aspect = (float)img_w / (float)img_h;

//   for (int i = global_tid; i < tile_pixels; i += num_threads_per_tile) {
//     int lx = i % tw, ly = i / tw;
//     int px = x0 + lx, py = y0 + ly;
//     int idx = py * img_w + px;

//     float u = ((px + 0.5f) / img_w - 0.5f) * 2.0f * aspect;
//     float v = -((py + 0.5f) / img_h - 0.5f) * 2.0f;

//     Ray r;
//     r.o = cam_o;
//     r.d = normalize3(add(cam_f, add(mul(cam_r, u), mul(cam_u, v))));
//     r.px = px; r.py = py; r.depth = 0;
//     r.attenuation = f3(1.0f, 1.0f, 1.0f);  // Start with full brightness
//     rays[idx] = r;

//     wf_enqueue(q, h, ENC_IDX(idx));
//   }

// #if HAS_LOCAL_BUF
//   if (h->local_buf.count > 0) wf_flush_local_buffer(q, h);
// #endif

//   if (global_tid == 0) atomicAdd(&enq_counts[tile_id], (unsigned long long)tile_pixels);
// }

// // Trace rays with multi-bounce support
// // Each bounce level uses a separate queue pair (input -> output)
// __global__ void k_trace_bounce(
//     QUEUE_TYPE** qs_in, HANDLE_TYPE** hs_in,
//     QUEUE_TYPE** qs_out, HANDLE_TYPE** hs_out,  // nullptr if last bounce
//     const Sphere* spheres, int ns,
//     const Plane* planes, int np,
//     Ray* rays_in,
//     Ray* rays_out,  // for next bounce (can be same as rays_in if ping-ponging)
//     Pixel* img, int img_w, int img_h,
//     const unsigned long long* expected,
//     unsigned long long* consumed,
//     unsigned long long* produced_next,  // nullptr if last bounce
//     int current_depth,
//     int max_depth,
//     int num_threads_per_tile)
// {
//   int tile_id = (int)blockIdx.x;
//   int global_tid = (int)blockIdx.y * (int)blockDim.x + (int)threadIdx.x;
//   if (global_tid >= num_threads_per_tile) return;

//   QUEUE_TYPE* q = qs_in[tile_id];
//   HANDLE_TYPE* h = &hs_in[tile_id][global_tid];

//   QUEUE_TYPE* q_out = qs_out ? qs_out[tile_id] : nullptr;
//   HANDLE_TYPE* h_out = (qs_out && hs_out) ? &hs_out[tile_id][global_tid] : nullptr;

//   const float3 light_dir = normalize3(f3(-0.5f, 1.0f, 0.2f));
//   const float3 ambient = f3(0.05f, 0.05f, 0.05f);

//   unsigned long long exp = expected[tile_id];

//   while (true) {
//     if (atomicAdd(&consumed[tile_id], 0ULL) >= exp) break;

//     uint64_t tok = wf_dequeue(q, h);
//     if (tok == QUEUE_EMPTY) continue;

//     int idx = DEC_TOK(tok);
//     Ray r = rays_in[idx];
//     Hit hit = scene_intersect(r, spheres, ns, planes, np);

//     float3 final_color = f3(0, 0, 0);

//     if (hit.hit) {
//       // Compute direct lighting
//       float NdotL = fmaxf(0.f, dot3(hit.n, light_dir));
//       float3 direct = add(ambient, mul(hit.albedo, NdotL));
      
//       // Apply attenuation and add to final color (contribution decreases with bounces)
//       float3 contribution = mul3(direct, r.attenuation);
      
//       // For reflective surfaces, we contribute a fraction now and reflect the rest
//       if (hit.reflect > 0.0f && current_depth < max_depth && q_out && h_out) {
//         // Non-reflective portion contributes immediately
//         float3 non_refl_contrib = mul(contribution, 1.0f - hit.reflect);
//         add_pixel(img, img_w, r.px, r.py, clamp01(non_refl_contrib));

//         // Create reflection ray
//         Ray refl;
//         refl.o = add(hit.p, mul(hit.n, 1e-4f));
//         refl.d = normalize3(reflect3(r.d, hit.n));
//         refl.px = r.px;
//         refl.py = r.py;
//         refl.depth = current_depth + 1;
//         // Attenuate by reflectivity and surface color
//         refl.attenuation = mul(mul3(r.attenuation, hit.albedo), hit.reflect);
//         rays_out[idx] = refl;

//         wf_enqueue(q_out, h_out, ENC_IDX(idx));
//         atomicAdd(&produced_next[tile_id], 1ULL);
//       } else {
//         // No more bounces: full contribution
//         add_pixel(img, img_w, r.px, r.py, clamp01(contribution));
//       }
//     } else {
//       // Sky gradient
//       float t = 0.5f * (r.d.y + 1.0f);
//       float3 sky = add(mul(f3(1, 1, 1), 1.0f - t), mul(f3(0.5f, 0.7f, 1.0f), t));
//       float3 contribution = mul3(sky, r.attenuation);
//       add_pixel(img, img_w, r.px, r.py, clamp01(contribution));
//     }

//     atomicAdd(&consumed[tile_id], 1ULL);
//   }

// #if HAS_LOCAL_BUF
//   if (h_out && h_out->local_buf.count > 0) wf_flush_local_buffer(q_out, h_out);
// #endif
// }

// // ============================================================================
// // Scene Builders
// // ============================================================================

// // Scene 0: Complex sphere field (same as existing)
// void build_scene_complex(std::vector<Sphere>& spheres, std::vector<Plane>& planes) {
//   // Ground plane
//   planes.push_back({f3(0, 1, 0), 1.0f, f3(0.75f, 0.75f, 0.75f), 0.05f});

//   // Grid of spheres
//   const int gx = 10, gz = 10;
//   const float spacing = 0.55f;
//   const float base_y = 0.25f;
//   const float radius = 0.22f;

//   for (int iz = 0; iz < gz; iz++) {
//     for (int ix = 0; ix < gx; ix++) {
//       float x = (ix - (gx - 1) * 0.5f) * spacing;
//       float z = -2.0f - iz * spacing;

//       int m = (ix + iz * 7) % 7;
//       float refl =
//           (m == 0) ? 0.9f :
//           (m == 1) ? 0.6f :
//           (m == 2) ? 0.3f :
//           (m == 3) ? 0.1f : 0.0f;

//       float3 alb =
//           (m == 0) ? f3(0.9f, 0.9f, 0.9f) :
//           (m == 1) ? f3(0.9f, 0.5f, 0.2f) :
//           (m == 2) ? f3(0.2f, 0.8f, 0.9f) :
//           (m == 3) ? f3(0.7f, 0.2f, 0.8f) :
//           (m == 4) ? f3(0.2f, 0.9f, 0.3f) :
//           (m == 5) ? f3(0.9f, 0.2f, 0.2f) : f3(0.8f, 0.8f, 0.3f);

//       spheres.push_back({f3(x, base_y, z), radius, alb, refl});
//     }
//   }

//   // Hero spheres
//   spheres.push_back({f3(-1.2f, 0.6f, -3.5f), 0.6f, f3(0.9f, 0.3f, 0.3f), 0.7f});
//   spheres.push_back({f3(1.4f, 0.9f, -4.2f), 0.9f, f3(0.3f, 0.9f, 0.4f), 0.2f});
//   spheres.push_back({f3(0.0f, 1.3f, -5.5f), 1.3f, f3(0.4f, 0.6f, 0.95f), 0.0f});
// }

// // Scene 1: Cornell Box style with reflective walls
// void build_scene_cornell(std::vector<Sphere>& spheres, std::vector<Plane>& planes) {
//   // Two main spheres (like original simple scene but more reflective)
//   // Left sphere: metallic red
//   spheres.push_back({f3(-0.6f, 0.5f, -2.5f), 0.5f, f3(0.9f, 0.3f, 0.3f), 0.85f});
//   // Right sphere: metallic silver/chrome
//   spheres.push_back({f3(0.6f, 0.6f, -2.8f), 0.6f, f3(0.95f, 0.95f, 0.95f), 0.95f});

//   // Floor - slightly reflective white
//   planes.push_back({f3(0, 1, 0), 1.0f, f3(0.9f, 0.9f, 0.9f), 0.3f});

//   // Back wall - metallic blue-gray (facing camera, normal pointing +Z)
//   planes.push_back({f3(0, 0, 1), 5.0f, f3(0.6f, 0.65f, 0.8f), 0.7f});

//   // Left wall - metallic red (normal pointing +X)
//   planes.push_back({f3(1, 0, 0), 2.0f, f3(0.8f, 0.2f, 0.2f), 0.6f});

//   // Right wall - metallic green (normal pointing -X)
//   planes.push_back({f3(-1, 0, 0), 2.0f, f3(0.2f, 0.8f, 0.3f), 0.6f});

//   // Optional: ceiling (slightly reflective white)
//   planes.push_back({f3(0, -1, 0), 2.5f, f3(0.95f, 0.95f, 0.95f), 0.2f});
// }

// // ============================================================================
// // Host Helpers
// // ============================================================================
// static void save_ppm(const std::string& path, const std::vector<Pixel>& img, int w, int h) {
//   FILE* f = fopen(path.c_str(), "wb");
//   if (!f) { perror("ppm"); return; }
//   fprintf(f, "P6\n%d %d\n255\n", w, h);
//   for (int i = 0; i < w * h; i++) {
//     unsigned char r = (unsigned char)std::min(255.f, std::max(0.f, img[i].r * 255.f));
//     unsigned char g = (unsigned char)std::min(255.f, std::max(0.f, img[i].g * 255.f));
//     unsigned char b = (unsigned char)std::min(255.f, std::max(0.f, img[i].b * 255.f));
//     fwrite(&r, 1, 1, f); fwrite(&g, 1, 1, f); fwrite(&b, 1, 1, f);
//   }
//   fclose(f);
// }

// std::string getGpu(const char* name) {
//   std::string full = std::string(name);
//   size_t pos = full.find_last_of(" ");
//   return (pos != std::string::npos && pos + 1 < full.size()) ? full.substr(pos + 1) : full;
// }

// // ============================================================================
// // Main
// // ============================================================================
// int main(int argc, char** argv) {
//   hipDeviceProp_t prop{};
//   HIP_CHECK(hipGetDeviceProperties(&prop, 0));
//   std::cout << "Wavefront Raytracer FINAL on " << prop.name << " using " << QUEUE_NAME << "\n";

//   // Parse arguments
//   int NUM_THREADS = (argc > 1) ? atoi(argv[1]) : 1024;
//   int scene_id = (argc > 2) ? atoi(argv[2]) : 0;
//   int MAX_BOUNCES = (argc > 3) ? atoi(argv[3]) : 4;

//   std::cout << "Threads per tile: " << NUM_THREADS << "\n";
//   std::cout << "Scene: " << scene_id << " (" << (scene_id == 0 ? "Complex Spheres" : "Cornell Box") << ")\n";
//   std::cout << "Max bounces: " << MAX_BOUNCES << "\n";

//   // Image and tiling setup
//   const int W = 1280, H = 720;
//   const int TILES_X = 8, TILES_Y = 8;
//   const int NUM_TILES = TILES_X * TILES_Y;
//   const int TILE_W = (W + TILES_X - 1) / TILES_X;
//   const int TILE_H = (H + TILES_Y - 1) / TILES_Y;

//   HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(d_tiles_x), &TILES_X, sizeof(int)));
//   HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(d_tiles_y), &TILES_Y, sizeof(int)));
//   HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(d_tile_w), &TILE_W, sizeof(int)));
//   HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(d_tile_h), &TILE_H, sizeof(int)));

//   const int TPB = 256;
//   const int NUM_BLOCKS_Y = (NUM_THREADS + TPB - 1) / TPB;

//   // Build scene
//   std::vector<Sphere> h_spheres;
//   std::vector<Plane> h_planes;

//   if (scene_id == 0) {
//     build_scene_complex(h_spheres, h_planes);
//   } else {
//     build_scene_cornell(h_spheres, h_planes);
//   }

//   int ns = (int)h_spheres.size();
//   int np = (int)h_planes.size();
//   std::cout << "Scene: " << ns << " spheres, " << np << " planes\n";

//   // Upload scene to device
//   Sphere* d_spheres = nullptr;
//   Plane* d_planes = nullptr;
//   HIP_CHECK(hipMalloc(&d_spheres, h_spheres.size() * sizeof(Sphere)));
//   HIP_CHECK(hipMalloc(&d_planes, h_planes.size() * sizeof(Plane)));
//   HIP_CHECK(hipMemcpy(d_spheres, h_spheres.data(), h_spheres.size() * sizeof(Sphere), hipMemcpyHostToDevice));
//   HIP_CHECK(hipMemcpy(d_planes, h_planes.data(), h_planes.size() * sizeof(Plane), hipMemcpyHostToDevice));

//   // Allocate queues/handles per bounce level.
//   // Some backends do not support safe ping-pong reuse without explicit reset.
//   const int NUM_QUEUE_SETS = MAX_BOUNCES + 1;
  
//   std::vector<std::vector<QUEUE_TYPE*>> h_queues(NUM_QUEUE_SETS, std::vector<QUEUE_TYPE*>(NUM_TILES, nullptr));
//   std::vector<std::vector<HANDLE_TYPE*>> h_handles(NUM_QUEUE_SETS, std::vector<HANDLE_TYPE*>(NUM_TILES, nullptr));
  
// #ifdef NEEDS_RECORDS
//   std::vector<std::vector<wf_thread_record*>> h_records(NUM_QUEUE_SETS, std::vector<wf_thread_record*>(NUM_TILES, nullptr));
// #endif

//   for (int level = 0; level < NUM_QUEUE_SETS; level++) {
//     for (int t = 0; t < NUM_TILES; t++) {
// #ifdef NEEDS_RECORDS
//       wf_queue_host_init(&h_queues[level][t], &h_handles[level][t], &h_records[level][t], NUM_THREADS);
// #else
//       wf_queue_host_init(&h_queues[level][t], &h_handles[level][t], NUM_THREADS);
// #endif
//     }
//   }

//   // Device pointer arrays for each level
//   std::vector<QUEUE_TYPE**> d_queues(NUM_QUEUE_SETS);
//   std::vector<HANDLE_TYPE**> d_handles(NUM_QUEUE_SETS);

//   for (int level = 0; level < NUM_QUEUE_SETS; level++) {
//     HIP_CHECK(hipMalloc(&d_queues[level], NUM_TILES * sizeof(QUEUE_TYPE*)));
//     HIP_CHECK(hipMalloc(&d_handles[level], NUM_TILES * sizeof(HANDLE_TYPE*)));
//     HIP_CHECK(hipMemcpy(d_queues[level], h_queues[level].data(), NUM_TILES * sizeof(QUEUE_TYPE*), hipMemcpyHostToDevice));
//     HIP_CHECK(hipMemcpy(d_handles[level], h_handles[level].data(), NUM_TILES * sizeof(HANDLE_TYPE*), hipMemcpyHostToDevice));
//   }

//   // Rays (ping-pong buffers for bounces)
//   Ray *d_rays0 = nullptr, *d_rays1 = nullptr;
//   Pixel* d_img = nullptr;
//   HIP_CHECK(hipMalloc(&d_rays0, W * H * sizeof(Ray)));
//   HIP_CHECK(hipMalloc(&d_rays1, W * H * sizeof(Ray)));
//   HIP_CHECK(hipMalloc(&d_img, W * H * sizeof(Pixel)));
//   HIP_CHECK(hipMemset(d_img, 0, W * H * sizeof(Pixel)));

//   // Per-tile counters for each level
//   std::vector<unsigned long long*> d_expected(NUM_QUEUE_SETS);
//   std::vector<unsigned long long*> d_consumed(NUM_QUEUE_SETS);

//   for (int level = 0; level < NUM_QUEUE_SETS; level++) {
//     HIP_CHECK(hipMalloc(&d_expected[level], NUM_TILES * sizeof(unsigned long long)));
//     HIP_CHECK(hipMalloc(&d_consumed[level], NUM_TILES * sizeof(unsigned long long)));
//     HIP_CHECK(hipMemset(d_expected[level], 0, NUM_TILES * sizeof(unsigned long long)));
//     HIP_CHECK(hipMemset(d_consumed[level], 0, NUM_TILES * sizeof(unsigned long long)));
//   }

//   // Camera setup (different for each scene)
//   float3 cam_o, cam_f, cam_r, cam_u;
//   if (scene_id == 0) {
//     cam_o = f3(0, 0.3f, 0.5f);
//     cam_f = normalize3(f3(0, -0.2f, -1.0f));
//     cam_r = normalize3(f3(1, 0, 0));
//     cam_u = normalize3(f3(0, 1, 0));
//   } else {
//     // Cornell box view - slightly above and back to see the room
//     cam_o = f3(0, 0.5f, 1.5f);
//     cam_f = normalize3(f3(0, -0.1f, -1.0f));
//     cam_r = normalize3(f3(1, 0, 0));
//     cam_u = normalize3(f3(0, 1, 0));
//   }

//   // Timing
//   hipEvent_t e0, e1;
//   HIP_CHECK(hipEventCreate(&e0));
//   HIP_CHECK(hipEventCreate(&e1));

//   dim3 grid0(NUM_TILES, NUM_BLOCKS_Y, 1);
//   dim3 block(TPB, 1, 1);

//   float total_time_ms = 0.0f;
//   unsigned long long total_rays = 0;

//   // ============================================================================
//   // PASS 0: Generate primary rays
//   // ============================================================================
//   std::cout << "\nPASS 0: Generating primary rays...\n";
//   HIP_CHECK(hipEventRecord(e0));

//   k_generate_primaries<<<grid0, block>>>(
//       d_queues[0], d_handles[0],
//       d_rays0, W, H,
//       cam_o, cam_f, cam_r, cam_u,
//       d_expected[0],
//       NUM_THREADS);

//   HIP_CHECK(hipEventRecord(e1));
//   HIP_CHECK(hipEventSynchronize(e1));
//   float ms_gen = 0;
//   HIP_CHECK(hipEventElapsedTime(&ms_gen, e0, e1));
//   HIP_LAUNCH_CHECK();
//   std::cout << "  Pass 0 time: " << ms_gen << " ms\n";
//   total_time_ms += ms_gen;

//   // Get primary ray count
//   std::vector<unsigned long long> h_expected(NUM_TILES);
//   HIP_CHECK(hipMemcpy(h_expected.data(), d_expected[0], NUM_TILES * sizeof(unsigned long long), hipMemcpyDeviceToHost));
  
//   unsigned long long primary_rays = 0;
//   for (int t = 0; t < NUM_TILES; t++) primary_rays += h_expected[t];
//   std::cout << "  Generated " << primary_rays << " primary rays\n";
//   total_rays += primary_rays;

//   // ============================================================================
//   // Multi-bounce tracing passes
//   // ============================================================================
//   Ray* rays_in = d_rays0;
//   Ray* rays_out = d_rays1;

//   for (int bounce = 0; bounce <= MAX_BOUNCES; bounce++) {
//     // Check if there's any work for this bounce level
//     HIP_CHECK(hipMemcpy(h_expected.data(), d_expected[bounce], NUM_TILES * sizeof(unsigned long long), hipMemcpyDeviceToHost));
    
//     unsigned long long rays_this_level = 0;
//     for (int t = 0; t < NUM_TILES; t++) rays_this_level += h_expected[t];
    
//     if (rays_this_level == 0) {
//       std::cout << "PASS " << (bounce + 1) << ": No rays to trace (done)\n";
//       break;
//     }

//     std::cout << "PASS " << (bounce + 1) << ": Tracing " << rays_this_level 
//               << " rays (bounce " << bounce << ")...\n";

//     // Determine if there's a next level
//     bool has_next_level = (bounce < MAX_BOUNCES);
//     QUEUE_TYPE** qs_out = has_next_level ? d_queues[bounce + 1] : nullptr;
//     HANDLE_TYPE** hs_out = has_next_level ? d_handles[bounce + 1] : nullptr;
//     unsigned long long* produced_next = has_next_level ? d_expected[bounce + 1] : nullptr;

//     // Reset consumed counter for this level
//     HIP_CHECK(hipMemset(d_consumed[bounce], 0, NUM_TILES * sizeof(unsigned long long)));
    
//     // Reset expected counter for next level if there is one
//     if (has_next_level) {
//       HIP_CHECK(hipMemset(d_expected[bounce + 1], 0, NUM_TILES * sizeof(unsigned long long)));
//     }

//     HIP_CHECK(hipEventRecord(e0));

//     k_trace_bounce<<<grid0, block>>>(
//       d_queues[bounce], d_handles[bounce],
//         qs_out, hs_out,
//         d_spheres, ns,
//         d_planes, np,
//         rays_in, rays_out,
//         d_img, W, H,
//       d_expected[bounce],
//       d_consumed[bounce],
//         produced_next,
//         bounce,
//         MAX_BOUNCES,
//         NUM_THREADS);

//     HIP_CHECK(hipEventRecord(e1));
//     HIP_CHECK(hipEventSynchronize(e1));
//     float ms_bounce = 0;
//     HIP_CHECK(hipEventElapsedTime(&ms_bounce, e0, e1));
//     HIP_LAUNCH_CHECK();
//     std::cout << "  Pass " << (bounce + 1) << " time: " << ms_bounce << " ms\n";
//     total_time_ms += ms_bounce;

//     // Count rays for next level
//     if (has_next_level) {
//       HIP_CHECK(hipMemcpy(h_expected.data(), d_expected[bounce + 1], NUM_TILES * sizeof(unsigned long long), hipMemcpyDeviceToHost));
//       unsigned long long next_rays = 0;
//       for (int t = 0; t < NUM_TILES; t++) next_rays += h_expected[t];
//       if (next_rays > 0) {
//         total_rays += next_rays;
//       }
//     }

//     // Swap ray buffers for ping-pong
//     std::swap(rays_in, rays_out);
//   }

//   // ============================================================================
//   // Results
//   // ============================================================================
//   double mrays = (total_time_ms > 0.0) ? ((double)total_rays / 1.0e6) / (total_time_ms / 1.0e3) : 0.0;

//   std::cout << "\n=== FINAL RESULTS ===\n";
//   std::cout << "Total rays traced: " << total_rays << "\n";
//   std::cout << "Total time: " << total_time_ms << " ms\n";
//   std::cout << "Throughput: " << mrays << " MRays/s\n";

//   // Save image
//   std::vector<Pixel> h_img(W * H);
//   HIP_CHECK(hipMemcpy(h_img.data(), d_img, W * H * sizeof(Pixel), hipMemcpyDeviceToHost));

//   std::string scene_name = (scene_id == 0) ? "complex" : "cornell";
//   std::string filename = "rt_final_" + scene_name + "_" + QUEUE_NAME + ".ppm";
//   save_ppm(filename, h_img, W, H);
//   std::cout << "Image saved: " << filename << "\n";

//   // Write CSV results
//   const char* csv_path = "rt_final_performance.csv";
//   std::string DEVICE = getGpu(prop.name);

//   bool file_exists = static_cast<bool>(std::ifstream(csv_path));
//   std::ofstream csv(csv_path, std::ios::app);
//   if (csv) {
//     if (!file_exists) {
//       csv << "DEVICE,QUEUE,SCENE,THREADS,MAX_BOUNCES,TOTAL_RAYS,TOTAL_TIME_MS,MRAYS_PER_S\n";
//     }
//     csv << DEVICE << ","
//         << QUEUE_NAME << ","
//         << scene_name << ","
//         << NUM_THREADS << ","
//         << MAX_BOUNCES << ","
//         << total_rays << ","
//         << total_time_ms << ","
//         << mrays << "\n";
//   }

//   // ============================================================================
//   // Cleanup
//   // ============================================================================
//   for (int level = 0; level < NUM_QUEUE_SETS; level++) {
//     HIP_CHECK(hipFree(d_queues[level]));
//     HIP_CHECK(hipFree(d_handles[level]));
//     HIP_CHECK(hipFree(d_expected[level]));
//     HIP_CHECK(hipFree(d_consumed[level]));

//     for (int t = 0; t < NUM_TILES; t++) {
// #ifdef USE_SFQ
//       hipFree(h_queues[level][t]);
//       hipFree(h_handles[level][t]);
// #else
//   #ifdef NEEDS_RECORDS
//       wf_queue_destroy(h_queues[level][t], h_handles[level][t]);
//   #else
//       hipFree(h_queues[level][t]);
//       hipFree(h_handles[level][t]);
//   #endif
// #endif
//     }
//   }

//   HIP_CHECK(hipFree(d_rays0));
//   HIP_CHECK(hipFree(d_rays1));
//   HIP_CHECK(hipFree(d_img));
//   HIP_CHECK(hipFree(d_spheres));
//   HIP_CHECK(hipFree(d_planes));
//   HIP_CHECK(hipEventDestroy(e0));
//   HIP_CHECK(hipEventDestroy(e1));

//   return 0;
// }


// raytrace_final.cpp - Persistent-megakernel wavefront ray tracer
// 
// Key design: ONE queue per tile, ONE trace kernel launch for ALL bounces.
// Reflection rays re-enqueue into the same queue they were dequeued from.
// Threads persistently consume work until all rays terminate.
// This is the correct way to use concurrent queues — zero host sync between bounces.
//
// Scene 0: Complex sphere field (100+ spheres with varied materials)
// Scene 1: Cornell Box style - 2 spheres + floor + 3 reflective walls, multiple bounces
//
// Compile examples:
//   hipcc -O3 raytrace_final.cpp -o rt_wfq
//   hipcc -O3 raytrace_final.cpp -DUSE_CAWFQ -o rt_cawfq
//   hipcc -O3 raytrace_final.cpp -DUSE_BWFQ -o rt_bwfq
//   hipcc -O3 raytrace_final.cpp -DUSE_SFQ -o rt_sfq
//
// Usage: ./rt_wfq [threads] [scene] [max_bounces]
//   threads: number of threads per tile (default: 1024)
//   scene: 0 = complex spheres, 1 = cornell box (default: 0)
//   max_bounces: reflection depth for scene 1 (default: 4)

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

#define HIP_CHECK(call) do {                                 \
  hipError_t _e = (call);                                    \
  if (_e != hipSuccess) {                                    \
    fprintf(stderr, "HIP error %s:%d: %s\n",                \
            __FILE__, __LINE__, hipGetErrorString(_e));      \
    exit(1);                                                 \
  }                                                          \
} while(0)

#define HIP_LAUNCH_CHECK() do {                              \
  HIP_CHECK(hipGetLastError());                              \
  HIP_CHECK(hipDeviceSynchronize());                         \
} while(0)

// ============================================================================
// Queue Sizing
// ============================================================================
// Per-tile pixel count is small (~14K for 8x8 tiles at 1280x720).
// Bounded queues only need capacity >= max concurrent live items.
// Linked queues need enough segments for total lifetime operations.
#ifndef RT_QUEUE_RING_CAPACITY
// #define RT_QUEUE_RING_CAPACITY 65536ULL
#define RT_QUEUE_RING_CAPACITY 262144ULL
#endif

#ifndef RT_QUEUE_SEGMENT_SAFETY
#define RT_QUEUE_SEGMENT_SAFETY 4
#endif

// ============================================================================
// Queue Selection
// ============================================================================
#ifdef USE_CAWFQ
  #include "../wfqueue_cawfq.hpp"
  #include "../wfqueue_cawfq.cpp"
  #define QUEUE_TYPE    wf_queue
  #define HANDLE_TYPE   wf_handle
  #define QUEUE_EMPTY   WF_EMPTY
  #define QUEUE_NAME    "CAWFQ"
  #define HAS_LOCAL_BUF 1
#elif defined(USE_BLFQ)
  #ifndef CAWFQ_PREALLOC_OPS_PER_THREAD
    #define CAWFQ_PREALLOC_OPS_PER_THREAD 32
  #endif
  #ifndef CAWFQ_SEGMENT_SAFETY
    #define CAWFQ_SEGMENT_SAFETY RT_QUEUE_SEGMENT_SAFETY
  #endif
  #include "../wfqueue_cawfq2.hpp"
  #define QUEUE_TYPE    wf_queue
  #define HANDLE_TYPE   wf_handle
  #define QUEUE_EMPTY   WF_EMPTY
  #define QUEUE_NAME    "BLFQ-WarpBatch"
  #define HAS_LOCAL_BUF 0
  #define BLFQ_STATUS_API 1
#elif defined(USE_BWFQ)
  #ifndef WF_RING_CAPACITY
    #define WF_RING_CAPACITY RT_QUEUE_RING_CAPACITY
  #endif
  #define WF_TID ((uint32_t)(blockIdx.y * blockDim.x + threadIdx.x))
  #define WF_ENQ_RET_BOOL 1
  #include "../wfqueue_bounded.hpp"`
  #define QUEUE_TYPE    wf_queue
  #define HANDLE_TYPE   wf_handle
  #define QUEUE_EMPTY   WF_EMPTY
  #define QUEUE_NAME    "BWFQ"
  #define HAS_LOCAL_BUF 0
  #define NEEDS_RECORDS 1
#elif defined(USE_BWFQ2)
// BWFQ CODE THAT IS FLAWED BOUNDED PROOF
  #ifndef WF_RING_CAPACITY
    #define WF_RING_CAPACITY RT_QUEUE_RING_CAPACITY
  #endif
  #define WF_TID ((uint32_t)(blockIdx.y * blockDim.x + threadIdx.x))
  #define WF_ENQ_RET_BOOL 1
  #include "../wfqueue_bwfq.hpp"
  // #define WF_TID ((uint32_t)(blockIdx.y * blockDim.x + threadIdx.x))
  #define QUEUE_TYPE    wf_queue
  #define HANDLE_TYPE   wf_handle
  #define QUEUE_EMPTY   WF_EMPTY
  #define QUEUE_NAME    "BWFQ2"
  #define HAS_LOCAL_BUF 0
  #define NEEDS_RECORDS 1
#elif defined(USE_BWFREE)
// MAIN CODE THAT IS BOUNDED PROOF for BWFQ wait freedom
  #ifndef WF_RING_CAPACITY
    #define WF_RING_CAPACITY RT_QUEUE_RING_CAPACITY
  #endif
  #define WFQ_TID ((uint32_t)(blockIdx.y * blockDim.x + threadIdx.x))
  #include "../bwfq_wf.hpp"

  #define QUEUE_TYPE    wfq_queue
  #define HANDLE_TYPE   wfq_handle
  #define QUEUE_EMPTY   WFQ_EMPTY_INDEX
  #define QUEUE_NAME    "WFQ64"
  #define HAS_LOCAL_BUF 0
  #define NEEDS_RECORDS 1

  // generic RT app API -> wfq64 API
  __device__ __forceinline__
  bool wf_enqueue(QUEUE_TYPE* q, HANDLE_TYPE* h, uint64_t val) {
    wfq_enqueue(q, h, (uint32_t)val);
    return true;
  }

  __device__ __forceinline__
  uint64_t wf_dequeue(QUEUE_TYPE* q, HANDLE_TYPE* h) {
    return (uint64_t)wfq_dequeue(q, h);
  }
  #define wf_queue_host_init   wfq_queue_host_init
  #define wf_queue_destroy     wfq_queue_destroy
#elif defined(USE_SFQ)
  #include "../sfqueue_hip.hpp"
  #include "../sfqueue_hip.cpp"
  #define QUEUE_TYPE    sfq_queue
  #define HANDLE_TYPE   sfq_handle
  #define QUEUE_EMPTY   SFQ_EMPTY
  #define QUEUE_NAME    "SFQ"
  #define HAS_LOCAL_BUF 0
  #define wf_enqueue    sfq_enqueue
  #define wf_dequeue    sfq_dequeue
  #define wf_queue_host_init sfq_queue_host_init
#elif defined(USE_WLQ)
  #include "rt_qshim.hpp"
#else
  // Default: WFQ
  #include "../wfqueue_hip_opt.hpp"
  #define QUEUE_TYPE    wf_queue
  #define HANDLE_TYPE   wf_handle
  #define QUEUE_EMPTY   WF_EMPTY
  #define QUEUE_NAME    "WFQ"
  #define HAS_LOCAL_BUF 0
#endif

// Token encoding: never enqueue 0
#define ENC_IDX(i)  ((uint64_t)((i) + 1))
#define DEC_TOK(t)  ((int)((t) - 1))

#ifndef BLFQ_STATUS_API
#define BLFQ_STATUS_API 0
#endif


// ============================================================================
// Math Helpers
// ============================================================================
__host__ __device__ inline float3 f3(float x, float y, float z) { return make_float3(x, y, z); }
__host__ __device__ inline float3 add(const float3& a, const float3& b) { return f3(a.x+b.x, a.y+b.y, a.z+b.z); }
__host__ __device__ inline float3 sub(const float3& a, const float3& b) { return f3(a.x-b.x, a.y-b.y, a.z-b.z); }
__host__ __device__ inline float3 mul(const float3& a, float s) { return f3(a.x*s, a.y*s, a.z*s); }
__host__ __device__ inline float3 mul3(const float3& a, const float3& b) { return f3(a.x*b.x, a.y*b.y, a.z*b.z); }
__host__ __device__ inline float  dot3(const float3& a, const float3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
__host__ __device__ inline float3 normalize3(const float3& v) {
  float d = sqrtf(dot3(v, v));
  return d > 0 ? mul(v, 1.0f/d) : v;
}
__host__ __device__ inline float3 reflect3(const float3& v, const float3& n) {
  return sub(v, mul(n, 2.f * dot3(v, n)));
}
__host__ __device__ inline float3 clamp01(const float3& c) {
  return f3(fminf(fmaxf(c.x, 0.f), 1.f),
            fminf(fmaxf(c.y, 0.f), 1.f),
            fminf(fmaxf(c.z, 0.f), 1.f));
}

// ============================================================================
// Scene Structures
// ============================================================================
struct Sphere { float3 c; float r; float3 albedo; float reflect; };
struct Plane  { float3 n; float d; float3 albedo; float reflect; };

struct Ray {
  float3 o, d;
  int px, py;
  int depth;
  float3 attenuation;
};

struct Hit {
  float t;
  float3 n;
  float3 p;
  float3 albedo;
  float reflect;
  bool hit;
};

struct Pixel { float r, g, b; };

// ============================================================================
// Intersection Tests
// ============================================================================
__device__ Hit hit_sphere(const Ray& ray, const Sphere& s) {
  float3 oc = sub(ray.o, s.c);
  float b = dot3(oc, ray.d);
  float c = dot3(oc, oc) - s.r * s.r;
  float disc = b * b - c;
  Hit h; h.hit = false; h.t = 1e30f;
  if (disc >= 0.f) {
    float t = -b - sqrtf(disc);
    if (t > 1e-4f && t < h.t) {
      h.t = t; h.hit = true;
      h.p = add(ray.o, mul(ray.d, t));
      h.n = normalize3(sub(h.p, s.c));
      h.albedo = s.albedo;
      h.reflect = s.reflect;
    }
  }
  return h;
}

__device__ Hit hit_plane(const Ray& ray, const Plane& pl) {
  Hit h; h.hit = false; h.t = 1e30f;
  float denom = dot3(pl.n, ray.d);
  if (fabsf(denom) > 1e-5f) {
    float t = -(dot3(pl.n, ray.o) + pl.d) / denom;
    if (t > 1e-4f && t < h.t) {
      h.t = t; h.hit = true;
      h.p = add(ray.o, mul(ray.d, t));
      h.n = pl.n;
      h.albedo = pl.albedo;
      h.reflect = pl.reflect;
    }
  }
  return h;
}

__device__ Hit scene_intersect(const Ray& r, const Sphere* spheres, int ns,
                                const Plane* planes, int np) {
  Hit best; best.hit = false; best.t = 1e30f;
  for (int i = 0; i < ns; i++) {
    Hit h = hit_sphere(r, spheres[i]);
    if (h.hit && h.t < best.t) best = h;
  }
  for (int i = 0; i < np; i++) {
    Hit h = hit_plane(r, planes[i]);
    if (h.hit && h.t < best.t) best = h;
  }
  return best;
}

// ============================================================================
// Atomic pixel accumulation (needed for persistent kernel - multiple bounces
// can write the same pixel concurrently from different threads)
// ============================================================================
__device__ inline void atomic_add_pixel(Pixel* img, int w, int x, int y, const float3& c) {
  int idx = y * w + x;
  atomicAdd(&img[idx].r, c.x);
  atomicAdd(&img[idx].g, c.y);
  atomicAdd(&img[idx].b, c.z);
}

// ============================================================================
// Tiling Constants
// ============================================================================
__constant__ int d_tiles_x, d_tiles_y, d_tile_w, d_tile_h;

// ============================================================================
// Kernel 1: Generate primary rays and enqueue into per-tile queues
// ============================================================================
__global__ void k_generate_primaries(
    QUEUE_TYPE** qs, HANDLE_TYPE** hs,
    Ray* rays, int img_w, int img_h,
    float3 cam_o, float3 cam_f, float3 cam_r, float3 cam_u,
    unsigned long long* rays_alive,   // per-tile live ray counter
    int num_threads_per_tile)
{
  int tile_id = (int)blockIdx.x;
  int global_tid = (int)blockIdx.y * (int)blockDim.x + (int)threadIdx.x;
  if (global_tid >= num_threads_per_tile) return;

  QUEUE_TYPE* q = qs[tile_id];
  HANDLE_TYPE* h = &hs[tile_id][global_tid];

  int tx = tile_id % d_tiles_x;
  int ty = tile_id / d_tiles_x;
  int x0 = tx * d_tile_w;
  int y0 = ty * d_tile_h;
  int tw = min(d_tile_w, img_w - x0);
  int th = min(d_tile_h, img_h - y0);
  int tile_pixels = tw * th;

  float aspect = (float)img_w / (float)img_h;

  int my_count = 0;
  int fail_count = 0;
  for (int i = global_tid; i < tile_pixels; i += num_threads_per_tile) {
    int lx = i % tw, ly = i / tw;
    int px = x0 + lx, py = y0 + ly;
    int idx = py * img_w + px;

    float u = ((px + 0.5f) / img_w - 0.5f) * 2.0f * aspect;
    float v = -((py + 0.5f) / img_h - 0.5f) * 2.0f;

    Ray r;
    r.o = cam_o;
    r.d = normalize3(add(cam_f, add(mul(cam_r, u), mul(cam_u, v))));
    r.px = px; r.py = py; r.depth = 0;
    r.attenuation = f3(1.0f, 1.0f, 1.0f);
    rays[idx] = r;

    // wf_enqueue(q, h, ENC_IDX(idx));
    // my_count++;
    // bool ok = wf_enqueue(q, h, ENC_IDX(idx));
    // if (ok) {
    //   my_count++;
    // } else {
    //   fail_count++;
    // }

    #if BLFQ_STATUS_API
      int enq_status = wf_enqueue(q, h, ENC_IDX(idx));
      bool ok = (enq_status == WF_OK);
    #else
      bool ok = wf_enqueue(q, h, ENC_IDX(idx));
    #endif

    if (ok) {
      my_count++;
    } else {
      fail_count++;
    }
  }

#if HAS_LOCAL_BUF
  if (h->local_buf.count > 0) wf_flush_local_buffer(q, h);
#endif

  // Contribute this thread's count to tile-level rays_alive
  if (my_count > 0)
    atomicAdd(&rays_alive[tile_id], (unsigned long long)my_count);
}

// ============================================================================
// Kernel 2: Persistent trace - one launch, all bounces, single queue per tile
//
// Each thread loops: dequeue ray, trace, shade. If reflective and depth < max,
// write reflection ray back to same slot, re-enqueue into SAME queue.
// Terminates when rays_alive[tile_id] == 0.
// ============================================================================
__global__ void k_trace_persistent(
    QUEUE_TYPE** qs, HANDLE_TYPE** hs,
    const Sphere* spheres, int ns,
    const Plane* planes, int np,
    Ray* rays,
    Pixel* img, int img_w, int img_h,
    unsigned long long* rays_alive,  // per-tile: decremented on terminal rays
    unsigned long long* total_rays_traced, // per-tile: total work done (for stats)
    int max_depth,
    int num_threads_per_tile)
{
  int tile_id = (int)blockIdx.x;
  int global_tid = (int)blockIdx.y * (int)blockDim.x + (int)threadIdx.x;
  if (global_tid >= num_threads_per_tile) return;

  QUEUE_TYPE* q = qs[tile_id];
  HANDLE_TYPE* h = &hs[tile_id][global_tid];

  const float3 light_dir = normalize3(f3(-0.5f, 1.0f, 0.2f));
  const float3 ambient = f3(0.05f, 0.05f, 0.05f);

  unsigned long long my_traced = 0;
  int empty_spins = 0;

  while (true) {
    // ---- Termination check ----
    // All rays have been fully resolved when rays_alive reaches 0.
    // Use relaxed atomic load (atomicAdd +0) for coherent read.
    unsigned long long alive = atomicAdd(&rays_alive[tile_id], 0ULL);
    if (alive == 0) break;

    // ---- Dequeue ----
    // uint64_t tok = wf_dequeue(q, h);
    uint64_t tok = QUEUE_EMPTY;

    #if BLFQ_STATUS_API
      int dq_status = wf_dequeue(q, h, &tok);
      if (dq_status != WF_OK || tok == QUEUE_EMPTY) {
    #else
      tok = wf_dequeue(q, h);
      if (tok == QUEUE_EMPTY) {
    #endif
        empty_spins++;
        if (empty_spins > 64) {
          #if defined(__AMDGCN__)
            __builtin_amdgcn_s_sleep(2);
          #endif
        }
        continue;
      }
    // if (tok == QUEUE_EMPTY) {
    //   // Backoff to reduce contention on empty queue.
    //   // Rays might be in-flight (being processed by other threads).
    //   empty_spins++;
    //   if (empty_spins > 64) {
    //     // Heavy backoff: yield execution resources
    //     #if defined(__AMDGCN__)
    //       __builtin_amdgcn_s_sleep(2);
    //     #endif
    //   }
    //   continue;
    // }
    empty_spins = 0;

    int idx = DEC_TOK(tok);
    Ray r = rays[idx];
    Hit hit = scene_intersect(r, spheres, ns, planes, np);

    if (hit.hit) {
      // Direct lighting
      float NdotL = fmaxf(0.f, dot3(hit.n, light_dir));
      float3 direct = add(ambient, mul(hit.albedo, NdotL));
      float3 contribution = mul3(direct, r.attenuation);

      if (hit.reflect > 0.0f && r.depth < max_depth) {
        // ---- Reflective surface: split contribution ----
        // Non-reflective portion contributes immediately
        float3 non_refl = mul(contribution, 1.0f - hit.reflect);
        float3 clamped = clamp01(non_refl);
        if (clamped.x > 0.f || clamped.y > 0.f || clamped.z > 0.f)
          atomic_add_pixel(img, img_w, r.px, r.py, clamped);

        // Build reflection ray, write to same slot, re-enqueue
        Ray refl;
        refl.o = add(hit.p, mul(hit.n, 1e-4f));
        refl.d = normalize3(reflect3(r.d, hit.n));
        refl.px = r.px;
        refl.py = r.py;
        refl.depth = r.depth + 1;
        refl.attenuation = mul(mul3(r.attenuation, hit.albedo), hit.reflect);
        rays[idx] = refl;

        // bool ok = wf_enqueue(q, h, ENC_IDX(idx));
        #if BLFQ_STATUS_API
          int enq_status = wf_enqueue(q, h, ENC_IDX(idx));
          bool ok = (enq_status == WF_OK);
        #else
          bool ok = wf_enqueue(q, h, ENC_IDX(idx));
        #endif

        

#if HAS_LOCAL_BUF
        // CAWFQ: must flush immediately in persistent kernel so other
        // threads can dequeue this work. Batching across iterations is
        // unsafe — could starve consumers and break termination.
        if (h->local_buf.count > 0) wf_flush_local_buffer(q, h);
#endif

        if (!ok) {
  // We failed to re-enqueue a continuation ray.
  // To preserve termination, treat it as terminal.
  atomicAdd(&rays_alive[tile_id], (unsigned long long)(-1ULL));
}
        // rays_alive unchanged: consumed one, produced one (net zero)
      } else {
        // ---- Terminal: no reflection or max depth ----
        atomic_add_pixel(img, img_w, r.px, r.py, clamp01(contribution));
        atomicAdd(&rays_alive[tile_id], (unsigned long long)(-1ULL)); // decrement
      }
    } else {
      // ---- Sky miss: terminal ray ----
      float t = 0.5f * (r.d.y + 1.0f);
      float3 sky = add(mul(f3(1, 1, 1), 1.0f - t), mul(f3(0.5f, 0.7f, 1.0f), t));
      float3 contribution = mul3(sky, r.attenuation);
      atomic_add_pixel(img, img_w, r.px, r.py, clamp01(contribution));
      atomicAdd(&rays_alive[tile_id], (unsigned long long)(-1ULL)); // decrement
    }

    my_traced++;
  }

  // Accumulate per-thread stats
  if (my_traced > 0)
    atomicAdd(&total_rays_traced[tile_id], my_traced);
}

// ============================================================================
// Scene Builders
// ============================================================================

// Scene 0: Complex sphere field
void build_scene_complex(std::vector<Sphere>& spheres, std::vector<Plane>& planes) {
  planes.push_back({f3(0, 1, 0), 1.0f, f3(0.75f, 0.75f, 0.75f), 0.05f});

  const int gx = 10, gz = 10;
  const float spacing = 0.55f;
  const float base_y = 0.25f;
  const float radius = 0.22f;

  for (int iz = 0; iz < gz; iz++) {
    for (int ix = 0; ix < gx; ix++) {
      float x = (ix - (gx - 1) * 0.5f) * spacing;
      float z = -2.0f - iz * spacing;

      int m = (ix + iz * 7) % 7;
      float refl =
          (m == 0) ? 0.9f :
          (m == 1) ? 0.6f :
          (m == 2) ? 0.3f :
          (m == 3) ? 0.1f : 0.0f;

      float3 alb =
          (m == 0) ? f3(0.9f, 0.9f, 0.9f) :
          (m == 1) ? f3(0.9f, 0.5f, 0.2f) :
          (m == 2) ? f3(0.2f, 0.8f, 0.9f) :
          (m == 3) ? f3(0.7f, 0.2f, 0.8f) :
          (m == 4) ? f3(0.2f, 0.9f, 0.3f) :
          (m == 5) ? f3(0.9f, 0.2f, 0.2f) : f3(0.8f, 0.8f, 0.3f);

      spheres.push_back({f3(x, base_y, z), radius, alb, refl});
    }
  }

  spheres.push_back({f3(-1.2f, 0.6f, -3.5f), 0.6f, f3(0.9f, 0.3f, 0.3f), 0.7f});
  spheres.push_back({f3(1.4f, 0.9f, -4.2f), 0.9f, f3(0.3f, 0.9f, 0.4f), 0.2f});
  spheres.push_back({f3(0.0f, 1.3f, -5.5f), 1.3f, f3(0.4f, 0.6f, 0.95f), 0.0f});
}

// Scene 1: Cornell Box style with reflective walls
void build_scene_cornell(std::vector<Sphere>& spheres, std::vector<Plane>& planes) {
  spheres.push_back({f3(-0.6f, 0.5f, -2.5f), 0.5f, f3(0.9f, 0.3f, 0.3f), 0.85f});
  spheres.push_back({f3(0.6f, 0.6f, -2.8f), 0.6f, f3(0.95f, 0.95f, 0.95f), 0.95f});

  planes.push_back({f3(0, 1, 0), 1.0f, f3(0.9f, 0.9f, 0.9f), 0.3f});
  planes.push_back({f3(0, 0, 1), 5.0f, f3(0.6f, 0.65f, 0.8f), 0.7f});
  planes.push_back({f3(1, 0, 0), 2.0f, f3(0.8f, 0.2f, 0.2f), 0.6f});
  planes.push_back({f3(-1, 0, 0), 2.0f, f3(0.2f, 0.8f, 0.3f), 0.6f});
  planes.push_back({f3(0, -1, 0), 2.5f, f3(0.95f, 0.95f, 0.95f), 0.2f});
}

// ============================================================================
// Host Helpers
// ============================================================================
static void save_ppm(const std::string& path, const std::vector<Pixel>& img, int w, int h) {
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) { perror("ppm"); return; }
  fprintf(f, "P6\n%d %d\n255\n", w, h);
  for (int i = 0; i < w * h; i++) {
    unsigned char r = (unsigned char)std::min(255.f, std::max(0.f, img[i].r * 255.f));
    unsigned char g = (unsigned char)std::min(255.f, std::max(0.f, img[i].g * 255.f));
    unsigned char b = (unsigned char)std::min(255.f, std::max(0.f, img[i].b * 255.f));
    fwrite(&r, 1, 1, f); fwrite(&g, 1, 1, f); fwrite(&b, 1, 1, f);
  }
  fclose(f);
}

std::string getGpu(const char* name) {
  std::string full = std::string(name);
  size_t pos = full.find_last_of(" ");
  return (pos != std::string::npos && pos + 1 < full.size()) ? full.substr(pos + 1) : full;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  std::cout << "Persistent Wavefront Raytracer on " << prop.name << " using " << QUEUE_NAME << "\n";

  // Parse arguments
  int NUM_THREADS = (argc > 1) ? atoi(argv[1]) : 1024;
  int scene_id = (argc > 2) ? atoi(argv[2]) : 0;
  int MAX_BOUNCES = (argc > 3) ? atoi(argv[3]) : 4;

  std::cout << "Threads per tile: " << NUM_THREADS << "\n";
  std::cout << "Scene: " << scene_id << " (" << (scene_id == 0 ? "Complex Spheres" : "Cornell Box") << ")\n";
  std::cout << "Max bounces: " << MAX_BOUNCES << "\n";

  // Image and tiling setup
  const int W = 1280, H = 720;
  const int TILES_X = 8, TILES_Y = 8;
  const int NUM_TILES = TILES_X * TILES_Y;
  const int TILE_W = (W + TILES_X - 1) / TILES_X;
  const int TILE_H = (H + TILES_Y - 1) / TILES_Y;

  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(d_tiles_x), &TILES_X, sizeof(int)));
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(d_tiles_y), &TILES_Y, sizeof(int)));
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(d_tile_w), &TILE_W, sizeof(int)));
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(d_tile_h), &TILE_H, sizeof(int)));

  const int TPB = 256;
  const int NUM_BLOCKS_Y = (NUM_THREADS + TPB - 1) / TPB;

  // Build scene
  std::vector<Sphere> h_spheres;
  std::vector<Plane> h_planes;

  if (scene_id == 0) {
    build_scene_complex(h_spheres, h_planes);
  } else {
    build_scene_cornell(h_spheres, h_planes);
  }

  int ns = (int)h_spheres.size();
  int np = (int)h_planes.size();
  std::cout << "Scene: " << ns << " spheres, " << np << " planes\n";

  // Upload scene
  Sphere* d_spheres = nullptr;
  Plane* d_planes = nullptr;
  HIP_CHECK(hipMalloc(&d_spheres, h_spheres.size() * sizeof(Sphere)));
  HIP_CHECK(hipMalloc(&d_planes, h_planes.size() * sizeof(Plane)));
  HIP_CHECK(hipMemcpy(d_spheres, h_spheres.data(), h_spheres.size() * sizeof(Sphere), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_planes, h_planes.data(), h_planes.size() * sizeof(Plane), hipMemcpyHostToDevice));

  // ========================================================================
  // Allocate ONE set of queues (not MAX_BOUNCES+1 sets!)
  // ========================================================================
  std::vector<QUEUE_TYPE*>  h_queues(NUM_TILES, nullptr);
  std::vector<HANDLE_TYPE*> h_handles(NUM_TILES, nullptr);
#ifdef NEEDS_RECORDS
  #if defined(USE_BWFREE)
    std::vector<wfq_record*> h_records(NUM_TILES, nullptr);
  #else
    std::vector<wf_thread_record*> h_records(NUM_TILES, nullptr);
  #endif
#endif

  for (int t = 0; t < NUM_TILES; t++) {
#ifdef NEEDS_RECORDS
    wf_queue_host_init(&h_queues[t], &h_handles[t], &h_records[t], NUM_THREADS);
#else
    wf_queue_host_init(&h_queues[t], &h_handles[t], NUM_THREADS);
#endif
  }

  // Device pointer arrays
  QUEUE_TYPE** d_queues = nullptr;
  HANDLE_TYPE** d_handles = nullptr;
  HIP_CHECK(hipMalloc(&d_queues, NUM_TILES * sizeof(QUEUE_TYPE*)));
  HIP_CHECK(hipMalloc(&d_handles, NUM_TILES * sizeof(HANDLE_TYPE*)));
  HIP_CHECK(hipMemcpy(d_queues, h_queues.data(), NUM_TILES * sizeof(QUEUE_TYPE*), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_handles, h_handles.data(), NUM_TILES * sizeof(HANDLE_TYPE*), hipMemcpyHostToDevice));

  // Single ray buffer (no ping-pong needed — thread owns index exclusively)
  Ray* d_rays = nullptr;
  Pixel* d_img = nullptr;
  HIP_CHECK(hipMalloc(&d_rays, W * H * sizeof(Ray)));
  HIP_CHECK(hipMalloc(&d_img, W * H * sizeof(Pixel)));
  HIP_CHECK(hipMemset(d_img, 0, W * H * sizeof(Pixel)));

  // Per-tile counters
  unsigned long long* d_rays_alive = nullptr;       // termination: dec on terminal rays
  unsigned long long* d_total_rays_traced = nullptr; // stats only
  HIP_CHECK(hipMalloc(&d_rays_alive, NUM_TILES * sizeof(unsigned long long)));
  HIP_CHECK(hipMalloc(&d_total_rays_traced, NUM_TILES * sizeof(unsigned long long)));
  HIP_CHECK(hipMemset(d_rays_alive, 0, NUM_TILES * sizeof(unsigned long long)));
  HIP_CHECK(hipMemset(d_total_rays_traced, 0, NUM_TILES * sizeof(unsigned long long)));

  // per tile check if it's FULL
  // unsigned long long* enqueue_fails; // NUM_TILES
  // HIP_CHECK(hipMalloc(&enqueue_fails, NUM_TILES * sizeof(unsigned long long)));

  // Camera
  float3 cam_o, cam_f, cam_r, cam_u;
  if (scene_id == 0) {
    cam_o = f3(0, 0.3f, 0.5f);
    cam_f = normalize3(f3(0, -0.2f, -1.0f));
    cam_r = normalize3(f3(1, 0, 0));
    cam_u = normalize3(f3(0, 1, 0));
  } else {
    cam_o = f3(0, 0.5f, 1.5f);
    cam_f = normalize3(f3(0, -0.1f, -1.0f));
    cam_r = normalize3(f3(1, 0, 0));
    cam_u = normalize3(f3(0, 1, 0));
  }

  // Timing
  hipEvent_t e0, e1;
  HIP_CHECK(hipEventCreate(&e0));
  HIP_CHECK(hipEventCreate(&e1));

  dim3 grid(NUM_TILES, NUM_BLOCKS_Y, 1);
  dim3 block(TPB, 1, 1);

  // ========================================================================
  // LAUNCH 1: Generate primary rays
  // ========================================================================
  std::cout << "\nGenerating primary rays...\n";
  HIP_CHECK(hipEventRecord(e0));

  k_generate_primaries<<<grid, block>>>(
      d_queues, d_handles,
      d_rays, W, H,
      cam_o, cam_f, cam_r, cam_u,
      d_rays_alive,
      NUM_THREADS);

  HIP_CHECK(hipEventRecord(e1));
  HIP_CHECK(hipEventSynchronize(e1));
  float ms_gen = 0;
  HIP_CHECK(hipEventElapsedTime(&ms_gen, e0, e1));
  HIP_LAUNCH_CHECK();

  // Read primary ray count
  std::vector<unsigned long long> h_alive(NUM_TILES);
  HIP_CHECK(hipMemcpy(h_alive.data(), d_rays_alive, NUM_TILES * sizeof(unsigned long long), hipMemcpyDeviceToHost));
  unsigned long long primary_rays = 0;
  for (int t = 0; t < NUM_TILES; t++) primary_rays += h_alive[t];
  std::cout << "  Generated " << primary_rays << " primary rays in " << ms_gen << " ms\n";

  // ========================================================================
  // LAUNCH 2: Persistent trace — ALL bounces, SINGLE kernel launch
  // ========================================================================
  std::cout << "Persistent trace (max " << MAX_BOUNCES << " bounces)...\n";
  HIP_CHECK(hipEventRecord(e0));

  k_trace_persistent<<<grid, block>>>(
      d_queues, d_handles,
      d_spheres, ns,
      d_planes, np,
      d_rays,
      d_img, W, H,
      d_rays_alive,
      d_total_rays_traced,
      MAX_BOUNCES,
      NUM_THREADS);

  HIP_CHECK(hipEventRecord(e1));
  HIP_CHECK(hipEventSynchronize(e1));
  float ms_trace = 0;
  HIP_CHECK(hipEventElapsedTime(&ms_trace, e0, e1));
  HIP_LAUNCH_CHECK();

  // Read total rays traced
  std::vector<unsigned long long> h_traced(NUM_TILES);
  HIP_CHECK(hipMemcpy(h_traced.data(), d_total_rays_traced, NUM_TILES * sizeof(unsigned long long), hipMemcpyDeviceToHost));
  unsigned long long total_rays = 0;
  for (int t = 0; t < NUM_TILES; t++) total_rays += h_traced[t];

  // Verify termination
  HIP_CHECK(hipMemcpy(h_alive.data(), d_rays_alive, NUM_TILES * sizeof(unsigned long long), hipMemcpyDeviceToHost));
  unsigned long long leftover = 0;
  for (int t = 0; t < NUM_TILES; t++) leftover += h_alive[t];
  if (leftover != 0) {
    std::cerr << "WARNING: " << leftover << " rays still alive after trace!\n";
  }

  // ========================================================================
  // Results
  // ========================================================================
  float total_time_ms = ms_gen + ms_trace;
  double mrays = (total_time_ms > 0.0) ? ((double)total_rays / 1.0e6) / (total_time_ms / 1.0e3) : 0.0;
  double mrays_trace_only = (ms_trace > 0.0) ? ((double)total_rays / 1.0e6) / (ms_trace / 1.0e3) : 0.0;

  std::cout << "\n=== RESULTS ===\n";
  std::cout << "Primary rays:      " << primary_rays << "\n";
  std::cout << "Total rays traced:  " << total_rays << "\n";
  std::cout << "Secondary rays:     " << (total_rays - primary_rays) << "\n";
  std::cout << "Generate time:      " << ms_gen << " ms\n";
  std::cout << "Trace time:         " << ms_trace << " ms  (single persistent launch)\n";
  std::cout << "Total time:         " << total_time_ms << " ms\n";
  std::cout << "Throughput (total): " << mrays << " MRays/s\n";
  std::cout << "Throughput (trace): " << mrays_trace_only << " MRays/s\n";
  std::cout << "Kernel launches:    2  (generate + persistent trace)\n";

  // Save image
  std::vector<Pixel> h_img(W * H);
  HIP_CHECK(hipMemcpy(h_img.data(), d_img, W * H * sizeof(Pixel), hipMemcpyDeviceToHost));

  std::string scene_name = (scene_id == 0) ? "complex" : "cornell";
  std::string filename = "rt_final_" + scene_name + "_" + QUEUE_NAME + ".ppm";
  save_ppm(filename, h_img, W, H);
  std::cout << "Image saved: " << filename << "\n";

  // CSV
  const char* csv_path = "rt_final_performance.csv";
  std::string DEVICE = getGpu(prop.name);

  bool file_exists = static_cast<bool>(std::ifstream(csv_path));
  std::ofstream csv(csv_path, std::ios::app);
  if (csv) {
    if (!file_exists) {
      csv << "DEVICE,QUEUE,SCENE,THREADS,MAX_BOUNCES,PRIMARY_RAYS,TOTAL_RAYS,GEN_MS,TRACE_MS,TOTAL_MS,MRAYS_PER_S\n";
    }
    csv << DEVICE << ","
        << QUEUE_NAME << ","
        << scene_name << ","
        << NUM_THREADS << ","
        << MAX_BOUNCES << ","
        << primary_rays << ","
        << total_rays << ","
        << ms_gen << ","
        << ms_trace << ","
        << total_time_ms << ","
        << mrays << "\n";
  }

  // ========================================================================
  // Cleanup
  // ========================================================================
  HIP_CHECK(hipFree(d_queues));
  HIP_CHECK(hipFree(d_handles));

  for (int t = 0; t < NUM_TILES; t++) {
#ifdef USE_SFQ
    hipFree(h_queues[t]);
    hipFree(h_handles[t]);
#else
  #ifdef NEEDS_RECORDS
    wf_queue_destroy(h_queues[t], h_handles[t]);
  #else
    hipFree(h_queues[t]);
    hipFree(h_handles[t]);
  #endif
#endif
  }

  HIP_CHECK(hipFree(d_rays));
  HIP_CHECK(hipFree(d_img));
  HIP_CHECK(hipFree(d_spheres));
  HIP_CHECK(hipFree(d_planes));
  HIP_CHECK(hipFree(d_rays_alive));
  HIP_CHECK(hipFree(d_total_rays_traced));
  HIP_CHECK(hipEventDestroy(e0));
  HIP_CHECK(hipEventDestroy(e1));

  return 0;
}