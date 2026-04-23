// rt_gwfq.cpp — Persistent wavefront raytracer on gWFQ
//
// DESIGN PRINCIPLES: ONE queue per tile, ONE trace launch for ALL bounces.
//      Reflection rays re-enqueue into the same tile queue.
//      Zero host synchronization between bounces.
//
// Compile:
//   hipcc -O3 -std=c++17 rt_gwfq.cpp -o rt_gwfq
//   hipcc -O3 -std=c++17 -DRT_MAX_THREADS=4096 rt_gwfq.cpp -o rt_gwfq
//
// Usage: ./rt_gwfq [scene] [max_bounces] [threads_per_tile]
//   scene: 0 = complex sphere field, 1 = cornell box (default: 0)
//   max_bounces: reflection depth (default: 4)
//   threads_per_tile: worker threads per tile queue (default: 1024)

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

#define HIP_CHECK(call) do {                                       \
    hipError_t _e = (call);                                        \
    if (_e != hipSuccess) {                                        \
        fprintf(stderr, "HIP error %s:%d: %s\n",                  \
                __FILE__, __LINE__, hipGetErrorString(_e));        \
        std::exit(1);                                              \
    }                                                              \
} while (0)

// ============================================================================
// Queue Configuration
// ============================================================================
//
// RT_MAX_THREADS: compile-time ceiling on threads per tile.
// The queue's Record[] array is sized to MAX_THREADS (from the selected config),

#ifndef RT_MAX_THREADS
#define RT_MAX_THREADS 32768
#endif


#ifndef RT_QUEUE_N
#define RT_QUEUE_N 32768
#endif


#if defined(USE_GLFQ)
  #ifndef FQ_N
  #define FQ_N RT_QUEUE_N
  #endif
  #include "../glf_queue.hpp"

  using rt_queue_t = fq_ring_t;
  static constexpr const char* RT_QUEUE_NAME = "gLFQ";
  static constexpr int RT_MAX_WORKERS = RT_MAX_THREADS;

  struct rt_worker_t { int dummy; };

  __device__ __forceinline__
  void rt_worker_init(rt_worker_t* w, int tid) {
      (void)w; (void)tid;
  }

  __device__ __forceinline__
  bool rt_enqueue_idx(rt_queue_t* q, rt_worker_t* w, uint32_t idx) {
      (void)w;
      fq_enqueue(q, idx);
      return true;
  }

  __device__ __forceinline__
  bool rt_dequeue_idx(rt_queue_t* q, rt_worker_t* w, uint32_t* out) {
      (void)w;
      *out = fq_dequeue(q);
      return (*out != FQ_BOT);
  }

  __global__ void k_init_queue(rt_queue_t* q) {
      if (threadIdx.x == 0) fq_ring_init_empty(q);
  }

#elif defined(USE_GWFQ)
  #include "../gwf_ring.hpp"

  #if RT_MAX_THREADS < 1024
    using rt_cfg = gwf_1k;
  #elif RT_MAX_THREADS < 2048
    using rt_cfg = gwf_2k;
  #elif RT_MAX_THREADS < 4096
    using rt_cfg = gwf_4k;
  #elif RT_MAX_THREADS < 8192
    using rt_cfg = gwf_8k;
  #elif RT_MAX_THREADS < 16384
    using rt_cfg = gwf_16k;
  #elif RT_MAX_THREADS < 32768
    using rt_cfg = gwf_32k;
  #else
    using rt_cfg = gwf_64k;
  #endif

  using rt_queue_t = wf_t<rt_cfg, RT_QUEUE_N>;
  static constexpr const char* RT_QUEUE_NAME = "gWFQ";
  static constexpr int RT_MAX_WORKERS = rt_cfg::MAX_THREADS;

  static_assert(rt_cfg::MAX_THREADS > RT_MAX_THREADS,
      "Config MAX_THREADS must exceed RT_MAX_THREADS to avoid NULL_TID collision.");
  static_assert(RT_QUEUE_N >= rt_cfg::MAX_THREADS,
      "Queue capacity must be >= MAX_THREADS.");

  struct rt_worker_t { int dummy; };

  __device__ __forceinline__
  void rt_worker_init(rt_worker_t* w, int tid) {
      (void)w; (void)tid;
  }

  __device__ __forceinline__
  bool rt_enqueue_idx(rt_queue_t* q, rt_worker_t* w, uint32_t idx) {
      (void)w;
      Enqueue_wf<rt_cfg, RT_QUEUE_N>(q, idx);
      return true;
  }

  __device__ __forceinline__
  bool rt_dequeue_idx(rt_queue_t* q, rt_worker_t* w, uint32_t* out) {
      (void)w;
      *out = Dequeue_wf<rt_cfg, RT_QUEUE_N>(q);
      return (*out != wf_entry::BOT);
  }

  __global__ void k_init_queue(rt_queue_t* q) {
      if (threadIdx.x == 0) wf_init<rt_cfg, RT_QUEUE_N>(q);
  }

#elif defined(USE_WFQ)
  #include "../wfqueue_hip_opt.hpp"

  using rt_queue_t  = wf_queue;
  using rt_handle_t = wf_handle;
  static constexpr const char* RT_QUEUE_NAME = "WFQ";
  static constexpr int RT_MAX_WORKERS = RT_MAX_THREADS;

  struct rt_worker_t { rt_handle_t* h; };

  __device__ __forceinline__
  void rt_worker_init(rt_worker_t* w, rt_handle_t* tile_handles, int tid) {
      w->h = &tile_handles[tid];
  }

  __device__ __forceinline__
  bool rt_enqueue_idx(rt_queue_t* q, rt_worker_t* w, uint32_t idx) {
    // Offset by +1 so ray 0 doesn't get swallowed by WF_BOTTOM (0)
      bool ok = wf_enqueue(q, w->h, (uint64_t)(idx + 1u));
      return ok;
  }

  __device__ __forceinline__
  bool rt_dequeue_idx(rt_queue_t* q, rt_worker_t* w, uint32_t* out) {
      uint64_t v = wf_dequeue(q, w->h);
      if (v == WF_EMPTY) return false;
      *out = (uint32_t)(v - 1u);
      return true;
  }
#elif defined(USE_BWD)
  #include "../bq.hpp"
  static const uint32_t NUM_QUEUES = 64;
  using rt_queue_t = bq::BrokerQueue<uint32_t, RT_QUEUE_N, RT_MAX_THREADS>;
  static constexpr const char* RT_QUEUE_NAME = "BWD";
  static constexpr int RT_MAX_WORKERS = RT_MAX_THREADS;

  struct rt_worker_t { int dummy; };

  __device__ __forceinline__
  void rt_worker_init(rt_worker_t* w, int tid) {
    //   w->my_q = (uint32_t)tid;
    (void)w; (void)tid;
  }

  __device__ __forceinline__
  bool rt_enqueue_idx(rt_queue_t* q, rt_worker_t* w, uint32_t idx) {
      return q->enqueue_bwd(idx) == bq::QueueStatus::Success;
  }

  __device__ __forceinline__
  bool rt_dequeue_idx(rt_queue_t* q, rt_worker_t* w, uint32_t* out) {
    //   bq::QueueStatus s = q->dequeue(*out, w->my_q);
      return q->dequeue_bwd(*out) == bq::QueueStatus::Success;
  }

  __global__ void k_init_queue(rt_queue_t* q) {
    //   if (threadIdx.x == 0) q->host_init();
    if (threadIdx.x == 0) {
        for (uint32_t i = 0; i < RT_QUEUE_N; ++i) {
            q->tickets[i] =0;
        }
        q->head_tail_packed = 0ULL;
        q->count = 0;
    }
  }

#elif defined(USE_BQ)
  #include "../bq.hpp"
  static const uint32_t NUM_QUEUES = 64;
  using rt_queue_t = bq::BrokerQueue<uint32_t, RT_QUEUE_N, RT_MAX_THREADS>;
  static constexpr const char* RT_QUEUE_NAME = "BQ";
  static constexpr int RT_MAX_WORKERS = RT_MAX_THREADS;

  struct rt_worker_t { int dummy; };

  __device__ __forceinline__
  void rt_worker_init(rt_worker_t* w, int tid) {
    //   w->my_q = (uint32_t)tid;
    (void)w; (void)tid;
  }

  __device__ __forceinline__
  bool rt_enqueue_idx(rt_queue_t* q, rt_worker_t* w, uint32_t idx) {
      return q->enqueue(idx) == bq::QueueStatus::Success;
  }

  __device__ __forceinline__
  bool rt_dequeue_idx(rt_queue_t* q, rt_worker_t* w, uint32_t* out) {
    //   bq::QueueStatus s = q->dequeue(*out, w->my_q);
      return q->dequeue(*out) == bq::QueueStatus::Success;
  }

  __global__ void k_init_queue(rt_queue_t* q) {
    //   if (threadIdx.x == 0) q->host_init();
    if (threadIdx.x == 0) {
        for (uint32_t i = 0; i < RT_QUEUE_N; ++i) {
            q->tickets[i] =0;
        }
        q->head_tail_packed = 0ULL;
        q->count = 0;
    }
  }

#elif defined(USE_SFQ)
  #include "../sfqueue_hip.hpp"
  #include "../sfqueue_hip.cpp"

  using rt_queue_t  = sfq_queue;
  using rt_handle_t = sfq_handle;
  static constexpr const char* RT_QUEUE_NAME = "SFQ";
  static constexpr int RT_MAX_WORKERS = RT_MAX_THREADS;

  struct rt_worker_t { rt_handle_t* h; };

  __device__ __forceinline__
  void rt_worker_init(rt_worker_t* w, rt_handle_t* tile_handles, int tid) {
      w->h = &tile_handles[tid];
  }

  __device__ __forceinline__
  bool rt_enqueue_idx(rt_queue_t* q, rt_worker_t* w, uint32_t idx) {
      (void)w;
      const uint32_t item = idx + 1u;
      const int rc = sfq_enqueue_blocking_u32(q, item);
      return (rc == SFQ_SUCCESS);
  }

  __device__ __forceinline__
  bool rt_dequeue_idx(rt_queue_t* q, rt_worker_t* w, uint32_t* out) {
      (void)w;
      uint32_t item = 0u;
      const int rc = sfq_dequeue_nb_u32(q, &item);
      if (rc != SFQ_SUCCESS) return false;
      *out = item - 1u;
      return true;
  }

  __global__ void k_init_queue(rt_queue_t* q, rt_handle_t* handles, int threads_per_tile) {
      const int tid = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
      const int total = (int)blockDim.x * (int)gridDim.x;

      if (tid == 0) {
          sfq_queue_init(q, threads_per_tile);
      }

      for (int i = tid; i < (int)SFQ_QUEUE_LENGTH; i += total) {
          q->items[i] = 0u;
          q->ids[i] = 0u;
      }

      for (int i = tid; i < threads_per_tile; i += total) {
          sfq_handle_init(&handles[i], (uint32_t)i);
      }
  }

#else
  #error "Define exactly one of USE_GLFQ, USE_GWFQ, USE_WFQ, USE_BQ, USE_SFQ"
#endif


__device__ __forceinline__
unsigned long long load_u64_relaxed(const unsigned long long* p) {
#if defined(__clang__)
    return __atomic_load_n((const unsigned long long*)p, __ATOMIC_RELAXED);
#else
    return atomicAdd((unsigned long long*)p, 0ULL); // fallback
#endif
}

// ============================================================================
// Image / Tiling Constants
// ============================================================================
static constexpr int IMG_W = 1280;
static constexpr int IMG_H = 720;
static constexpr int TILES_X = 8;
static constexpr int TILES_Y = 8;
static constexpr int NUM_TILES = TILES_X * TILES_Y;
static constexpr int TILE_W = (IMG_W + TILES_X - 1) / TILES_X;   // 160
static constexpr int TILE_H = (IMG_H + TILES_Y - 1) / TILES_Y;   // 90
static constexpr int TILE_PIXELS_MAX = TILE_W * TILE_H;
static constexpr int TPB = 256;

// Device-side constants
__constant__ int d_tiles_x, d_tiles_y, d_tile_w, d_tile_h;

// ============================================================================
// Math Helpers
// ============================================================================
__host__ __device__ inline float3 f3(float x, float y, float z) { return make_float3(x, y, z); }
__host__ __device__ inline float3 operator+(float3 a, float3 b) { return f3(a.x+b.x, a.y+b.y, a.z+b.z); }
__host__ __device__ inline float3 operator-(float3 a, float3 b) { return f3(a.x-b.x, a.y-b.y, a.z-b.z); }
__host__ __device__ inline float3 operator*(float3 a, float s) { return f3(a.x*s, a.y*s, a.z*s); }
// __host__ __device__ inline float3 operator*(float3 a, float3 b) { return f3(a.x*b.x, a.y*b.y, a.z*b.z); }
__host__ __device__ inline float3 mul(float3 a, float3 b) { return f3(a.x*b.x, a.y*b.y, a.z*b.z); }
__host__ __device__ inline float  dot(float3 a, float3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
__host__ __device__ inline float3 norm(float3 v) {
    float d = sqrtf(dot(v, v));
    return d > 0.f ? v * (1.f / d) : v;
}
__host__ __device__ inline float3 reflect(float3 v, float3 n) { return v - n * (2.f * dot(v, n)); }
__host__ __device__ inline float3 clamp01(float3 c) {
    return f3(fminf(fmaxf(c.x, 0.f), 1.f),
              fminf(fmaxf(c.y, 0.f), 1.f),
              fminf(fmaxf(c.z, 0.f), 1.f));
}

// ============================================================================
// Scene Structures
// ============================================================================
struct Sphere { float3 c; float r; float3 albedo; float refl; };
struct Plane  { float3 n; float d; float3 albedo; float refl; };
struct Pixel  { float r, g, b; };

struct Ray {
    float3 o, d;
    int px, py;
    int depth;
    float3 atten;     // accumulated attenuation through bounces
};

struct Hit {
    float t;
    float3 n, p, albedo;
    float refl;
    bool did_hit;
};

// ============================================================================
// Intersection
// ============================================================================
__device__ Hit intersect_sphere(const Ray& ray, const Sphere& s) {
    Hit h; h.did_hit = false; h.t = 1e30f;
    float3 oc = ray.o - s.c;
    float b = dot(oc, ray.d);
    float c = dot(oc, oc) - s.r * s.r;
    float disc = b * b - c;
    if (disc >= 0.f) {
        float t = -b - sqrtf(disc);
        if (t > 1e-4f) {
            h.t = t; h.did_hit = true;
            h.p = ray.o + ray.d * t;
            h.n = norm(h.p - s.c);
            h.albedo = s.albedo;
            h.refl = s.refl;
        }
    }
    return h;
}

__device__ Hit intersect_plane(const Ray& ray, const Plane& pl) {
    Hit h; h.did_hit = false; h.t = 1e30f;
    float denom = dot(pl.n, ray.d);
    if (fabsf(denom) > 1e-5f) {
        float t = -(dot(pl.n, ray.o) + pl.d) / denom;
        if (t > 1e-4f) {
            h.t = t; h.did_hit = true;
            h.p = ray.o + ray.d * t;
            h.n = pl.n;
            h.albedo = pl.albedo;
            h.refl = pl.refl;
        }
    }
    return h;
}

__device__ Hit trace_scene(const Ray& ray,
                           const Sphere* sph, int ns,
                           const Plane* pln, int np) {
    Hit best; best.did_hit = false; best.t = 1e30f;
    for (int i = 0; i < ns; i++) {
        Hit h = intersect_sphere(ray, sph[i]);
        if (h.did_hit && h.t < best.t) best = h;
    }
    for (int i = 0; i < np; i++) {
        Hit h = intersect_plane(ray, pln[i]);
        if (h.did_hit && h.t < best.t) best = h;
    }
    return best;
}

// ============================================================================
// Pixel Accumulation (multiple bounces can contribute to the same pixel
// ============================================================================
__device__ inline void accum_pixel(Pixel* img, int w, int x, int y, float3 c) {
    int idx = y * w + x;
    atomicAdd(&img[idx].r, c.x);
    atomicAdd(&img[idx].g, c.y);
    atomicAdd(&img[idx].b, c.z);
}


// ============================================================================
// Kernel 1: Generate Primary Rays
// ============================================================================
#if defined(USE_WFQ)
__global__ void k_generate(
    rt_queue_t* const* __restrict__ queues,
    rt_handle_t* const* __restrict__ handles,
    Ray*        __restrict__ rays,
#elif defined(USE_SFQ)
__global__ void k_generate(
    rt_queue_t* __restrict__ queues,
    rt_handle_t* __restrict__ handles,
    Ray*        __restrict__ rays,
#else
__global__ void k_generate(
    rt_queue_t* __restrict__ queues,
    Ray*        __restrict__ rays,
#endif
    int img_w, int img_h,
    float3 cam_o, float3 cam_f, float3 cam_r, float3 cam_u,
    unsigned long long* __restrict__ alive,
    int threads_per_tile)
{
    const int tile_id = (int)blockIdx.y;
    const int tid     = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (tid >= threads_per_tile) return;

#if defined(USE_WFQ)
    rt_queue_t* q = queues[tile_id];
    rt_handle_t* tile_handles = handles[tile_id];
    rt_worker_t worker;
    rt_worker_init(&worker, tile_handles, tid);
#elif defined(USE_SFQ)
    rt_queue_t* q = &queues[tile_id];
    rt_handle_t* tile_handles = &handles[tile_id * threads_per_tile];
    rt_worker_t worker;
    rt_worker_init(&worker, tile_handles, tid);
#else
    rt_queue_t* q = &queues[tile_id];
    rt_worker_t worker;
    rt_worker_init(&worker, tid);
#endif

    const int tx = tile_id % d_tiles_x;
    const int ty = tile_id / d_tiles_x;
    const int x0 = tx * d_tile_w;
    const int y0 = ty * d_tile_h;
    const int tw = min(d_tile_w, img_w - x0);
    const int th = min(d_tile_h, img_h - y0);
    const int tile_pixels = tw * th;
    const float aspect = (float)img_w / (float)img_h;

    int enqueued = 0;
    for (int i = tid; i < tile_pixels; i += threads_per_tile) {
        int lx = i % tw, ly = i / tw;
        int px = x0 + lx, py = y0 + ly;
        int idx = py * img_w + px;

        // Build primary ray
        float u =  ((px + 0.5f) / img_w - 0.5f) * 2.f * aspect;
        float v = -((py + 0.5f) / img_h - 0.5f) * 2.f;

        Ray r;
        r.o = cam_o;
        r.d = norm(cam_f + cam_r * u + cam_u * v);
        r.px = px; r.py = py;
        r.depth = 0;
        r.atten = f3(1.f, 1.f, 1.f);
        rays[idx] = r;

        // Fence: ray data must be globally visible before the index is published
        __threadfence();

        bool ok = rt_enqueue_idx(q, &worker, (uint32_t)i);
        if (ok) enqueued++;
    }
    if (enqueued > 0) {
        atomicAdd(&alive[tile_id], (unsigned long long)enqueued);
    }
}

// ============================================================================
// Kernel 2: Persistent Trace — ALL bounces in a single launch
// ============================================================================
#if defined(USE_WFQ)
__global__ void k_trace(
    rt_queue_t* const* __restrict__ queues,
    rt_handle_t* const* __restrict__ handles,
    const Sphere* __restrict__ spheres, int ns,
#elif defined(USE_SFQ)
__global__ void k_trace(
    rt_queue_t* __restrict__ queues,
    rt_handle_t* __restrict__ handles,
    const Sphere* __restrict__ spheres, int ns,
#else
__global__ void k_trace(
    rt_queue_t* __restrict__ queues,
    const Sphere* __restrict__ spheres, int ns,
#endif
    const Plane*  __restrict__ planes,  int np,
    Ray*   __restrict__ rays,
    Pixel* __restrict__ img,
    int img_w, int img_h,
    unsigned long long* __restrict__ alive,
    unsigned long long* __restrict__ traced,
    int max_depth,
    int threads_per_tile)
{
    const int tile_id = (int)blockIdx.y;
    const int tid     = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (tid >= threads_per_tile) return;

#if defined(USE_WFQ)
    rt_queue_t* q = queues[tile_id];
    rt_handle_t* tile_handles = handles[tile_id];
    rt_worker_t worker;
    rt_worker_init(&worker, tile_handles, tid);
#elif defined(USE_SFQ)
    rt_queue_t* q = &queues[tile_id];
    rt_handle_t* tile_handles = &handles[tile_id * threads_per_tile];
    rt_worker_t worker;
    rt_worker_init(&worker, tile_handles, tid);
#else
    rt_queue_t* q = &queues[tile_id];
    rt_worker_t worker;
    rt_worker_init(&worker, tid);
#endif

    const int tx = tile_id % d_tiles_x;
    const int ty = tile_id / d_tiles_x;
    const int x0 = tx * d_tile_w;
    const int y0 = ty * d_tile_h;
    const int tw = min(d_tile_w, img_w - x0);
    const int th = min(d_tile_h, img_h - y0);
    (void)th;

    const float3 light_dir = norm(f3(-0.5f, 1.f, 0.2f));
    const float3 ambient   = f3(0.05f, 0.05f, 0.05f);

    unsigned long long local_traced = 0;
    int empty_spins = 0;

    while (true) {
        unsigned long long a = load_u64_relaxed(&alive[tile_id]);
        if (a == 0) break;
        // ── Dequeue ──
        uint32_t local_idx;
        bool got = rt_dequeue_idx(q, &worker, &local_idx);
        if (!got) {
            empty_spins++;
            if (empty_spins > 64) {
#if defined(__AMDGCN__)
                __builtin_amdgcn_s_sleep(2);
#endif
            }
            continue;
        }
        empty_spins = 0;

        int lx = local_idx % tw;
        int ly = local_idx / tw;
        int px = x0 + lx, py = y0 + ly;
        int global_idx = py * img_w + px;

        // int idx = (int)(tok - 1);   // decode (+1 encoding)
        Ray r = rays[global_idx];
        Hit hit = trace_scene(r, spheres, ns, planes, np);

        if (hit.did_hit) {
            // Direct illumination
            float NdotL = fmaxf(0.f, dot(hit.n, light_dir));
            float3 direct = ambient + hit.albedo * NdotL;
            float3 contrib = direct * r.atten;

            if (hit.refl > 0.f && r.depth < max_depth) {
                // ── Reflective: split contribution ──
                float3 non_refl = clamp01(contrib * (1.f - hit.refl));
                if (non_refl.x > 0.f || non_refl.y > 0.f || non_refl.z > 0.f)
                    accum_pixel(img, img_w, r.px, r.py, non_refl);

                // Build reflection ray, reuse the same ray slot
                Ray refl;
                refl.o = hit.p + hit.n * 1e-4f;
                refl.d = norm(reflect(r.d, hit.n));
                refl.px = r.px;
                refl.py = r.py;
                refl.depth = r.depth + 1;
                refl.atten = r.atten * hit.albedo * hit.refl;
                rays[global_idx] = refl;

                __threadfence();   // ray data visible before re-enqueue

                bool ok = rt_enqueue_idx(q, &worker, local_idx);
                if (!ok) {
                    atomicAdd(&alive[tile_id], (unsigned long long)(-1ULL));
                }
                // On success: alive unchanged (consumed 1, produced 1, net zero)
            } else {
                // ── Terminal: max depth or non-reflective ──
                accum_pixel(img, img_w, r.px, r.py, clamp01(contrib));
                atomicAdd(&alive[tile_id], (unsigned long long)(-1ULL));
            }
        } else {
            float t = 0.5f * (r.d.y + 1.f);
            float3 sky = f3(1,1,1) * (1.f - t) + f3(0.5f, 0.7f, 1.f) * t;
            accum_pixel(img, img_w, r.px, r.py, clamp01(sky * r.atten));
            atomicAdd(&alive[tile_id], (unsigned long long)(-1ULL));
        }
        local_traced++;
    }

    if (local_traced > 0)
        atomicAdd(&traced[tile_id], local_traced);
}

// ============================================================================
// Scene Builders
// ============================================================================
static void build_scene_complex(std::vector<Sphere>& sph, std::vector<Plane>& pln) {
    // Floor
    pln.push_back({f3(0,1,0), 1.f, f3(0.75f,0.75f,0.75f), 0.05f});

    // 10x10 sphere grid
    const int gx = 10, gz = 10;
    const float sp = 0.55f, rad = 0.22f, base_y = 0.25f;
    for (int iz = 0; iz < gz; iz++) {
        for (int ix = 0; ix < gx; ix++) {
            float x = (ix - (gx-1)*0.5f) * sp;
            float z = -2.f - iz * sp;
            int m = (ix + iz * 7) % 7;
            float r = (m==0)?0.9f : (m==1)?0.6f : (m==2)?0.3f : (m==3)?0.1f : 0.f;
            float3 a = (m==0)?f3(.9f,.9f,.9f) : (m==1)?f3(.9f,.5f,.2f) :
                       (m==2)?f3(.2f,.8f,.9f) : (m==3)?f3(.7f,.2f,.8f) :
                       (m==4)?f3(.2f,.9f,.3f) : (m==5)?f3(.9f,.2f,.2f) : f3(.8f,.8f,.3f);
            sph.push_back({f3(x, base_y, z), rad, a, r});
        }
    }
    // Hero spheres
    sph.push_back({f3(-1.2f,0.6f,-3.5f), 0.6f, f3(.9f,.3f,.3f), 0.7f});
    sph.push_back({f3( 1.4f,0.9f,-4.2f), 0.9f, f3(.3f,.9f,.4f), 0.2f});
    sph.push_back({f3( 0.0f,1.3f,-5.5f), 1.3f, f3(.4f,.6f,.95f), 0.0f});
}

static void build_scene_cornell(std::vector<Sphere>& sph, std::vector<Plane>& pln) {
    sph.push_back({f3(-0.6f,0.5f,-2.5f), 0.5f, f3(.9f,.3f,.3f),  0.85f});
    sph.push_back({f3( 0.6f,0.6f,-2.8f), 0.6f, f3(.95f,.95f,.95f), 0.95f});

    pln.push_back({f3( 0, 1,0),  1.f,  f3(.9f,.9f,.9f),    0.3f});  // floor
    pln.push_back({f3( 0, 0,1),  5.f,  f3(.6f,.65f,.8f),   0.7f});  // back
    pln.push_back({f3( 1, 0,0),  2.f,  f3(.8f,.2f,.2f),    0.6f});  // left (red)
    pln.push_back({f3(-1, 0,0),  2.f,  f3(.2f,.8f,.3f),    0.6f});  // right (green)
    pln.push_back({f3( 0,-1,0),  2.5f, f3(.95f,.95f,.95f), 0.2f});  // ceiling
}

// ============================================================================
// PPM Writer
// ============================================================================
static void save_ppm(const char* path, const std::vector<Pixel>& img, int w, int h) {
    FILE* f = fopen(path, "wb");
    if (!f) { perror("ppm"); return; }
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    for (int i = 0; i < w * h; i++) {
        unsigned char r = (unsigned char)fminf(255.f, fmaxf(0.f, img[i].r * 255.f));
        unsigned char g = (unsigned char)fminf(255.f, fmaxf(0.f, img[i].g * 255.f));
        unsigned char b = (unsigned char)fminf(255.f, fmaxf(0.f, img[i].b * 255.f));
        fwrite(&r, 1, 1, f); fwrite(&g, 1, 1, f); fwrite(&b, 1, 1, f);
    }
    fclose(f);
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    const char* gpu_env = std::getenv("GPU_NAME");
    const std::string gpu_name = (gpu_env && gpu_env[0] != '\0') ? std::string(gpu_env)
                                                                   : std::string(prop.name);

    // ── Parse args ──
    int scene_id    = (argc > 1) ? atoi(argv[1]) : 0;
    int max_bounces = (argc > 2) ? atoi(argv[2]) : 4;
    int threads_per_tile = (argc > 3) ? atoi(argv[3]) : 1024;

    // Runtime check: thread count must not exceed compile-time config
    if (threads_per_tile > RT_MAX_WORKERS) {
        fprintf(stderr,
            "ERROR: %d threads exceeds backend limit %d.\n"
            "Recompile with a larger RT_MAX_THREADS / queue configuration.\n",
            threads_per_tile, RT_MAX_WORKERS);
        return 1;
    }

#if defined(USE_SFQ)
    if ((unsigned)threads_per_tile > SFQ_MAX_THREADS) {
        fprintf(stderr,
            "ERROR: %d threads exceeds SFQ_MAX_THREADS=%u.\n"
            "Recompile with a larger SFQ_MAX_THREADS.\n",
            threads_per_tile, SFQ_MAX_THREADS);
        return 1;
    }
    if (TILE_PIXELS_MAX > (int)SFQ_QUEUE_LENGTH) {
        fprintf(stderr,
            "ERROR: tile pixel count %d exceeds SFQ queue capacity %u.\n"
            "Increase SFQ_QUEUE_LENGTH or reduce tile size.\n",
            TILE_PIXELS_MAX, SFQ_QUEUE_LENGTH);
        return 1;
    }
#endif

    const int blocks_per_tile = (threads_per_tile + TPB - 1) / TPB;

    std::cout << "=== Persistent Wavefront Raytracer (" << RT_QUEUE_NAME << ") ===\n"
            << "GPU:              " << gpu_name << "\n"
          << "Scene:            " << scene_id << " ("
          << (scene_id == 0 ? "Complex Spheres" : "Cornell Box") << ")\n"
          << "Max bounces:      " << max_bounces << "\n"
          << "Threads/tile:     " << threads_per_tile << "\n"
          << "Tiles:            " << TILES_X << "x" << TILES_Y << " = " << NUM_TILES << "\n"
          << "Blocks/tile:      " << blocks_per_tile << "\n"
          << "Queue capacity:   " << RT_QUEUE_N << "\n";
#if defined(USE_GWFQ)
    std::cout << "Config:           MAX_THREADS=" << rt_cfg::MAX_THREADS
              << " THRIDX_BITS=" << (64 - rt_cfg::COUNTER_BITS) << "\n"
              << "Record[] / queue: " << (rt_cfg::MAX_THREADS * 128) / 1024 << " KB\n";
#elif defined(USE_WFQ)
    std::cout << "Config:           WFQ host-init backend\n";
#elif defined(USE_SFQ)
    std::cout << "Config:           SFQ_QUEUE_LENGTH=" << SFQ_QUEUE_LENGTH << "\n";
#elif defined(USE_GLFQ)
    std::cout << "Config:           FQ_N=" << FQ_N << "\n";
#endif
#if defined(USE_WFQ)
    std::cout << "Total queue mem:  dynamic per-tile (host helper)\n\n";
#else
    std::cout << "Total queue mem: " << (NUM_TILES * sizeof(rt_queue_t)) / (1024 * 1024) << " MB\n\n";
#endif

    // ── Upload tiling constants ──
    int tx = TILES_X, ty = TILES_Y, tw = TILE_W, th = TILE_H;
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(d_tiles_x), &tx, sizeof(int)));
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(d_tiles_y), &ty, sizeof(int)));
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(d_tile_w),  &tw, sizeof(int)));
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(d_tile_h),  &th, sizeof(int)));

    // ── Build scene ──
    std::vector<Sphere> h_sph;
    std::vector<Plane>  h_pln;
    if (scene_id == 0) build_scene_complex(h_sph, h_pln);
    else               build_scene_cornell(h_sph, h_pln);

    int ns = (int)h_sph.size(), np = (int)h_pln.size();
    std::cout << "Scene: " << ns << " spheres, " << np << " planes\n";

    Sphere* d_sph = nullptr;
    Plane*  d_pln = nullptr;
    HIP_CHECK(hipMalloc(&d_sph, h_sph.size() * sizeof(Sphere)));
    HIP_CHECK(hipMalloc(&d_pln, h_pln.size() * sizeof(Plane)));
    HIP_CHECK(hipMemcpy(d_sph, h_sph.data(), h_sph.size() * sizeof(Sphere), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_pln, h_pln.data(), h_pln.size() * sizeof(Plane), hipMemcpyHostToDevice));

    // ── Allocate queues ──
#if defined(USE_WFQ)
    std::vector<rt_queue_t*> h_queues(NUM_TILES, nullptr);
    std::vector<rt_handle_t*> h_handles(NUM_TILES, nullptr);
    for (int t = 0; t < NUM_TILES; t++) {
        wf_queue_host_init(&h_queues[t], &h_handles[t], threads_per_tile);
    }

    rt_queue_t** d_queues = nullptr;
    rt_handle_t** d_handles = nullptr;
    HIP_CHECK(hipMalloc(&d_queues, NUM_TILES * sizeof(rt_queue_t*)));
    HIP_CHECK(hipMalloc(&d_handles, NUM_TILES * sizeof(rt_handle_t*)));
    HIP_CHECK(hipMemcpy(d_queues, h_queues.data(), NUM_TILES * sizeof(rt_queue_t*), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_handles, h_handles.data(), NUM_TILES * sizeof(rt_handle_t*), hipMemcpyHostToDevice));

#elif defined(USE_SFQ)
    rt_queue_t* d_queues = nullptr;
    rt_handle_t* d_handles = nullptr;
    HIP_CHECK(hipMalloc(&d_queues, NUM_TILES * sizeof(rt_queue_t)));
    HIP_CHECK(hipMalloc(&d_handles, NUM_TILES * threads_per_tile * sizeof(rt_handle_t)));

    const int init_threads = TPB;
    const int init_work = std::max((int)SFQ_QUEUE_LENGTH, threads_per_tile);
    const int init_blocks = (init_work + init_threads - 1) / init_threads;

    for (int t = 0; t < NUM_TILES; t++) {
        k_init_queue<<<init_blocks, init_threads>>>(&d_queues[t], &d_handles[t * threads_per_tile], threads_per_tile);
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

#else
    rt_queue_t* d_queues = nullptr;
    HIP_CHECK(hipMalloc(&d_queues, NUM_TILES * sizeof(rt_queue_t)));
    for (int t = 0; t < NUM_TILES; t++) {
        k_init_queue<<<1, 1>>>(&d_queues[t]);
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
#endif

    // ── Allocate buffers ──
    Ray*   d_rays = nullptr;
    Pixel* d_img  = nullptr;
    HIP_CHECK(hipMalloc(&d_rays, IMG_W * IMG_H * sizeof(Ray)));
    HIP_CHECK(hipMalloc(&d_img,  IMG_W * IMG_H * sizeof(Pixel)));
    HIP_CHECK(hipMemset(d_img, 0, IMG_W * IMG_H * sizeof(Pixel)));

    unsigned long long* d_alive  = nullptr;
    unsigned long long* d_traced = nullptr;
    HIP_CHECK(hipMalloc(&d_alive,  NUM_TILES * sizeof(unsigned long long)));
    HIP_CHECK(hipMalloc(&d_traced, NUM_TILES * sizeof(unsigned long long)));
    HIP_CHECK(hipMemset(d_alive,  0, NUM_TILES * sizeof(unsigned long long)));
    HIP_CHECK(hipMemset(d_traced, 0, NUM_TILES * sizeof(unsigned long long)));

    // ── Camera ──
    float3 cam_o, cam_f, cam_r, cam_u;
    if (scene_id == 0) {
        cam_o = f3(0, 0.3f, 0.5f);
        cam_f = norm(f3(0, -0.2f, -1.f));
    } else {
        cam_o = f3(0, 0.5f, 1.5f);
        cam_f = norm(f3(0, -0.1f, -1.f));
    }
    cam_r = norm(f3(1, 0, 0));
    cam_u = norm(f3(0, 1, 0));

    dim3 grid(blocks_per_tile, NUM_TILES);
    dim3 block(TPB);

    hipEvent_t e0, e1;
    HIP_CHECK(hipEventCreate(&e0));
    HIP_CHECK(hipEventCreate(&e1));

    // ========================================================================
    // LAUNCH 1: Generate primary rays
    // ========================================================================
    std::cout << "\nGenerating primary rays...\n";
    HIP_CHECK(hipEventRecord(e0));

    k_generate<<<grid, block>>>(
#if defined(USE_WFQ) || defined(USE_SFQ)
        d_queues, d_handles, d_rays, IMG_W, IMG_H,
#else
        d_queues, d_rays, IMG_W, IMG_H,
#endif
        cam_o, cam_f, cam_r, cam_u,
        d_alive, threads_per_tile);

    HIP_CHECK(hipEventRecord(e1));
    HIP_CHECK(hipEventSynchronize(e1));
    HIP_CHECK(hipGetLastError());
    float ms_gen = 0;
    HIP_CHECK(hipEventElapsedTime(&ms_gen, e0, e1));

    // Read alive counts
    std::vector<unsigned long long> h_alive(NUM_TILES);
    HIP_CHECK(hipMemcpy(h_alive.data(), d_alive,
                         NUM_TILES * sizeof(unsigned long long), hipMemcpyDeviceToHost));
    unsigned long long primary_rays = 0;
    for (int t = 0; t < NUM_TILES; t++) primary_rays += h_alive[t];
    std::cout << "  " << primary_rays << " primary rays in " << ms_gen << " ms\n";

    // ========================================================================
    // LAUNCH 2: Persistent trace — single kernel, all bounces
    // ========================================================================
    std::cout << "Persistent trace (max " << max_bounces << " bounces)...\n";
    HIP_CHECK(hipEventRecord(e0));

    k_trace<<<grid, block>>>(
#if defined(USE_WFQ) || defined(USE_SFQ)
        d_queues, d_handles,
#else
        d_queues,
#endif
        d_sph, ns, d_pln, np,
        d_rays, d_img, IMG_W, IMG_H,
        d_alive, d_traced,
        max_bounces, threads_per_tile);

    HIP_CHECK(hipEventRecord(e1));
    HIP_CHECK(hipEventSynchronize(e1));
    HIP_CHECK(hipGetLastError());
    float ms_trace = 0;
    HIP_CHECK(hipEventElapsedTime(&ms_trace, e0, e1));

    // Read stats
    std::vector<unsigned long long> h_traced(NUM_TILES);
    HIP_CHECK(hipMemcpy(h_traced.data(), d_traced,
                         NUM_TILES * sizeof(unsigned long long), hipMemcpyDeviceToHost));
    unsigned long long total_rays = 0;
    for (int t = 0; t < NUM_TILES; t++) total_rays += h_traced[t];

    // Verify termination
    HIP_CHECK(hipMemcpy(h_alive.data(), d_alive,
                         NUM_TILES * sizeof(unsigned long long), hipMemcpyDeviceToHost));
    unsigned long long leftover = 0;
    for (int t = 0; t < NUM_TILES; t++) leftover += h_alive[t];

    // ========================================================================
    // Results
    // ========================================================================
    float total_ms = ms_gen + ms_trace;
    double mrays_total = (total_ms > 0.0) ? ((double)total_rays / 1e6) / (total_ms / 1e3) : 0.0;
    double mrays_trace = (ms_trace > 0.0) ? ((double)total_rays / 1e6) / (ms_trace / 1e3) : 0.0;

    std::cout << "\n=== RESULTS ===\n"
              << "Primary rays:       " << primary_rays << "\n"
              << "Total rays traced:  " << total_rays << "\n"
              << "Secondary rays:     " << (total_rays - primary_rays) << "\n"
              << "Generate time:      " << ms_gen << " ms\n"
              << "Trace time:         " << ms_trace << " ms\n"
              << "Total time:         " << total_ms << " ms\n"
              << "Throughput (total): " << mrays_total << " MRays/s\n"
              << "Throughput (trace): " << mrays_trace << " MRays/s\n"
              << "Kernel launches:    2\n";

    if (leftover != 0)
        std::cerr << "WARNING: " << leftover << " rays still alive after trace!\n";

    // ── Save image ──
    std::vector<Pixel> h_img(IMG_W * IMG_H);
    HIP_CHECK(hipMemcpy(h_img.data(), d_img, IMG_W * IMG_H * sizeof(Pixel), hipMemcpyDeviceToHost));

    std::string scene_str = (scene_id == 0) ? "complex" : "cornell";
    // std::string filename = "rt_gwfq_" + scene_str + ".ppm";
#if defined(USE_GLFQ)
    std::string filename = "rt_glfq_" + scene_str + ".ppm";
#elif defined(USE_GWFQ)
    std::string filename = "rt_gwfq_" + scene_str + ".ppm";
#elif defined(USE_WFQ)
    std::string filename = "rt_wfq_" + scene_str + ".ppm";
#else
    std::string filename = "rt_sfq_" + scene_str + ".ppm";
#endif
    save_ppm(filename.c_str(), h_img, IMG_W, IMG_H);
    std::cout << "Image saved: " << filename << "\n";

    // ── CSV append ──
    const char* csv = "rt_queue_perf.csv";
    bool csv_exists = std::ifstream(csv).good();
    std::ofstream ofs(csv, std::ios::app);
    if (ofs) {
        if (!csv_exists){
            ofs << "GPU,QUEUE,SCENE,THREADS,BOUNCES,PRIMARY,TOTAL,GEN_MS,TRACE_MS,TOTAL_MS,MRAYS\n";
        }
        #if defined(USE_GLFQ)
                const char* queue_name = "GLFQ";
        #elif defined(USE_GWFQ)
                const char* queue_name = "GWFQ";
        #elif defined(USE_WFQ)
                const char* queue_name = "WFQ";
        #elif defined(USE_SFQ)
                const char* queue_name = "SFQ";
        #elif defined(USE_BQ)
                const char* queue_name = "BrokerQueue";
        #endif

        ofs << "\"" << gpu_name << "\"," 
            << queue_name << ","
            << scene_str << ","
            << threads_per_tile << ","
            << max_bounces << ","
            << total_rays << ","
            << total_ms << ","
            << mrays_trace
            << "\n";

    }

    // ── Cleanup ──
#if defined(USE_WFQ) || defined(USE_SFQ)
    HIP_CHECK(hipFree(d_handles));
#endif
    HIP_CHECK(hipFree(d_queues));
    HIP_CHECK(hipFree(d_rays));
    HIP_CHECK(hipFree(d_img));
    HIP_CHECK(hipFree(d_sph));
    HIP_CHECK(hipFree(d_pln));
    HIP_CHECK(hipFree(d_alive));
    HIP_CHECK(hipFree(d_traced));
    HIP_CHECK(hipEventDestroy(e0));
    HIP_CHECK(hipEventDestroy(e1));

    return 0;
}