/**
 * bfs_bench.hip — Level-synchronous BFS using GPU FIFO queues
 *
 * Compile:
 *   hipcc -O3 -std=c++17 bfs_bench.cpp -o out_pin/bfs_glfq              # lock-free (default) or put it in out_od/bfs_glfqa for MI300A
 *   hipcc -O3 -std=c++17 -DUSE_GWFQ bfs_bench.cpp -o out_pin/bfs_gwfq   # wait-free or put it in out_od/bfs_gwfqa for MI300A
 *
 * Run:
 *   ./out_pin/bfs_glfq --graph road_usa.mtx --threads 8192 --block 256
 *   ./out_pin/bfs_gwfq --graph road_usa.mtx --threads 8192 --block 256
 */

#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

/* ======================================================================
 * QUEUE SELECTION
 * ====================================================================== */

/* ======================================================================
 * QUEUE SELECTION
 * ====================================================================== */

#if defined(USE_GWFQ)
  #include "../gwf_queue.hpp"
  #include "../gwf_ring.hpp"
  using GWFCfg = gwf_32k;
  static constexpr uint32_t GWF_N = 32768;
  using QueueT = wf_mpmc_t<GWFCfg, GWF_N>;
  using QueueH = void; // Unused
  static constexpr char const* QUEUE_NAME = "GWFQ";

  struct bfs_worker_t { int dummy; };
  __device__ __forceinline__ void bfs_worker_init(bfs_worker_t* w, QueueH*, int) {}
  __device__ __forceinline__ void q_init(QueueT* q) { wf_mpmc_init<GWFCfg, GWF_N>(q); }
  __device__ __forceinline__ bool q_enqueue(QueueT* q, bfs_worker_t*, uint64_t v) { return Enqueue_Ptr<GWFCfg, GWF_N>(q, v); }
  __device__ __forceinline__ bool q_dequeue(QueueT* q, bfs_worker_t*, uint64_t* out) { return Dequeue_Ptr<GWFCfg, GWF_N>(q, out); }

#elif defined(USE_GLFQ)
  #include "../glf_queue.hpp"
  using QueueT = fq_mpmc_t;
  using QueueH = void; // Unused
  static constexpr char const* QUEUE_NAME = "GLFQ";

  struct bfs_worker_t { int dummy; };
  __device__ __forceinline__ void bfs_worker_init(bfs_worker_t* w, QueueH*, int) {}
  __device__ __forceinline__ void q_init(QueueT* q) { fq_init(q); }
  __device__ __forceinline__ bool q_enqueue(QueueT* q, bfs_worker_t*, uint64_t v) { return fq_enqueue_ptr(q, v); }
  __device__ __forceinline__ bool q_dequeue(QueueT* q, bfs_worker_t*, uint64_t* out) { return fq_dequeue_ptr(q, out); }

#elif defined(USE_WFQ)
  #include "../wfqueue_hip_opt.hpp"
  using QueueT = wf_queue;
  using QueueH = wf_handle;
  static constexpr char const* QUEUE_NAME = "WFQ";

  struct bfs_worker_t { QueueH* h; };
  __device__ __forceinline__ void bfs_worker_init(bfs_worker_t* w, QueueH* handles, int tid) { w->h = &handles[tid]; }
  
  __device__ __forceinline__ bool q_enqueue(QueueT* q, bfs_worker_t* w, uint64_t v) {
      return wf_enqueue(q, w->h, v + 1ULL);
  }
  __device__ __forceinline__ bool q_dequeue(QueueT* q, bfs_worker_t* w, uint64_t* out) {
      uint64_t v = wf_dequeue(q, w->h);
      if (v == WF_EMPTY) return false;
      *out = v - 1ULL;
      return true;
  }

  __device__ __forceinline__ void q_init(QueueT* q) { /* Handled by host init */ }

#elif defined(USE_BQ)
  #include "../bq.hpp"
  // BQ needs massive power-of-2 capacity for BFS!
  using QueueT = bq::BrokerQueue<uint64_t, 4194304, 65536>; 
  using QueueH = void; // Unused
  static constexpr char const* QUEUE_NAME = "BQ";

  struct bfs_worker_t { int dummy; };
  __device__ __forceinline__ void bfs_worker_init(bfs_worker_t* w, QueueH*, int) {}
  __device__ __forceinline__ void q_init(QueueT* q) { q->device_init_cooperative(); }
  __device__ __forceinline__ bool q_enqueue(QueueT* q, bfs_worker_t*, uint64_t v) { return q->enqueue(v) == bq::QueueStatus::Success; }
  __device__ __forceinline__ bool q_dequeue(QueueT* q, bfs_worker_t*, uint64_t* out) { return q->dequeue(*out) == bq::QueueStatus::Success; }

#elif defined(USE_SFQ)
  #include "../sfqueue_hip.hpp"
  #include "../sfqueue_hip.cpp"
  using QueueT = sfq_queue;
  using QueueH = sfq_handle;
  static constexpr char const* QUEUE_NAME = "SFQ";

  struct bfs_worker_t { QueueH* h; };
  __device__ __forceinline__ void bfs_worker_init(bfs_worker_t* w, QueueH* handles, int tid) { w->h = &handles[tid]; }
  __device__ __forceinline__ void q_init(QueueT* q) { /* Handled by host init */ }
  
  __device__ __forceinline__ bool q_enqueue(QueueT* q, bfs_worker_t* w, uint64_t v) {
      uint32_t item = (uint32_t)(v & 0xFFFFFFFFu);
      if (item == 0u) item = 1u;
      sfq_enqueue_blocking_u32(q, item);
      return true;
  }

  __device__ __forceinline__ bool q_dequeue(QueueT* q, bfs_worker_t* w, uint64_t* out) {
      while (true) {
          uint32_t item = 0;
          int rc = sfq_dequeue_nb_u32(q, &item);
          if (rc == SFQ_SUCCESS) {
              *out = (uint64_t)item;
              return true;
          }
          // Direct atomic reads since the sfq helpers aren't in the header
          uint32_t h = atomicAdd((unsigned int*)&q->head, 0u);
          uint32_t t = atomicAdd((unsigned int*)&q->tail, 0u);
          if (h >= t) {
              return false; // Truly empty
          }
          // Simple backoff loop
          for(int i=0; i<100; i++) __asm__ volatile("");
      }
  }
#endif

/* ======================================================================
 * VERTEX ENCODING: +1 shift avoids collision with queue sentinels
 * ====================================================================== */

__host__ __device__ __forceinline__
uint64_t encode_vtx(int v) { return (uint64_t)(uint32_t)v + 1ULL; }

__host__ __device__ __forceinline__
int decode_vtx(uint64_t x) { return (int)(x - 1ULL); }

/* ======================================================================
 * WAVE BALLOT / SHUFFLE — AMD 64-wide
 * ====================================================================== */

#ifdef __HIP_PLATFORM_AMD__
  using wave_mask_t = uint64_t;
  #define WAVE_FULL 0xFFFFFFFFFFFFFFFFULL

  __device__ __forceinline__
  wave_mask_t wave_ballot(int pred) {
      return (wave_mask_t)__ballot(pred);
  }

  __device__ __forceinline__
  int wave_shfl(int v, int lane) {
      return __shfl(v, lane);
  }

#else
  using wave_mask_t = uint32_t;
  #define WAVE_FULL 0xFFFFFFFFu

  __device__ __forceinline__
  wave_mask_t wave_ballot(int pred) {
      return (wave_mask_t)__ballot_sync(WAVE_FULL, pred);
  }

  __device__ __forceinline__
  int wave_shfl(int v, int lane) {
      return __shfl_sync(WAVE_FULL, v, lane);
  }
#endif

/* ======================================================================
 * CSR GRAPH + LOADERS
 * ====================================================================== */

struct CSR {
    int V = 0, E = 0;
    std::vector<int> offsets, indices;
};

static CSR csr_from_edges(int V, std::vector<std::pair<int,int>>& edges) {
    CSR g; g.V = V;
    std::vector<int> deg(V, 0);
    for (auto& [u,v] : edges)
        if ((unsigned)u < (unsigned)V && (unsigned)v < (unsigned)V) deg[u]++;
    g.offsets.resize(V + 1, 0);
    for (int i = 0; i < V; ++i) g.offsets[i+1] = g.offsets[i] + deg[i];
    g.E = g.offsets[V];
    g.indices.resize(g.E);
    std::vector<int> cur(V, 0);
    for (auto& [u,v] : edges)
        if ((unsigned)u < (unsigned)V && (unsigned)v < (unsigned)V)
            g.indices[g.offsets[u] + cur[u]++] = v;
    return g;
}

static CSR load_mtx(const std::string& path, bool undir = true) {
    std::ifstream f(path); std::string line;
    if (!f || !std::getline(f, line) || line.rfind("%%MatrixMarket",0) != 0) return {};
    int nr=0, nc=0; long long nz=0;
    while (std::getline(f, line)) {
        if (line.empty() || line[0]=='%') continue;
        std::istringstream(line) >> nr >> nc >> nz; break;
    }
    std::vector<std::pair<int,int>> edges;
    edges.reserve((size_t)nz * (undir?2:1));
    long long i,j,cnt=0;
    while (cnt < nz && std::getline(f, line)) {
        if (line.empty() || line[0]=='%') continue;
        std::istringstream iss(line); if (!(iss >> i >> j)) continue;
        --i; --j; cnt++;
        if (i==j || i<0 || j<0 || i>=nr || j>=nc) continue;
        edges.emplace_back((int)i,(int)j);
        if (undir) edges.emplace_back((int)j,(int)i);
    }
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    return csr_from_edges(std::max(nr,nc), edges);
}

static CSR load_snap(const std::string& path, bool undir = true) {
    std::ifstream f(path); std::string line;
    std::vector<std::pair<int,int>> edges;
    long long mn=(1LL<<60), mx=-1;
    while (std::getline(f, line)) {
        if (line.empty() || line[0]=='#' || line[0]=='%') continue;
        long long u,v; std::istringstream iss(line);
        if (!(iss >> u >> v) || u==v) continue;
        mn = std::min(mn, std::min(u,v));
        mx = std::max(mx, std::max(u,v));
        edges.emplace_back((int)u,(int)v);
        if (undir) edges.emplace_back((int)v,(int)u);
    }
    if (mn == 1) { for (auto& e : edges) { e.first--; e.second--; } mx--; }
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    return csr_from_edges((int)(mx+1), edges);
}

static CSR build_grid(int R, int C) {
    std::vector<std::pair<int,int>> edges;
    auto id = [C](int r, int c){ return r*C+c; };
    for (int r=0;r<R;r++) for (int c=0;c<C;c++) {
        int u = id(r,c);
        if (r+1<R) { int v=id(r+1,c); edges.push_back({u,v}); edges.push_back({v,u}); }
        if (c+1<C) { int v=id(r,c+1); edges.push_back({u,v}); edges.push_back({v,u}); }
    }
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    return csr_from_edges(R*C, edges);
}

static std::string gname_from_path(const std::string& path) {
    std::string name = path.substr(path.find_last_of("/\\") + 1);
    // return full name of graph file, e.g. "road_usa.mtx"
    return name;

}

/* ======================================================================
 * HIP HELPERS
 * ====================================================================== */

#define HIP_CHECK(x) do { hipError_t e=(x); if(e!=hipSuccess){ \
    std::cerr<<"HIP error "<<__FILE__<<":"<<__LINE__<<" "<<hipGetErrorString(e)<<"\n"; \
    std::exit(1); }} while(0)

struct Timer {
    std::chrono::high_resolution_clock::time_point t;
    void start() { t = std::chrono::high_resolution_clock::now(); }
    double ms() const {
        return std::chrono::duration<double,std::milli>(
            std::chrono::high_resolution_clock::now() - t).count();
    }
};

/* ======================================================================
 * KERNELS
 * ====================================================================== */

#ifndef INLINE_DEG
#define INLINE_DEG 32
#endif

#ifndef HUB_DEG
#define HUB_DEG 4096
#endif

__global__ void init_queue_kernel(QueueT* q) {
    #if defined(USE_BQ)
    // cooperative init: all threads in this one block participate
    q_init(q);
    #else
    if (blockIdx.x == 0 && threadIdx.x == 0)
        q_init(q);
    #endif
}

__global__ void enqueue_source_kernel(
    QueueT* q,
#if defined(USE_WFQ) || defined(USE_SFQ)
    QueueH* handles_in, QueueH* handles_out,
#endif
    int src
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    bfs_worker_t w_in, w_out;
#if defined(USE_WFQ) || defined(USE_SFQ)
    bfs_worker_init(&w_in, handles_in, tid);
    bfs_worker_init(&w_out, handles_out, tid);
#else
    bfs_worker_init(&w_in, nullptr, tid);
    bfs_worker_init(&w_out, nullptr, tid);
#endif
    if (blockIdx.x == 0 && threadIdx.x == 0)
        q_enqueue(q, &w_in, encode_vtx(src));
}

/**
 * Level-expand kernel: dequeue from q_in, process neighbors, enqueue to q_out.
 */
__global__ void bfs_expand_kernel(
    QueueT* q_in, QueueT* q_out,
#if defined(USE_WFQ) || defined(USE_SFQ)
    QueueH* handles_in, QueueH* handles_out,
#endif
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    int* __restrict__ level,
    int curr_level,
    unsigned int* __restrict__ next_count,
    unsigned long long* __restrict__ edges_total,
    int* __restrict__ hubs,
    unsigned int* __restrict__ hub_count,
    int num_threads,
    int inline_thresh,
    int hub_thresh,
    int enable_hubs)
{
    extern __shared__ char smem[];
    auto s_edges = (unsigned long long*)smem;
    auto s_next  = (unsigned int*)(s_edges + blockDim.x);

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_threads) return;

    bfs_worker_t w_in, w_out;
#if defined(USE_WFQ) || defined(USE_SFQ)
    bfs_worker_init(&w_in, handles_in, tid);
    bfs_worker_init(&w_out, handles_out, tid);
#else
    bfs_worker_init(&w_in, nullptr, tid);
    bfs_worker_init(&w_out, nullptr, tid);
#endif

    const int lane = (int)(threadIdx.x % warpSize);

    unsigned long long loc_edges = 0;
    unsigned int loc_next = 0;
    int pending = -1;

    while (true) {
        uint64_t item = 0;
        bool got = q_dequeue(q_in, &w_in, &item);
        int u = got ? decode_vtx(item) : -1;

        if (u >= 0) {
            const int beg = offsets[u], end = offsets[u+1], deg = end - beg;

            if (deg <= inline_thresh) {
                for (int e = beg; e < end; ++e) {
                    loc_edges++;
                    int w = indices[e];
                    if (atomicCAS(&level[w], -1, curr_level+1) == -1) {
                        q_enqueue(q_out, &w_out, encode_vtx(w));
                        loc_next++;
                    }
                }
            } else if (enable_hubs && deg >= hub_thresh) {
                unsigned int idx = atomicAdd(hub_count, 1u);
                hubs[idx] = u;
            } else {
                pending = u;
            }
        }

        /* Warp/wavefront cooperative processing of medium-degree vertices */
        while (true) {
            wave_mask_t mask = wave_ballot(pending >= 0);
            if (mask == 0) break;
            int owner = __ffsll((unsigned long long)mask) - 1;
            int owner_u = wave_shfl(pending, owner);
            const int beg = offsets[owner_u], end = offsets[owner_u+1];

            for (int e = beg + lane; e < end; e += (int)warpSize) {
                loc_edges++;
                int w = indices[e];
                if (atomicCAS(&level[w], -1, curr_level+1) == -1) {
                    q_enqueue(q_out, &w_out, encode_vtx(w));
                    loc_next++;
                }
            }
            if (lane == owner) pending = -1;
        }

        /* Exit when entire wavefront is idle */
        wave_mask_t all_empty = wave_ballot(u < 0);
        if (all_empty == (wave_mask_t)WAVE_FULL) break;
    }

    /* Block reduction → one atomicAdd per block */
    s_edges[threadIdx.x] = loc_edges;
    s_next[threadIdx.x]  = loc_next;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) {
            s_edges[threadIdx.x] += s_edges[threadIdx.x + s];
            s_next[threadIdx.x]  += s_next[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (s_edges[0]) atomicAdd(edges_total, s_edges[0]);
        if (s_next[0])  atomicAdd(next_count, s_next[0]);
    }
}

/** Block-cooperative hub kernel for very high-degree vertices. */
__global__ void bfs_hub_kernel(
    const int* __restrict__ hubs,
#if defined(USE_WFQ) || defined(USE_SFQ)
    QueueH* handles_in, QueueH* handles_out,
#endif
    const unsigned int* __restrict__ hub_count,
    QueueT* q_out,
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    int* __restrict__ level,
    int curr_level,
    unsigned int* __restrict__ next_count,
    unsigned long long* __restrict__ edges_total,
    int num_threads)
{
    extern __shared__ char smem[];
    auto s_edges = (unsigned long long*)smem;
    auto s_next  = (unsigned int*)(s_edges + blockDim.x);

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_threads) return;

    unsigned long long loc_edges = 0;
    unsigned int loc_next = 0;

    const unsigned int nhubs = *hub_count;
    bfs_worker_t w_in, w_out;
#if defined(USE_WFQ) || defined(USE_SFQ)
    bfs_worker_init(&w_in, handles_in, tid);
    bfs_worker_init(&w_out, handles_out, tid);
#else
    bfs_worker_init(&w_in, nullptr, tid);
    bfs_worker_init(&w_out, nullptr, tid);
#endif
    for (unsigned int h = (unsigned int)blockIdx.x; h < nhubs; h += (unsigned int)gridDim.x) {
        int u = hubs[h];
        int beg = offsets[u], end = offsets[u+1];
        for (int e = beg + (int)threadIdx.x; e < end; e += (int)blockDim.x) {
            loc_edges++;
            int w = indices[e];
            if (atomicCAS(&level[w], -1, curr_level+1) == -1) {
                q_enqueue(q_out, &w_out, encode_vtx(w));
                loc_next++;
            }
        }
        __syncthreads();
    }

    s_edges[threadIdx.x] = loc_edges;
    s_next[threadIdx.x]  = loc_next;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) {
            s_edges[threadIdx.x] += s_edges[threadIdx.x + s];
            s_next[threadIdx.x]  += s_next[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (s_edges[0]) atomicAdd(edges_total, s_edges[0]);
        if (s_next[0])  atomicAdd(next_count, s_next[0]);
    }
}

/* ======================================================================
 * BFS RUNNER
 * ====================================================================== */

struct RunResult {
    double total_ms=0, kernel_ms=0, init_ms=0;
    unsigned long long edges_scanned=0;
    unsigned int reached=0;
    int levels=0;
};

static RunResult run_bfs(
    const CSR& g, int src, int threads, int block_size,
    int* d_off, int* d_idx, int* d_level,
    QueueT* d_q_a, QueueT* d_q_b,
#if defined(USE_WFQ) || defined(USE_SFQ)
    QueueH* d_h_a, QueueH* d_h_b, // ADDED THESE ARGUMENTS
#endif
    unsigned int* h_next, unsigned int* d_next,
    unsigned long long* h_edges, unsigned long long* d_edges,
    int* d_hubs, unsigned int* d_hub_count)
{
    RunResult rr;
    const int grid = threads / block_size;
    const size_t shmem = (size_t)block_size * (sizeof(unsigned long long) + sizeof(unsigned int));

    Timer t_total; t_total.start();
    Timer t_init; t_init.start();

    /* Reset state for this BFS run */
    HIP_CHECK(hipMemset(d_level, 0xFF, (size_t)g.V * sizeof(int)));
    *h_next = 0; *h_edges = 0;

    /* Re-init both queues (full reset: counters + entries + fq indices) */
#if defined(USE_WFQ)
    HIP_CHECK(hipDeviceSynchronize());
    wf_queue_reset_for_bfs(d_q_a, d_h_a, threads);
    wf_queue_reset_for_bfs(d_q_b, d_h_b, threads);
#elif defined(USE_SFQ)
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemset(d_q_a, 0, sizeof(sfq_queue)));
    HIP_CHECK(hipMemset(d_q_b, 0, sizeof(sfq_queue)));
    int nprocs = threads;
    HIP_CHECK(hipMemcpy(&d_q_a->nprocs, &nprocs, sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(&d_q_b->nprocs, &nprocs, sizeof(int), hipMemcpyHostToDevice));
#else
    #if defined(USE_BQ)
        init_queue_kernel<<<1,256>>>(d_q_a);
        init_queue_kernel<<<1,256>>>(d_q_b);
    #else
        init_queue_kernel<<<1,1>>>(d_q_a);
        init_queue_kernel<<<1,1>>>(d_q_b);
    #endif
    HIP_CHECK(hipDeviceSynchronize());
#endif

    /* Set source */
    if (src < 0 || src >= g.V) src = 0;
    int zero = 0;
    HIP_CHECK(hipMemcpy(d_level + src, &zero, sizeof(int), hipMemcpyHostToDevice));
    #if defined(USE_WFQ) || defined(USE_SFQ)
    enqueue_source_kernel<<<1,1>>>(d_q_a, d_h_a, d_h_b, src);
#else
    enqueue_source_kernel<<<1,1>>>(d_q_a, src);
#endif
    HIP_CHECK(hipDeviceSynchronize());

    rr.init_ms = t_init.ms();

    /* Ping-pong pointers */
    QueueT* q_curr = d_q_a;
    QueueT* q_next = d_q_b;
#if defined(USE_WFQ) || defined(USE_SFQ)
    QueueH* hnd_curr = d_h_a;
    QueueH* hnd_next = d_h_b;
#endif

    hipEvent_t ev_a, ev_b;
    HIP_CHECK(hipEventCreate(&ev_a));
    HIP_CHECK(hipEventCreate(&ev_b));

    unsigned int frontier = 1;
    unsigned int reached = 1;
    int curr_level = 0;

    while (frontier > 0) {
        *h_next = 0;
        HIP_CHECK(hipMemsetAsync(d_hub_count, 0, sizeof(unsigned int), 0));

        HIP_CHECK(hipEventRecord(ev_a, 0));

        bfs_expand_kernel<<<grid, block_size, shmem>>>(
            q_curr, q_next,
        #if defined(USE_WFQ) || defined(USE_SFQ)
            hnd_curr, hnd_next,
        #endif
            d_off, d_idx, d_level, curr_level,
            d_next, d_edges,
            d_hubs, d_hub_count,
            threads, INLINE_DEG, HUB_DEG, 1);

        bfs_hub_kernel<<<grid, block_size, shmem>>>(
            d_hubs, 
        #if defined(USE_WFQ) || defined(USE_SFQ)
            hnd_curr, hnd_next,
        #endif
            d_hub_count, q_next,
            d_off, d_idx, d_level, curr_level,
            d_next, d_edges, threads);

        HIP_CHECK(hipEventRecord(ev_b, 0));
        HIP_CHECK(hipEventSynchronize(ev_b));

        float ms = 0; HIP_CHECK(hipEventElapsedTime(&ms, ev_a, ev_b));
        rr.kernel_ms += (double)ms;

        frontier = *h_next;
        reached += frontier;

        /* Swap queues — NO RESET NEEDED between levels.*/
        std::swap(q_curr, q_next);
    #if defined(USE_WFQ) || defined(USE_SFQ)
        std::swap(hnd_curr, hnd_next);
    #endif
        curr_level++;
    }

    HIP_CHECK(hipEventDestroy(ev_a));
    HIP_CHECK(hipEventDestroy(ev_b));

    rr.total_ms = t_total.ms();
    rr.edges_scanned = *h_edges;
    rr.reached = reached;
    rr.levels = curr_level;
    return rr;
}

/* ======================================================================
 * STATS
 * ====================================================================== */

struct Stats { double med=0, q1=0, q3=0, iqr=0; };
static Stats stats(std::vector<double> v) {
    Stats s; if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    auto pct = [&](double p) {
        double i = p*(v.size()-1); size_t a=(size_t)i;
        size_t b=std::min(a+1,v.size()-1); double f=i-(double)a;
        return v[a]*(1-f)+v[b]*f;
    };
    s.q1=pct(0.25); s.med=pct(0.5); s.q3=pct(0.75); s.iqr=s.q3-s.q1;
    return s;
}

/* ======================================================================
 * MAIN
 * ====================================================================== */

int main(int argc, char** argv) {
    /* Defaults */
    std::string graph_path;
    int src=0, threads=8192, block=256, warmup=3, iters=10;
    int grid_r=0, grid_c=0;
    bool verbose=false;

    for (int i=1; i<argc; ++i) {
        std::string a=argv[i];
        auto next = [&]{ return (i+1<argc) ? argv[++i] : (char*)""; };
        if (a=="--graph")   graph_path=next();
        else if (a=="--grid") { grid_r=atoi(next()); grid_c=atoi(next()); }
        else if (a=="--src")     src=atoi(next());
        else if (a=="--threads") threads=atoi(next());
        else if (a=="--block")   block=atoi(next());
        else if (a=="--warmup")  warmup=atoi(next());
        else if (a=="--iters")   iters=atoi(next());
        else if (a=="--verbose") verbose=atoi(next())!=0;
        else { std::cout<<"Unknown: "<<a<<"\n"; return 1; }
    }

    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    const char* gpu_env = std::getenv("GPU_NAME");
    const std::string gpu_name = (gpu_env && gpu_env[0] != '\0') ? std::string(gpu_env)
                                                                   : std::string(prop.name);
    std::cout << "GPU: " << gpu_name << "  CUs=" << prop.multiProcessorCount << "\n";
    std::cout << "Queue: " << QUEUE_NAME << "\n";

    /* Load graph */
    CSR g;
    if (grid_r > 0 && grid_c > 0) {
        g = build_grid(grid_r, grid_c);
        std::cout << "Graph: grid " << grid_r << "x" << grid_c << "\n";
    } else if (!graph_path.empty()) {
        bool mtx = graph_path.size()>=4 && graph_path.substr(graph_path.size()-4)==".mtx";
        g = mtx ? load_mtx(graph_path) : load_snap(graph_path);
        std::cout << "Graph: " << graph_path << "\n";
    } else {
        std::cout << "Usage: --graph <path> | --grid R C\n"; return 1;
    }
    if (g.V<=0) { std::cerr << "Empty graph\n"; return 1; }

    std::cout << "V=" << g.V << " E=" << g.E
              << " threads=" << threads << " block=" << block << "\n";

    /* Device allocations */
    int *d_off=nullptr, *d_idx=nullptr, *d_level=nullptr;
    HIP_CHECK(hipMalloc(&d_off, (size_t)(g.V+1)*sizeof(int)));
    HIP_CHECK(hipMalloc(&d_idx, (size_t)g.E*sizeof(int)));
    HIP_CHECK(hipMalloc(&d_level, (size_t)g.V*sizeof(int)));
    HIP_CHECK(hipMemcpy(d_off, g.offsets.data(), (size_t)(g.V+1)*sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_idx, g.indices.data(), (size_t)g.E*sizeof(int), hipMemcpyHostToDevice));

    /* Pinned host memory */
    unsigned int *h_next=nullptr, *d_next=nullptr;
    unsigned long long *h_edges=nullptr, *d_edges=nullptr;
    HIP_CHECK(hipHostMalloc(&h_next, sizeof(unsigned int)));
    HIP_CHECK(hipHostMalloc(&h_edges, sizeof(unsigned long long)));
    HIP_CHECK(hipHostGetDevicePointer((void**)&d_next, h_next, 0));
    HIP_CHECK(hipHostGetDevicePointer((void**)&d_edges, h_edges, 0));

    int* d_hubs=nullptr; unsigned int* d_hub_count=nullptr;
    HIP_CHECK(hipMalloc(&d_hubs, (size_t)g.V*sizeof(int)));
    HIP_CHECK(hipMalloc(&d_hub_count, sizeof(unsigned int)));

    /* Queue allocations */
QueueT *d_q_a = nullptr, *d_q_b = nullptr;
#if defined(USE_WFQ) || defined(USE_SFQ)
    QueueH *d_h_a = nullptr, *d_h_b = nullptr;
#endif

#if defined(USE_WFQ)
    wf_queue_host_init_for_bfs(&d_q_a, &d_h_a, threads, g.E, g.V, block);
    wf_queue_host_init_for_bfs(&d_q_b, &d_h_b, threads, g.E, g.V, block);
#elif defined(USE_SFQ)
    sfq_queue_host_init(&d_q_a, &d_h_a, threads);
    sfq_queue_host_init(&d_q_b, &d_h_b, threads);
#else
    HIP_CHECK(hipMalloc(&d_q_a, sizeof(QueueT)));
    HIP_CHECK(hipMalloc(&d_q_b, sizeof(QueueT)));
#endif
    /* Benchmark runs */
    std::vector<double> v_total, v_kernel, v_init, v_teps;
    unsigned int last_reached=0; int last_levels=0;

    for (int r = 0; r < warmup + iters; ++r) {
        RunResult rr = run_bfs(g, src, threads, block,
            d_off, d_idx, d_level,
            d_q_a, d_q_b,
        #if defined(USE_WFQ) || defined(USE_SFQ)
            d_h_a, d_h_b, // PASS HANDLES HERE
        #endif
            h_next, d_next,
            h_edges, d_edges,
            d_hubs, d_hub_count);

        if (r >= warmup) {
            v_total.push_back(rr.total_ms);
            v_kernel.push_back(rr.kernel_ms);
            v_init.push_back(rr.init_ms);
            double ks = rr.kernel_ms / 1000.0;
            v_teps.push_back(ks > 0 ? (double)rr.edges_scanned / ks : 0);
        }
        last_reached = rr.reached;
        last_levels = rr.levels;

        if (verbose)
            std::cout << (r<warmup?"[warm] ":"[run]  ")
                      << "total=" << rr.total_ms << " ms"
                      << " kernel=" << rr.kernel_ms << " ms"
                      << " edges=" << rr.edges_scanned
                      << " reached=" << rr.reached
                      << " levels=" << rr.levels << "\n";
    }

    /* Report */
    auto st = stats(v_total);
    auto sk = stats(v_kernel);
    auto si = stats(v_init);
    auto se = stats(v_teps);

    std::cout << "\n== Summary (" << iters << " runs, " << warmup << " warmup) ==\n";
    std::cout << "Reached: " << last_reached << "/" << g.V
              << " Levels: " << last_levels << "\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  total:  med=" << st.med << " ms  IQR=" << st.iqr << "\n";
    std::cout << "  kernel: med=" << sk.med << " ms  IQR=" << sk.iqr << "\n";
    std::cout << "  init:   med=" << si.med << " ms  IQR=" << si.iqr << "\n";
    std::cout << "  TEPS:   med=" << std::setprecision(4) << (se.med/1e9)
              << " Gedges/s  IQR=" << (se.iqr/1e9) << "\n";

    // put result into a csv file for easier plotting later
    {
        const char* csv_env = std::getenv("CSV_NAME");
        const char* csv = (csv_env && csv_env[0] != '\0') ? csv_env : "bfs_bench_final2.csv";
        bool csv_exists = std::ifstream(csv).good();
        std::ofstream ofs(csv, std::ios::app);
        ofs << gpu_name << "," << QUEUE_NAME << "," << gname_from_path(graph_path) << "," << g.V << "," << g.E
            << "," << threads << "," << block
            << "," << st.med << "," << st.iqr
            << "," << (se.med/1e9) << "," << (se.iqr/1e9) << "\n";
    }

    /* Cleanup */
    hipFree(d_off); hipFree(d_idx); hipFree(d_level);
    hipHostFree(h_next); hipHostFree(h_edges);
    hipFree(d_hubs); hipFree(d_hub_count);
    hipFree(d_q_a); hipFree(d_q_b);
    return 0;
}
