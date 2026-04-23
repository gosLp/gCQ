// throughput_bench.cpp
// Fixed-duration timed throughput harness.
// Supports either: G-WFQ, G-LFQ, G-WFQ-YMC, BQ, SFQ 

// Compile examples:
//   hipcc -O3 -std=c++17 -DUSE_GWFQ  throughput_bench.cpp -o gwfq
//   hipcc -O3 -std=c++17 -DUSE_GLFQ -DFQ_N=65536 throughput_bench.cpp -o glfq

// Optional knobs:
//   -DRUN_MS=500
//   -DWARMUP_MS=100
//   -DCHUNK_OPS=64
//   -DBLOCK_SIZE=256
//   -DONLY_BALANCED=1
//   -DONLY_SPLIT=1

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <stdint.h>
#include <fstream>

#include <roctracer/roctx.h>

#ifndef ENABLE_ROCTX
#define ENABLE_ROCTX 0
#endif

#ifndef RUN_MS
#define RUN_MS 500
#endif

#ifndef WARMUP_MS
#define WARMUP_MS 100
#endif

#ifndef CHUNK_OPS
#define CHUNK_OPS 64
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

#ifndef ONLY_BALANCED
#define ONLY_BALANCED 0
#endif

#ifndef ONLY_SPLIT
#define ONLY_SPLIT 0
#endif

#ifndef CSV_FILE
#define CSV_FILE "benchmark_results.csv"
#endif

#ifndef MODE_FIFO
#define MODE_FIFO 0
#endif

#ifndef FIFO_OPS_PER_THREAD
#define FIFO_OPS_PER_THREAD 64
#endif

#ifndef FIFO_BLOCK_SIZE
#define FIFO_BLOCK_SIZE BLOCK_SIZE
#endif

// -----------------------------------------------------------------------------
// HIP helpers
// -----------------------------------------------------------------------------

#define HIP_CHECK(call) do {                                      \
    hipError_t _e = (call);                                       \
    if (_e != hipSuccess) {                                       \
        std::cerr << "HIP error " << __FILE__ << ":" << __LINE__  \
                  << " : " << hipGetErrorString(_e) << "\n";      \
        std::exit(1);                                             \
    }                                                             \
} while (0)

static inline void roctx_push(const std::string& s) {
#if ENABLE_ROCTX
    roctxRangePushA(s.c_str());
#endif
}

static inline void roctx_pop() {
#if ENABLE_ROCTX
    roctxRangePop();
#endif
}

#if defined(USE_GWFQ) && defined(USE_GLFQ) && defined(USE_WFQ) && defined(USE_BQ) && defined(USE_SFQ)
#error "Choose only one backend: USE_GWFQ or USE_GLFQ or USE_WFQ or USE_BQ or USE_SFQ"
#endif

#if !defined(USE_GWFQ) && !defined(USE_GLFQ) && !defined(USE_WFQ) && !defined(USE_BQ) && !defined(USE_SFQ)
#error "Define one: USE_GWFQ or USE_GLFQ or USE_WFQ or USE_BQ or USE_SFQ"
#endif

// -----------------------------------------------------------------------------
// Backend selection
// -----------------------------------------------------------------------------

#if defined(USE_GWFQ)

// #include "gwf_queue.hpp"
#include "gwf_ring.hpp"
// #include "gwf_ring_warpowned_experimental.hpp"

#ifndef GWF_N
#define GWF_N 65536
#endif

using queue_t  = wf_mpmc_t<gwf_64k, GWF_N>;
using handle_t = int; // dummy handle, preserved for harness shape

static constexpr const char* QUEUE_NAME = "gWFQ";
static constexpr bool NEEDS_HANDLES = true;

__global__ void queue_init_kernel(queue_t* q) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        wf_mpmc_init<gwf_64k, GWF_N>(q);
    }
}

static __device__ __forceinline__
bool test_enqueue(queue_t* q, handle_t* /*h*/, uint64_t v) {
    return Enqueue_Ptr<gwf_64k, GWF_N>(q, v);
}

static __device__ __forceinline__
bool test_dequeue(queue_t* q, handle_t* /*h*/, uint64_t* out) {
    return Dequeue_Ptr<gwf_64k, GWF_N>(q, out);
}

static inline void print_backend_info() {
    std::cout << "GWF_N=" << GWF_N
              << " Config::MAX_THREADS=" << gwf_64k::MAX_THREADS << "\n\n";
}

#elif defined(USE_GLFQ)

#include "glf_queue.hpp"

using queue_t  = fq_mpmc_t;
using handle_t = int; // unused, but keeps kernel signatures uniform

static constexpr const char* QUEUE_NAME = "GLFQ";
static constexpr bool NEEDS_HANDLES = false;

__global__ void queue_init_kernel(queue_t* q) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        fq_init(q);
    }
}

static __device__ __forceinline__
bool test_enqueue(queue_t* q, handle_t* /*h*/, uint64_t v) {
    return fq_enqueue_ptr(q, v);
}

static __device__ __forceinline__
bool test_dequeue(queue_t* q, handle_t* /*h*/, uint64_t* out) {
    return fq_dequeue_ptr(q, out);
}

static inline void print_backend_info() {
    std::cout << "FQ_N=" << FQ_N
              << " FQ_SLOTS=" << FQ_SLOTS
              << " FQ_CYCLE_BITS=" << FQ_CYCLE_BITS << "\n\n";
}

#elif defined(USE_WFQ)
    #include "wfqueue_hip_opt.hpp"
    
    using queue_t = wf_queue;
    using handle_t = wf_handle;

    static constexpr const char* QUEUE_NAME = "WFQ-YMC";
    static constexpr bool NEEDS_HANDLES = true;

    // For this backend, creation is done by its provided host helper.
    static inline void backend_create_queue(queue_t** d_q, handle_t** d_h, int threads) {
        wf_queue_host_init(d_q, d_h, threads);
    }

    static inline void backend_destroy_queue(queue_t* d_q, handle_t* d_h) {
        if (d_q) {
            // Need to free the managed segment pool too.
            wf_queue hq{};
            HIP_CHECK(hipMemcpy(&hq, d_q, sizeof(queue_t), hipMemcpyDeviceToHost));
            if (hq.pool.segments) HIP_CHECK(hipFree(hq.pool.segments));
        }
        if (d_q) HIP_CHECK(hipFree(d_q));
        if (d_h) HIP_CHECK(hipFree(d_h));
    }

    static __device__ __forceinline__
    bool test_enqueue(queue_t* q, handle_t* h, uint64_t v) {
        wf_enqueue(q, h, v);
        return true;
    }

    static __device__ __forceinline__
    bool test_dequeue(queue_t* q, handle_t* h, uint64_t* out) {
        uint64_t v = wf_dequeue(q, h);
        if (v == WF_EMPTY) return false;
        *out = v;
        return true;
    }

    static inline void print_backend_info() {
        std::cout << "WF_SEGMENT_SIZE=" << WF_SEGMENT_SIZE
                << " WF_PATIENCE=" << WF_PATIENCE
                << " WF_PREALLOC_OPS_PER_THREAD=" << WF_PREALLOC_OPS_PER_THREAD
                << "\n\n";
    }

#elif defined(USE_SFQ)

    #include "sfqueue_hip.hpp"
    #include "sfqueue_hip.cpp"

    using queue_t = sfq_queue;
    using handle_t = sfq_handle;

    static constexpr const char* QUEUE_NAME = "SFQ";
    static constexpr bool NEEDS_HANDLES = true;

    static inline void backend_create_queue(queue_t** d_q, handle_t** d_h, int threads) {
        sfq_queue_host_init(d_q, d_h, threads);
    }

    static inline void backend_destroy_queue(queue_t* d_q, handle_t* d_h) {
        sfq_queue_destroy(d_q, d_h);
    }

    static __device__ __forceinline__ uint32_t sfq_pack_value(uint64_t v) {
        uint32_t x = static_cast<uint32_t>(v ^ (v >> 32));
        return (x == 0u) ? 1u : x;
    }

    static __device__ __forceinline__
    bool test_enqueue(queue_t* q, handle_t* /*h*/, uint64_t v) {
        const int rc = sfq_enqueue_blocking_u32(q, sfq_pack_value(v));
        return (rc == SFQ_SUCCESS);
    }

    static __device__ __forceinline__
    bool test_dequeue(queue_t* q, handle_t* /*h*/, uint64_t* out) {
        uint32_t item = 0;
        const int rc = sfq_dequeue_blocking_u32(q, &item);
        if (rc != SFQ_SUCCESS) {
            *out = 0;
            return false;
        }
        *out = static_cast<uint64_t>(item);
        return true;
    }

    static __device__ __forceinline__
    bool test_enqueue_split(queue_t* q, handle_t* /*h*/, uint64_t v) {
        const int rc = sfq_enqueue_nb_u32(q, sfq_pack_value(v));
        return (rc == SFQ_SUCCESS);
    }

    static __device__ __forceinline__
    bool test_dequeue_split(queue_t* q, handle_t* /*h*/, uint64_t* out) {
        uint32_t item = 0;
        const int rc = sfq_dequeue_nb_u32(q, &item);
        if (rc != SFQ_SUCCESS) {
            *out = 0;
            return false;
        }
        *out = static_cast<uint64_t>(item);
        return true;
    }

    static inline void print_backend_info() {
        std::cout << "SFQ_QUEUE_LENGTH=" << SFQ_QUEUE_LENGTH
                << " SFQ_MAX_THREADS=" << SFQ_MAX_THREADS
                << " SFQ_BACKOFF_ITERS=" << SFQ_BACKOFF_ITERS
                << "\n\n";
    }

#elif defined(USE_BQ)

    #include "bq.hpp"

    #ifndef BQ_CAPACITY
    #define BQ_CAPACITY 65536
    #endif

    #ifndef BQ_MAX_THREADS
    #define BQ_MAX_THREADS 65536
    #endif

    static constexpr const char* QUEUE_NAME = "BQ";
        static constexpr bool NEEDS_HANDLES = true;

    using value_t = uint64_t;
    using queue_t = bq::BrokerQueue<value_t, BQ_CAPACITY, BQ_MAX_THREADS>;

    struct handle_t {
        uint32_t tid;
    };

    static inline void queue_host_init(queue_t** d_q,
                                    handle_t** d_handles,
                                    int num_threads) {
        hipMalloc((void**)d_q, sizeof(queue_t));
        hipMalloc((void**)d_handles, sizeof(handle_t) * (size_t)num_threads);

        queue_t hq{};
        hq.host_init();
        hipMemcpy(*d_q, &hq, sizeof(queue_t), hipMemcpyHostToDevice);

        std::vector<handle_t> hh((size_t)num_threads);
        for (int i = 0; i < num_threads; ++i) {
            hh[i].tid = (uint32_t)i;
        }
        hipMemcpy(*d_handles, hh.data(),
                sizeof(handle_t) * (size_t)num_threads,
                hipMemcpyHostToDevice);
    }

    static inline void backend_create_queue(queue_t** d_q,
                                        handle_t** d_handles,
                                        int num_threads) {
        hipMalloc((void**)d_q, sizeof(queue_t));
        hipMalloc((void**)d_handles, sizeof(handle_t) * (size_t)num_threads);

        queue_t hq{};
        hq.host_init();
        hipMemcpy(*d_q, &hq, sizeof(queue_t), hipMemcpyHostToDevice);

        std::vector<handle_t> hh((size_t)num_threads);
        for (int i = 0; i < num_threads; ++i) hh[i].tid = (uint32_t)i;

        hipMemcpy(*d_handles, hh.data(),
                sizeof(handle_t) * (size_t)num_threads,
                hipMemcpyHostToDevice);
    }

    static inline void queue_reset(queue_t* d_q,
                                handle_t* d_handles,
                                int num_threads) {
        queue_t hq{};
        hq.host_init();
        hipMemcpy(d_q, &hq, sizeof(queue_t), hipMemcpyHostToDevice);

        std::vector<handle_t> hh((size_t)num_threads);
        for (int i = 0; i < num_threads; ++i) {
            hh[i].tid = (uint32_t)i;
        }
        hipMemcpy(d_handles, hh.data(),
                sizeof(handle_t) * (size_t)num_threads,
                hipMemcpyHostToDevice);
    }

    static inline void backend_destroy_queue(queue_t* d_q,
                                    handle_t* d_handles) {
        if (d_q) hipFree(d_q);
        if (d_handles) hipFree(d_handles);
    }

    __global__ void queue_init_kernel(handle_t* handles, int num_threads) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < num_threads) {
            handles[tid].tid = (uint32_t)tid;
        }
    }

    __device__ __forceinline__ bool test_enqueue(queue_t* q,
                                                handle_t* /*h*/,
                                                uint64_t v) {
        return q->enqueue(v) == bq::QueueStatus::Success;
    }

    __device__ __forceinline__ bool test_dequeue(queue_t* q,
                                                handle_t* /*h*/,
                                                uint64_t* out) {
        return q->dequeue(*out) == bq::QueueStatus::Success;
    }

    static inline const char* backend_name() {
        return "BQ";
    }

    static inline void print_backend_info() {
        std::cout << "BQ_CAPACITY=" << BQ_CAPACITY
                << " BQ_MAX_THREADS=" << BQ_MAX_THREADS
                << "\n\n";
    }

#endif

struct QueueInstance;
static QueueInstance create_queue(int threads);
static void destroy_queue(QueueInstance& qi);


// -----------------------------------------------------------------------------
// Queue instance
// -----------------------------------------------------------------------------

struct QueueInstance {
    queue_t*  d_q = nullptr;
    handle_t* d_h = nullptr; // only used by gWFQ, kept for uniformity
};

static QueueInstance create_queue(int threads) {
    QueueInstance qi{};

#if defined(USE_WFQ) || defined(USE_BQ) || defined(USE_SFQ)
    backend_create_queue(&qi.d_q, &qi.d_h, threads);
#else
    HIP_CHECK(hipMalloc(&qi.d_q, sizeof(queue_t)));

    if (NEEDS_HANDLES) {
        HIP_CHECK(hipMalloc(&qi.d_h, threads * sizeof(handle_t)));
        HIP_CHECK(hipMemset(qi.d_h, 0, threads * sizeof(handle_t)));
    }

#if defined(USE_BQ)
    queue_init_kernel<<<(threads + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(qi.d_h, threads);
#else
    queue_init_kernel<<<1, 1>>>(qi.d_q);
#endif
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
#endif
    return qi;
}

static void destroy_queue(QueueInstance& qi) {
#if defined(USE_WFQ) || defined(USE_BQ) || defined(USE_SFQ)
    backend_destroy_queue(qi.d_q, qi.d_h);
#else
    if (qi.d_q) HIP_CHECK(hipFree(qi.d_q));
    if (qi.d_h) HIP_CHECK(hipFree(qi.d_h));
    qi.d_q = nullptr;
    qi.d_h = nullptr;
#endif
}


#if MODE_FIFO

#if defined(USE_SFQ)
#ifndef FIFO_SFQ_TID_BITS
#define FIFO_SFQ_TID_BITS 15
#endif

#ifndef FIFO_SFQ_SEQ_BITS
#define FIFO_SFQ_SEQ_BITS 16
#endif

static_assert((FIFO_SFQ_TID_BITS + FIFO_SFQ_SEQ_BITS) <= 31,
              "FIFO_SFQ_TID_BITS + FIFO_SFQ_SEQ_BITS must be <= 31");

static constexpr uint32_t FIFO_SFQ_TID_MASK = (1u << FIFO_SFQ_TID_BITS) - 1u;
static constexpr uint32_t FIFO_SFQ_SEQ_MASK = (1u << FIFO_SFQ_SEQ_BITS) - 1u;
#endif

struct FifoCheckResult {
    unsigned long long in_count = 0;
    unsigned long long out_count = 0;
    unsigned int err_oob = 0;
    unsigned int err_dup = 0;
    unsigned int err_order = 0;
    unsigned int err_missing = 0;
    unsigned int err_gt1 = 0;
    bool verified = false;
};

static __device__ __forceinline__ uint64_t fifo_make_token(uint32_t tid, uint32_t seq) {
#if defined(USE_SFQ)
    uint32_t tok32 = ((tid & FIFO_SFQ_TID_MASK) << FIFO_SFQ_SEQ_BITS)
                   | ((seq + 1u) & FIFO_SFQ_SEQ_MASK);
    return (uint64_t)tok32;
#else
    return ((uint64_t)tid << 32) | (uint64_t)(seq + 1u);
#endif
}

static __device__ __forceinline__ void fifo_decode_token(uint64_t tok, uint32_t& ptid, uint32_t& seq) {
#if defined(USE_SFQ)
    uint32_t t = (uint32_t)tok;
    ptid = (t >> FIFO_SFQ_SEQ_BITS) & FIFO_SFQ_TID_MASK;
    uint32_t s = t & FIFO_SFQ_SEQ_MASK;
    seq = (s == 0u) ? 0u : (s - 1u);
#else
    ptid = (uint32_t)(tok >> 32);
    uint32_t s = (uint32_t)(tok & 0xFFFFFFFFu);
    seq = (s == 0u) ? 0u : (s - 1u);
#endif
}

static __device__ __forceinline__ bool fifo_try_enqueue(queue_t* q, handle_t* h, uint64_t tok) {
#if defined(USE_SFQ)
    return sfq_enqueue_nb_u32(q, (uint32_t)tok) == SFQ_SUCCESS;
#else
    return test_enqueue(q, h, tok);
#endif
}

static __device__ __forceinline__ bool fifo_try_dequeue(queue_t* q, handle_t* h, uint64_t* out) {
#if defined(USE_SFQ)
    uint32_t item = 0;
    int rc = sfq_dequeue_nb_u32(q, &item);
    if (rc != SFQ_SUCCESS) {
        *out = 0;
        return false;
    }
    *out = (uint64_t)item;
    return true;
#else
    return test_dequeue(q, h, out);
#endif
}

__global__ void fifo_produce_kernel(queue_t* q, handle_t* handles, int num_threads, int ops_per_thread) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_threads) return;

    handle_t* my_h = handles ? &handles[tid] : nullptr;
    for (int i = 0; i < ops_per_thread; ++i) {
        uint64_t tok = fifo_make_token((uint32_t)tid, (uint32_t)i);
        while (!fifo_try_enqueue(q, my_h, tok)) {
            for (volatile int d = 0; d < 32; ++d) {}
        }
    }
}

__global__ void fifo_drain_kernel(queue_t* q,
                                  handle_t* handles,
                                  int num_threads,
                                  int ops_per_thread,
                                  unsigned long long total_to_consume,
                                  uint32_t* counts,
                                  uint32_t* last_seq,
                                  unsigned int* errors_oob,
                                  unsigned int* errors_dup,
                                  unsigned int* errors_order,
                                  unsigned long long* consumed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    handle_t* my_h = (tid < num_threads && handles) ? &handles[tid] : nullptr;

    while (atomicAdd(consumed, 0ull) < total_to_consume) {
        if (tid >= num_threads) continue;

        uint64_t v = 0;
        if (!fifo_try_dequeue(q, my_h, &v)) {
            for (volatile int d = 0; d < 32; ++d) {}
            continue;
        }

        unsigned long long my_ticket = atomicAdd(consumed, 1ull);
        if (my_ticket >= total_to_consume) {
            continue;
        }

        uint32_t ptid = 0;
        uint32_t seq = 0;
        fifo_decode_token(v, ptid, seq);

        if (ptid >= (uint32_t)num_threads || seq >= (uint32_t)ops_per_thread) {
            atomicAdd(errors_oob, 1u);
            continue;
        }

        size_t idx = (size_t)ptid * (size_t)ops_per_thread + (size_t)seq;
        uint32_t old = atomicAdd(&counts[idx], 1u);
        if (old >= 1u) atomicAdd(errors_dup, 1u);

        uint32_t prev = atomicMax(&last_seq[ptid], seq);
        if (prev != 0xFFFFFFFFu && seq <= prev) {
            atomicAdd(errors_order, 1u);
        }
    }
}

__global__ void fifo_verify_counts_kernel(const uint32_t* counts,
                                          size_t total_slots,
                                          unsigned int* zeros,
                                          unsigned int* gt1) {
    size_t gid = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t stride = (size_t)gridDim.x * (size_t)blockDim.x;
    for (size_t i = gid; i < total_slots; i += stride) {
        uint32_t c = counts[i];
        if (c == 0u) atomicAdd(zeros, 1u);
        else if (c > 1u) atomicAdd(gt1, 1u);
    }
}

static inline size_t fifo_capacity_hint(int /*threads*/) {
#if defined(USE_GWFQ)
    return (size_t)GWF_N;
#elif defined(USE_GLFQ)
    return (size_t)FQ_N;
#elif defined(USE_SFQ)
    return (size_t)SFQ_QUEUE_LENGTH;
#elif defined(USE_BQ)
    return (size_t)BQ_CAPACITY;
#else
    return 0;
#endif
}

static inline bool fifo_has_bounded_capacity() {
#if defined(USE_WFQ)
    return false;
#else
    return true;
#endif
}

static inline int fifo_effective_ops_per_thread(int threads) {
    int ops = std::max(1, FIFO_OPS_PER_THREAD);

    if (fifo_has_bounded_capacity()) {
        size_t cap = fifo_capacity_hint(threads);
        size_t cap_limited_ops = (threads > 0) ? (cap / (size_t)threads) : 0;
        if (cap_limited_ops == 0) {
            return 0;
        }
        if ((size_t)ops > cap_limited_ops) {
            ops = (int)cap_limited_ops;
        }
    }

#if defined(USE_SFQ)
    const int max_threads_enc = (1 << FIFO_SFQ_TID_BITS);
    if (threads > max_threads_enc) {
        return 0;
    }
    const int max_seq_enc = (int)FIFO_SFQ_SEQ_MASK;
    if (ops > max_seq_enc) {
        ops = max_seq_enc;
    }
#endif

    return std::max(1, ops);
}

static FifoCheckResult run_fifo_check_once(int threads) {
    FifoCheckResult res{};

    int ops_per_thread = fifo_effective_ops_per_thread(threads);
    if (ops_per_thread <= 0 || threads <= 0) {
        return res;
    }

    QueueInstance qi = create_queue(threads);
    HIP_CHECK(hipDeviceSynchronize());

    const unsigned long long total_to_consume =
        (unsigned long long)threads * (unsigned long long)ops_per_thread;
    res.in_count = total_to_consume;

    uint32_t* d_counts = nullptr;
    uint32_t* d_last_seq = nullptr;
    unsigned int* d_errors_oob = nullptr;
    unsigned int* d_errors_dup = nullptr;
    unsigned int* d_errors_order = nullptr;
    unsigned int* d_zeros = nullptr;
    unsigned int* d_gt1 = nullptr;
    unsigned long long* d_consumed = nullptr;

    size_t total_slots = (size_t)threads * (size_t)ops_per_thread;

    HIP_CHECK(hipMalloc(&d_counts, total_slots * sizeof(uint32_t)));
    HIP_CHECK(hipMalloc(&d_last_seq, (size_t)threads * sizeof(uint32_t)));
    HIP_CHECK(hipMalloc(&d_errors_oob, sizeof(unsigned int)));
    HIP_CHECK(hipMalloc(&d_errors_dup, sizeof(unsigned int)));
    HIP_CHECK(hipMalloc(&d_errors_order, sizeof(unsigned int)));
    HIP_CHECK(hipMalloc(&d_zeros, sizeof(unsigned int)));
    HIP_CHECK(hipMalloc(&d_gt1, sizeof(unsigned int)));
    HIP_CHECK(hipMalloc(&d_consumed, sizeof(unsigned long long)));

    HIP_CHECK(hipMemset(d_counts, 0, total_slots * sizeof(uint32_t)));
    HIP_CHECK(hipMemset(d_last_seq, 0xFF, (size_t)threads * sizeof(uint32_t)));
    HIP_CHECK(hipMemset(d_errors_oob, 0, sizeof(unsigned int)));
    HIP_CHECK(hipMemset(d_errors_dup, 0, sizeof(unsigned int)));
    HIP_CHECK(hipMemset(d_errors_order, 0, sizeof(unsigned int)));
    HIP_CHECK(hipMemset(d_zeros, 0, sizeof(unsigned int)));
    HIP_CHECK(hipMemset(d_gt1, 0, sizeof(unsigned int)));
    HIP_CHECK(hipMemset(d_consumed, 0, sizeof(unsigned long long)));

    int block = FIFO_BLOCK_SIZE;
    int grid = (threads + block - 1) / block;

    fifo_produce_kernel<<<grid, block>>>(qi.d_q, qi.d_h, threads, ops_per_thread);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    fifo_drain_kernel<<<grid, block>>>(qi.d_q,
                                       qi.d_h,
                                       threads,
                                       ops_per_thread,
                                       total_to_consume,
                                       d_counts,
                                       d_last_seq,
                                       d_errors_oob,
                                       d_errors_dup,
                                       d_errors_order,
                                       d_consumed);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    int verify_grid = (int)std::min<size_t>((total_slots + (size_t)block - 1) / (size_t)block, (size_t)65535);
    verify_grid = std::max(1, verify_grid);
    fifo_verify_counts_kernel<<<verify_grid, block>>>(d_counts, total_slots, d_zeros, d_gt1);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(&res.out_count, d_consumed, sizeof(unsigned long long), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&res.err_oob, d_errors_oob, sizeof(unsigned int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&res.err_dup, d_errors_dup, sizeof(unsigned int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&res.err_order, d_errors_order, sizeof(unsigned int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&res.err_missing, d_zeros, sizeof(unsigned int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&res.err_gt1, d_gt1, sizeof(unsigned int), hipMemcpyDeviceToHost));

    res.verified = (res.out_count == res.in_count)
                && (res.err_oob == 0u)
                && (res.err_dup == 0u)
                && (res.err_order == 0u)
                && (res.err_missing == 0u)
                && (res.err_gt1 == 0u);

    HIP_CHECK(hipFree(d_counts));
    HIP_CHECK(hipFree(d_last_seq));
    HIP_CHECK(hipFree(d_errors_oob));
    HIP_CHECK(hipFree(d_errors_dup));
    HIP_CHECK(hipFree(d_errors_order));
    HIP_CHECK(hipFree(d_zeros));
    HIP_CHECK(hipFree(d_gt1));
    HIP_CHECK(hipFree(d_consumed));

    destroy_queue(qi);
    return res;
}

static void print_fifo_result_3lines(const FifoCheckResult& r) {
    std::cout << "FIFO in: " << r.in_count << "\n";
    std::cout << "FIFO out: " << r.out_count << "\n";
    std::cout << "FIFO verified: " << (r.verified ? "YES" : "NO");
    if (!r.verified) {
        std::cout << " (oob=" << r.err_oob
                  << ", dup=" << r.err_dup
                  << ", order=" << r.err_order
                  << ", missing=" << r.err_missing
                  << ", gt1=" << r.err_gt1 << ")";
    }
    std::cout << "\n";
}

#endif



// -----------------------------------------------------------------------------
// Stats
// -----------------------------------------------------------------------------

__device__ unsigned long long g_enq_success = 0;
__device__ unsigned long long g_enq_fail    = 0;
__device__ unsigned long long g_deq_success = 0;
__device__ unsigned long long g_deq_empty   = 0;

struct Stats {
    unsigned long long enq_success = 0;
    unsigned long long enq_fail    = 0;
    unsigned long long deq_success = 0;
    unsigned long long deq_empty   = 0;
};

static inline void reset_stats() {
    unsigned long long zero = 0;
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_enq_success), &zero, sizeof(zero)));
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_enq_fail),    &zero, sizeof(zero)));
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_deq_success), &zero, sizeof(zero)));
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_deq_empty),   &zero, sizeof(zero)));
}

static inline Stats fetch_stats() {
    Stats s;
    HIP_CHECK(hipMemcpyFromSymbol(&s.enq_success, HIP_SYMBOL(g_enq_success), sizeof(s.enq_success)));
    HIP_CHECK(hipMemcpyFromSymbol(&s.enq_fail,    HIP_SYMBOL(g_enq_fail),    sizeof(s.enq_fail)));
    HIP_CHECK(hipMemcpyFromSymbol(&s.deq_success, HIP_SYMBOL(g_deq_success), sizeof(s.deq_success)));
    HIP_CHECK(hipMemcpyFromSymbol(&s.deq_empty,   HIP_SYMBOL(g_deq_empty),   sizeof(s.deq_empty)));
    return s;
}



// -----------------------------------------------------------------------------
// Kernels
// -----------------------------------------------------------------------------

__global__ void balanced_chunk_kernel(queue_t* q, handle_t* h, int num_threads, int chunk_ops) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_threads) return;

    handle_t* my_h = h ? &h[tid] : nullptr;

    unsigned long long local_enq_ok    = 0;
    unsigned long long local_enq_fail  = 0;
    unsigned long long local_deq_ok    = 0;
    unsigned long long local_deq_empty = 0;

    for (int i = 0; i < chunk_ops; i++) {
        uint64_t val = (((uint64_t)tid) << 32)
                     ^ (uint64_t)(i + 1)
                     ^ 0x9e3779b97f4a7c15ull;

        bool ok_enq = test_enqueue(q, my_h, val);
        if (ok_enq) local_enq_ok++;
        else        local_enq_fail++;

        uint64_t out = 0;
        bool ok_deq = test_dequeue(q, my_h, &out);
        if (ok_deq) local_deq_ok++;
        else        local_deq_empty++;
    }

    atomicAdd(&g_enq_success, local_enq_ok);
    atomicAdd(&g_enq_fail,    local_enq_fail);
    atomicAdd(&g_deq_success, local_deq_ok);
    atomicAdd(&g_deq_empty,   local_deq_empty);
}

__global__ void split_chunk_kernel(queue_t* q, handle_t* h, int num_threads, int chunk_ops, int producer_percent) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_threads) return;

    handle_t* my_h = h ? &h[tid] : nullptr;
    bool is_producer = ((tid * 100) / num_threads) < producer_percent;

    unsigned long long local_enq_ok    = 0;
    unsigned long long local_enq_fail  = 0;
    unsigned long long local_deq_ok    = 0;
    unsigned long long local_deq_empty = 0;

    for (int i = 0; i < chunk_ops; i++) {
        if (is_producer) {
            uint64_t val = (((uint64_t)tid) << 32)
                         ^ (uint64_t)(i + 1)
                         ^ 0x517cc1b727220a95ull;
#if defined(USE_SFQ)
            bool ok_enq = test_enqueue_split(q, my_h, val);
#else
            bool ok_enq = test_enqueue(q, my_h, val);
#endif
            if (ok_enq) local_enq_ok++;
            else        local_enq_fail++;
        } else {
            uint64_t out = 0;
#if defined(USE_SFQ)
            bool ok_deq = test_dequeue_split(q, my_h, &out);
#else
            bool ok_deq = test_dequeue(q, my_h, &out);
#endif
            if (ok_deq) local_deq_ok++;
            else        local_deq_empty++;
        }
    }

    atomicAdd(&g_enq_success, local_enq_ok);
    atomicAdd(&g_enq_fail,    local_enq_fail);
    atomicAdd(&g_deq_success, local_deq_ok);
    atomicAdd(&g_deq_empty,   local_deq_empty);
}

// -----------------------------------------------------------------------------
// Reporting
// -----------------------------------------------------------------------------

static void print_result_line(const std::string& label, int threads, double ms, const Stats& s) {
    const double sec = ms / 1000.0;
    const double enq_mops   = (double)s.enq_success / sec / 1e6;
    const double deq_mops   = (double)s.deq_success / sec / 1e6;
    const double empty_mops = (double)s.deq_empty   / sec / 1e6;
    const double fail_mops  = (double)s.enq_fail    / sec / 1e6;
    const double succ_mops  = (double)(s.enq_success + s.deq_success) / sec / 1e6;

    std::cout << std::setw(12) << label
              << " | threads=" << std::setw(6) << threads
              << " | time=" << std::fixed << std::setprecision(1) << std::setw(7) << ms << " ms"
              << " | succ="  << std::setprecision(2) << std::setw(9) << succ_mops << " Mops/s"
              << " | enq="   << std::setw(9) << enq_mops
              << " | deq="   << std::setw(9) << deq_mops
              << " | empty=" << std::setw(9) << empty_mops
              << " | fail="  << std::setw(9) << fail_mops
              << "\n";
}

static std::string csv_escape(const std::string& in) {
    std::string out;
    out.reserve(in.size() + 8);
    out.push_back('"');
    for (char c : in) {
        if (c == '"') out.push_back('"');
        out.push_back(c);
    }
    out.push_back('"');
    return out;
}

static void append_result_csv(std::ofstream& csv,
                              const std::string& gpu_name,
                              const std::string& mode,
                              int producer_percent,
                              int threads,
                              double ms,
                              const Stats& s) {
    const double sec = ms / 1000.0;
    const double enq_mops   = (double)s.enq_success / sec / 1e6;
    const double deq_mops   = (double)s.deq_success / sec / 1e6;
    const double empty_mops = (double)s.deq_empty   / sec / 1e6;
    const double fail_mops  = (double)s.enq_fail    / sec / 1e6;
    const double succ_mops  = (double)(s.enq_success + s.deq_success) / sec / 1e6;

    csv << csv_escape(gpu_name)
        << "," << csv_escape(QUEUE_NAME)
        << "," << csv_escape(mode)
        << "," << producer_percent
        << "," << threads
        << "," << RUN_MS
        << "," << WARMUP_MS
        << "," << CHUNK_OPS
        << "," << BLOCK_SIZE
        << "," << std::fixed << std::setprecision(3) << ms
        << "," << s.enq_success
        << "," << s.enq_fail
        << "," << s.deq_success
        << "," << s.deq_empty
        << "," << std::setprecision(6) << succ_mops
        << "," << enq_mops
        << "," << deq_mops
        << "," << empty_mops
        << "," << fail_mops
        << "\n";
}

// -----------------------------------------------------------------------------
// Timed runners
// -----------------------------------------------------------------------------

static double run_balanced_timed(QueueInstance& qi, int threads, int block_size, int chunk_ops,
                                 int warmup_ms, int run_ms) {
    const int grid = (threads + block_size - 1) / block_size;

    {
        auto t0 = std::chrono::high_resolution_clock::now();
        while (true) {
            balanced_chunk_kernel<<<grid, block_size>>>(qi.d_q, qi.d_h, threads, chunk_ops);
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (ms >= warmup_ms) break;
        }
    }

    reset_stats();

    hipEvent_t ev_start, ev_stop;
    HIP_CHECK(hipEventCreate(&ev_start));
    HIP_CHECK(hipEventCreate(&ev_stop));

    auto t0 = std::chrono::high_resolution_clock::now();
    HIP_CHECK(hipEventRecord(ev_start, 0));

    while (true) {
        balanced_chunk_kernel<<<grid, block_size>>>(qi.d_q, qi.d_h, threads, chunk_ops);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ms >= run_ms) break;
    }

    HIP_CHECK(hipEventRecord(ev_stop, 0));
    HIP_CHECK(hipEventSynchronize(ev_stop));

    float gpu_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&gpu_ms, ev_start, ev_stop));
    HIP_CHECK(hipEventDestroy(ev_start));
    HIP_CHECK(hipEventDestroy(ev_stop));

    return (double)gpu_ms;
}

static double run_split_timed(QueueInstance& qi, int threads, int block_size, int chunk_ops,
                              int producer_percent, int warmup_ms, int run_ms) {
    const int grid = (threads + block_size - 1) / block_size;

    {
        auto t0 = std::chrono::high_resolution_clock::now();
        while (true) {
            split_chunk_kernel<<<grid, block_size>>>(qi.d_q, qi.d_h, threads, chunk_ops, producer_percent);
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (ms >= warmup_ms) break;
        }
    }

    reset_stats();

    hipEvent_t ev_start, ev_stop;
    HIP_CHECK(hipEventCreate(&ev_start));
    HIP_CHECK(hipEventCreate(&ev_stop));

    auto t0 = std::chrono::high_resolution_clock::now();
    HIP_CHECK(hipEventRecord(ev_start, 0));

    while (true) {
        split_chunk_kernel<<<grid, block_size>>>(qi.d_q, qi.d_h, threads, chunk_ops, producer_percent);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ms >= run_ms) break;
    }

    HIP_CHECK(hipEventRecord(ev_stop, 0));
    HIP_CHECK(hipEventSynchronize(ev_stop));

    float gpu_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&gpu_ms, ev_start, ev_stop));
    HIP_CHECK(hipEventDestroy(ev_start));
    HIP_CHECK(hipEventDestroy(ev_stop));

    return (double)gpu_ms;
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main() {
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    const char* gpu_env = std::getenv("GPU_NAME");
    const std::string gpu_name = (gpu_env && gpu_env[0] != '\0') ? std::string(gpu_env)
                                                                   : std::string(prop.name);

    const std::string csv_path = CSV_FILE;
    std::ifstream csv_check(csv_path);
    const bool csv_exists = csv_check.good();
    csv_check.close();

    std::ofstream csv(csv_path, std::ios::app);
    if (!csv) {
        std::cerr << "Failed to open CSV file: " << csv_path << "\n";
        return 1;
    }
    if (!csv_exists) {
        csv << "gpu_name,queue,mode,producer_percent,threads,run_ms,warmup_ms,chunk_ops,block_size,elapsed_ms,enq_success,enq_fail,deq_success,deq_empty,succ_mops,enq_mops,deq_mops,empty_mops,fail_mops\n";
    }

    std::cout << "GPU: " << gpu_name << "\n";
    std::cout << "Queue: " << QUEUE_NAME << "\n";
    std::cout << "Method: fixed-duration timed throughput harness\n";
    std::cout << "RUN_MS=" << RUN_MS
              << " WARMUP_MS=" << WARMUP_MS
              << " CHUNK_OPS=" << CHUNK_OPS
              << " BLOCK_SIZE=" << BLOCK_SIZE
              << " MODE_FIFO=" << MODE_FIFO << "\n";
    print_backend_info();

    std::vector<int> thread_counts = {512, 1024, 2048, 4096, 8192, 16384, 32768};
    std::vector<int> producer_percents = {25, 50, 75};

    if (!ONLY_SPLIT) {
        std::cout << "=== Balanced Pairwise Timed Benchmark ===\n";
        std::cout << "Each active thread repeatedly performs enqueue then dequeue.\n\n";

        for (int threads : thread_counts) {
            QueueInstance qi = create_queue(threads);
            HIP_CHECK(hipDeviceSynchronize());

#if ENABLE_ROCTX
            roctx_push("balanced-" + std::to_string(threads) + "t");
#endif
            double ms = run_balanced_timed(qi, threads, BLOCK_SIZE, CHUNK_OPS, WARMUP_MS, RUN_MS);
#if ENABLE_ROCTX
            roctx_pop();
#endif
            HIP_CHECK(hipDeviceSynchronize());

            Stats s = fetch_stats();
            print_result_line("balanced", threads, ms, s);
#if MODE_FIFO
            print_fifo_result_3lines(run_fifo_check_once(threads));
#endif
            append_result_csv(csv, gpu_name, "balanced", -1, threads, ms, s);
            destroy_queue(qi);
        }

        std::cout << "\n";
    }

    if (!ONLY_BALANCED) {
        std::cout << "=== Split Producer/Consumer Timed Benchmark ===\n";
        std::cout << "Producer percentage sweeps offered load by role population.\n";
        std::cout << "25% producers matches the classic 1P/3C imbalance pattern.\n\n";

        for (int p : producer_percents) {
            std::cout << "--- producer_percent=" << p
                      << " (" << p << "% producers / " << (100 - p) << "% consumers) ---\n";

            for (int threads : thread_counts) {
                QueueInstance qi = create_queue(threads);
                HIP_CHECK(hipDeviceSynchronize());

#if ENABLE_ROCTX
                roctx_push("split-" + std::to_string(p) + "p-" + std::to_string(threads) + "t");
#endif
                double ms = run_split_timed(qi, threads, BLOCK_SIZE, CHUNK_OPS, p, WARMUP_MS, RUN_MS);
#if ENABLE_ROCTX
                roctx_pop();
#endif
                HIP_CHECK(hipDeviceSynchronize());

                Stats s = fetch_stats();
                std::string label = "split-" + std::to_string(p);
                print_result_line(label, threads, ms, s);
#if MODE_FIFO
                print_fifo_result_3lines(run_fifo_check_once(threads));
#endif
                append_result_csv(csv, gpu_name, "split", p, threads, ms, s);
                destroy_queue(qi);
            }

            std::cout << "\n";
        }
    }

    csv.flush();

    std::cout << "Done.\n";
    return 0;
}