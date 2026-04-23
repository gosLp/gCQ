#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>

#ifndef WF_LOG
#define WF_LOG 1
#endif
#include "wf_linlog.hpp"

#define HIP_CHECK(call) do {                                                    \
    hipError_t _e = (call);                                                     \
    if (_e != hipSuccess) {                                                     \
        fprintf(stderr, "HIP error %s:%d : %s\n", __FILE__, __LINE__,         \
                hipGetErrorString(_e));                                         \
        std::exit(1);                                                           \
    }                                                                           \
} while (0)

#ifndef HIST_THREADS
#define HIST_THREADS 64
#endif

#ifndef HIST_OPS
#define HIST_OPS 16
#endif

#ifndef HIST_BLOCK_SIZE
#define HIST_BLOCK_SIZE 256
#endif

#ifndef HIST_MODE
#define HIST_MODE 0   // 0=alternating, 1=split
#endif

#ifndef HIST_PRODUCER_PERCENT
#define HIST_PRODUCER_PERCENT 50
#endif

#if defined(USE_GWFQ)
#include "../gwf_ring.hpp"
#ifndef GWF_N
#define GWF_N 65536
#endif
using queue_t  = wf_mpmc_t<gwf_64k, GWF_N>;
using handle_t = int;
static constexpr const char* QUEUE_NAME = "gwfq";
static constexpr bool NEEDS_HANDLES = false;
__global__ void queue_init_kernel(queue_t* q) {
    if (blockIdx.x == 0 && threadIdx.x == 0) wf_mpmc_init<gwf_64k, GWF_N>(q);
}
static __device__ __forceinline__ bool hist_enqueue(queue_t* q, handle_t*, uint64_t v) {
    return Enqueue_Ptr<gwf_64k, GWF_N>(q, v);
}
static __device__ __forceinline__ bool hist_dequeue(queue_t* q, handle_t*, uint64_t* out) {
    return Dequeue_Ptr<gwf_64k, GWF_N>(q, out);
}

#elif defined(USE_GLFQ)
#include "../glf_queue.hpp"
using queue_t  = fq_mpmc_t;
using handle_t = int;
static constexpr const char* QUEUE_NAME = "glfq";
static constexpr bool NEEDS_HANDLES = false;
__global__ void queue_init_kernel(queue_t* q) {
    if (blockIdx.x == 0 && threadIdx.x == 0) fq_init(q);
}
static __device__ __forceinline__ bool hist_enqueue(queue_t* q, handle_t*, uint64_t v) {
    return fq_enqueue_ptr(q, v);
}
static __device__ __forceinline__ bool hist_dequeue(queue_t* q, handle_t*, uint64_t* out) {
    return fq_dequeue_ptr(q, out);
}

#elif defined(USE_WFQ)
#include "../wfqueue_hip_opt.hpp"
using queue_t  = wf_queue;
using handle_t = wf_handle;
static constexpr const char* QUEUE_NAME = "wfq";
static constexpr bool NEEDS_HANDLES = true;
static inline void backend_create_queue(queue_t** d_q, handle_t** d_h, int threads) {
    wf_queue_host_init(d_q, d_h, threads);
}
static inline void backend_destroy_queue(queue_t* d_q, handle_t* d_h) {
    if (d_q) {
        wf_queue hq{};
        HIP_CHECK(hipMemcpy(&hq, d_q, sizeof(queue_t), hipMemcpyDeviceToHost));
        if (hq.pool.segments) HIP_CHECK(hipFree(hq.pool.segments));
        HIP_CHECK(hipFree(d_q));
    }
    if (d_h) HIP_CHECK(hipFree(d_h));
}
static __device__ __forceinline__ bool hist_enqueue(queue_t* q, handle_t* h, uint64_t v) {
    wf_enqueue(q, h, v);
    return true;
}
static __device__ __forceinline__ bool hist_dequeue(queue_t* q, handle_t* h, uint64_t* out) {
    uint64_t v = wf_dequeue(q, h);
    if (v == WF_EMPTY) return false;
    *out = v;
    return true;
}

#elif defined(USE_SFQ)
#include "../sfqueue_hip.hpp"
#include "../sfqueue_hip.cpp"
using queue_t  = sfq_queue;
using handle_t = sfq_handle;
static constexpr const char* QUEUE_NAME = "sfq";
static constexpr bool NEEDS_HANDLES = true;
static inline void backend_create_queue(queue_t** d_q, handle_t** d_h, int threads) {
    sfq_queue_host_init(d_q, d_h, threads);
}
static inline void backend_destroy_queue(queue_t* d_q, handle_t* d_h) {
    sfq_queue_destroy(d_q, d_h);
}
#ifndef HIST_SFQ_TID_BITS
#define HIST_SFQ_TID_BITS 15
#endif
#ifndef HIST_SFQ_SEQ_BITS
#define HIST_SFQ_SEQ_BITS 16
#endif
static_assert((HIST_SFQ_TID_BITS + HIST_SFQ_SEQ_BITS) <= 31,
              "HIST_SFQ_TID_BITS + HIST_SFQ_SEQ_BITS must be <= 31");
static constexpr uint32_t HIST_SFQ_TID_MASK = (1u << HIST_SFQ_TID_BITS) - 1u;
static constexpr uint32_t HIST_SFQ_SEQ_MASK = (1u << HIST_SFQ_SEQ_BITS) - 1u;
static __device__ __forceinline__ bool hist_enqueue(queue_t* q, handle_t*, uint64_t v) {
    return sfq_enqueue_nb_u32(q, (uint32_t)v) == SFQ_SUCCESS;
}
static __device__ __forceinline__ bool hist_dequeue(queue_t* q, handle_t*, uint64_t* out) {
    uint32_t item = 0;
    int rc = sfq_dequeue_nb_u32(q, &item);
    if (rc != SFQ_SUCCESS) return false;
    *out = (uint64_t)item;
    return true;
}

#else
#error "Define one of USE_GWFQ, USE_GLFQ, USE_WFQ, USE_SFQ"
#endif

struct QueueInstance {
    queue_t*  d_q = nullptr;
    handle_t* d_h = nullptr;
};

static QueueInstance create_queue(int threads) {
    QueueInstance qi{};
#if defined(USE_WFQ) || defined(USE_SFQ)
    backend_create_queue(&qi.d_q, &qi.d_h, threads);
#else
    HIP_CHECK(hipMalloc(&qi.d_q, sizeof(queue_t)));
    if (NEEDS_HANDLES) {
        HIP_CHECK(hipMalloc(&qi.d_h, (size_t)threads * sizeof(handle_t)));
        HIP_CHECK(hipMemset(qi.d_h, 0, (size_t)threads * sizeof(handle_t)));
    }
    queue_init_kernel<<<1,1>>>(qi.d_q);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
#endif
    return qi;
}

static void destroy_queue(QueueInstance& qi) {
#if defined(USE_WFQ) || defined(USE_SFQ)
    backend_destroy_queue(qi.d_q, qi.d_h);
#else
    if (qi.d_q) HIP_CHECK(hipFree(qi.d_q));
    if (qi.d_h) HIP_CHECK(hipFree(qi.d_h));
#endif
    qi.d_q = nullptr;
    qi.d_h = nullptr;
}

static __device__ __forceinline__ uint64_t make_hist_value(uint32_t tid, uint32_t seq) {
#if defined(USE_SFQ)
    return (uint64_t)(((tid & HIST_SFQ_TID_MASK) << HIST_SFQ_SEQ_BITS) |
                      ((seq + 1u) & HIST_SFQ_SEQ_MASK));
#else
    return ((uint64_t)tid << 32) | (uint64_t)(seq + 1u);
#endif
}

__global__ void alternating_history_kernel(queue_t* q, handle_t* h, int threads, int ops) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= threads) return;
    handle_t* my_h = h ? &h[tid] : nullptr;
    uint32_t log_idx = 0;

    for (int i = 0; i < ops; ++i) {
        uint64_t v = make_hist_value((uint32_t)tid, (uint32_t)i);
        WF_LOG_ENQ_BEGIN(tid, v, log_idx);
        bool ok_enq = hist_enqueue(q, my_h, v);
        WF_LOG_ENQ_END(ok_enq);

        WF_LOG_DEQ_BEGIN(tid, log_idx);
        uint64_t out = 0;
        bool ok_deq = hist_dequeue(q, my_h, &out);
        WF_LOG_DEQ_END(ok_deq ? out : 0ull);
    }
}

__global__ void split_history_kernel(queue_t* q, handle_t* h, int threads, int ops, int producer_percent) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= threads) return;
    handle_t* my_h = h ? &h[tid] : nullptr;
    uint32_t log_idx = 0;
    bool is_producer = ((tid * 100) / threads) < producer_percent;

    for (int i = 0; i < ops; ++i) {
        if (is_producer) {
            uint64_t v = make_hist_value((uint32_t)tid, (uint32_t)i);
            WF_LOG_ENQ_BEGIN(tid, v, log_idx);
            bool ok_enq = hist_enqueue(q, my_h, v);
            WF_LOG_ENQ_END(ok_enq);
        } else {
            WF_LOG_DEQ_BEGIN(tid, log_idx);
            uint64_t out = 0;
            bool ok_deq = hist_dequeue(q, my_h, &out);
            WF_LOG_DEQ_END(ok_deq ? out : 0ull);
        }
    }
}

static WFLogEntry* g_host_log = nullptr;
static size_t      g_host_entries = 0;

static void wf_log_alloc(int threads, int ops, int mode) {
    uint32_t stride = (mode == 0 ? 2u : 1u) * (uint32_t)ops + 8u;
    g_host_entries = (size_t)threads * (size_t)stride;
    HIP_CHECK(hipMallocManaged(&g_host_log, g_host_entries * sizeof(WFLogEntry)));
    HIP_CHECK(hipMemset(g_host_log, 0, g_host_entries * sizeof(WFLogEntry)));
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_wf_log), &g_host_log, sizeof(g_host_log)));
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_wf_stride), &stride, sizeof(stride)));
    unsigned long long one = 1ull;
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_wf_time), &one, sizeof(one)));
}

static void wf_log_free() {
    if (g_host_log) HIP_CHECK(hipFree(g_host_log));
    g_host_log = nullptr;
    g_host_entries = 0;
}

static int wf_log_dump_jsonl(const char* path) {
    std::filesystem::path out_path(path);
    if (out_path.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(out_path.parent_path(), ec);
        if (ec) return 1;
    }

    std::vector<WFLogEntry> used;
    used.reserve(g_host_entries);
    for (size_t i = 0; i < g_host_entries; ++i) {
        if (g_host_log[i].t1 != 0ull) used.push_back(g_host_log[i]);
    }
    std::sort(used.begin(), used.end(), [](const WFLogEntry& a, const WFLogEntry& b) {
        if (a.t0 != b.t0) return a.t0 < b.t0;
        if (a.t1 != b.t1) return a.t1 < b.t1;
        if (a.tid != b.tid) return a.tid < b.tid;
        return a.op < b.op;
    });

    FILE* f = fopen(path, "wb");
    if (!f) return 1;
    for (const auto& e : used) {
        fprintf(f,
                "{\"proc\":%llu,\"op\":%u,\"arg\":%llu,\"ret\":%llu,\"call\":%llu,\"end\":%llu}\n",
                (unsigned long long)e.tid,
                e.op,
                (unsigned long long)e.arg,
                (unsigned long long)e.ret,
                (unsigned long long)e.t0,
                (unsigned long long)e.t1);
    }
    fclose(f);
    return 0;
}

static int cap_ops_for_backend(int threads, int ops) {
#if defined(USE_SFQ)
    const int max_threads_enc = (1 << HIST_SFQ_TID_BITS);
    if (threads > max_threads_enc) return 0;
    if (ops > (int)HIST_SFQ_SEQ_MASK) ops = (int)HIST_SFQ_SEQ_MASK;
#endif
    return ops;
}

int main(int argc, char** argv) {
    const char* out_path = (argc > 1) ? argv[1] : "correctness/histories/history.jsonl";
    const int threads = (argc > 2) ? atoi(argv[2]) : HIST_THREADS;
    int ops = (argc > 3) ? atoi(argv[3]) : HIST_OPS;
    const int mode = (argc > 4) ? atoi(argv[4]) : HIST_MODE;
    const int producer_percent = (argc > 5) ? atoi(argv[5]) : HIST_PRODUCER_PERCENT;

    if (threads <= 0 || ops <= 0) return 2;
    ops = cap_ops_for_backend(threads, ops);
    if (ops <= 0) return 3;

    QueueInstance qi = create_queue(threads);
    wf_log_alloc(threads, ops, mode);

    const int block = HIST_BLOCK_SIZE;
    const int grid = (threads + block - 1) / block;

    if (mode == 0) {
        alternating_history_kernel<<<grid, block>>>(qi.d_q, qi.d_h, threads, ops);
    } else {
        split_history_kernel<<<grid, block>>>(qi.d_q, qi.d_h, threads, ops, producer_percent);
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    int rc = wf_log_dump_jsonl(out_path);
    wf_log_free();
    destroy_queue(qi);
    return rc;
}
