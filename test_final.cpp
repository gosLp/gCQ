
// test_final.cpp
// Timed throughput harness for GPU queues.
// Goal: keep queue API usage intact while moving from fixed-total-work
// to fixed-duration measurement with balanced + producer/consumer tests.
//
// Example:
//   hipcc -O3 -DUSE_WFQ test_final.cpp -o test_wfq
//   hipcc -O3 -DUSE_BWFQ test_final.cpp -o test_bwfq
//
// Optional knobs:
//   -DRUN_MS=2000
//   -DWARMUP_MS=250
//   -DCHUNK_OPS=64
//   -DBLOCK_SIZE=256
//   -DONLY_BALANCED=1
//   -DONLY_SPLIT=1

#if defined(USE_WFQ)
    #include "wfqueue_hip_opt.hpp"
    #define QUEUE_NAME "WFQ"
    #define QUEUE_EMPTY WF_EMPTY
    #define HAS_SHARED_BUF 0
#elif defined(USE_BWFQ)
    #include "wfqueue_bounded.hpp"
    #define QUEUE_NAME "Bounded-WFQ"
    #define HAS_SHARED_BUF 0
    #define QUEUE_EMPTY WF_EMPTY
    #define NEEDS_RECORDS 1
    #define HAS_DESTROY 1
    // #define ENQ_RET_STATUS 1
    // #define DEQ_RET_STATUS 1
    #define ENQ_RET_STATUS 0
    #define DEQ_RET_STATUS 0
#elif defined(USE_BWFQ2)
    #include "wfqueue_bwfq.hpp"
    #define QUEUE_NAME "Bounded-WFQ2"
    #define HAS_SHARED_BUF 0
    #define QUEUE_EMPTY WF_EMPTY
    #define NEEDS_RECORDS 1
    #define HAS_DESTROY 1
    #define ENQ_RET_STATUS 0
    #define DEQ_RET_STATUS 0
#elif defined(USE_CAWFQ)
    #include "wfqueue_cawfq.hpp"
    #include "wfqueue_cawfq.cpp"
    #define QUEUE_NAME "CAWFQ"
    #define QUEUE_EMPTY WF_EMPTY
    #define HAS_SHARED_BUF 0
    #define HAS_DESTROY 1
#elif defined(USE_CAWFQ2)
    #include "wfqueue_cawfq2.hpp"
    #define QUEUE_NAME "CAWFQ2-WarpBatch"
    #define QUEUE_EMPTY WF_EMPTY
    #define BLFQ_STATUS_API 1
    #define HAS_SHARED_BUF 0
    #define HAS_DESTROY 1
    #define ENQ_RET_STATUS 1
    #define DEQ_RET_STATUS 1
#elif defined(USE_DET)
    #include "wfqueue_det.hpp"
    #define QUEUE_NAME "WFQ-DET"
    #define QUEUE_EMPTY WF_EMPTY
    #define HAS_SHARED_BUF 0
    #define FLUSH 1
    #define queue_flush wf_flush
    #define HAS_DESTROY 1
#elif defined(USE_SFQ)
    #include "sfqueue_hip.hpp"
    #include "sfqueue_hip.cpp"
    #define QUEUE_NAME "SFQ"
    #define HAS_SHARED_BUF 0

    #define wf_enqueue sfq_enqueue
    #define wf_dequeue sfq_dequeue
    #define wf_queue   sfq_queue
    #define wf_handle  sfq_handle
    #define wf_queue_host_init sfq_queue_host_init
    #define QUEUE_EMPTY SFQ_EMPTY

    #define HAS_DESTROY 1
    #define wf_queue_destroy sfq_queue_destroy
#elif defined(USE_WFQ64)
    #include "bwfq_wf.hpp"
    // #include "new_queue.hpp"

    // Adapt this queue to the harness naming convention
    #define QUEUE_NAME "GWFQ-CAS1-Proto"
    #define QUEUE_EMPTY GWFQ_EMPTY

    #define HAS_SHARED_BUF 0
    #define HAS_DESTROY 1
    #define NEEDS_RECORDS 1
    #define ENQ_RET_STATUS 0
    #define DEQ_RET_STATUS 0

    // Type aliases expected by the harness
    #define wf_queue         wfq_queue
    #define wf_handle        wfq_handle
    #define wf_thread_record wfq_record

    // Function aliases expected by the harness
    #define wf_enqueue          wfq_enqueue
    #define wf_dequeue          wfq_dequeue
    #define wf_queue_host_init  wfq_queue_host_init
    #define wf_queue_destroy    wfq_queue_destroy
#else
    #error "Define one of: USE_WFQ, USE_BWFQ, USE_BWFQ2, USE_CAWFQ, USE_CAWFQ2, USE_DET, USE_SFQ, USE_WFQ64"
#endif

#ifndef BLFQ_STATUS_API
#define BLFQ_STATUS_API 0
#endif

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <cstdint>
#include <cstdlib>

#ifndef HAS_SHARED_BUF
#define HAS_SHARED_BUF 0
#endif

#ifndef HAS_DESTROY
#define HAS_DESTROY 0
#endif

#ifndef NEEDS_RECORDS
#define NEEDS_RECORDS 0
#endif

#ifndef ENQ_RET_STATUS
#define ENQ_RET_STATUS 0
#endif

#ifndef DEQ_RET_STATUS
#define DEQ_RET_STATUS 0
#endif

#ifndef RUN_MS
#define RUN_MS 2000
#endif

#ifndef WARMUP_MS
#define WARMUP_MS 250
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

#ifndef ENABLE_DRAIN
#define ENABLE_DRAIN 0
#endif

#define HIP_CHECK(call) do {                                      \
    hipError_t _e = (call);                                       \
    if (_e != hipSuccess) {                                       \
        std::cerr << "HIP error " << __FILE__ << ":" << __LINE__  \
                  << " : " << hipGetErrorString(_e) << "\n";      \
        std::exit(1);                                             \
    }                                                             \
} while (0)

static constexpr uint64_t VALUE_BASE = 1ull;

// ============================================================
// Stats
// ============================================================

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

// ============================================================
// Unified wrappers
// ============================================================

static __device__ __forceinline__
bool test_enqueue(wf_queue* q, wf_handle* h, uint64_t v) {
#if ENQ_RET_STATUS
    return wf_enqueue(q, h, v) == WF_OK;
#else
    wf_enqueue(q, h, v);
    return true;
#endif
}

// static __device__ __forceinline__
// bool test_dequeue(wf_queue* q, wf_handle* h, uint64_t* out) {
// #if DEQ_RET_STATUS
//     return wf_dequeue(q, h, out) == WF_OK;
// #else
//     uint64_t v = wf_dequeue(q, h);
//     *out = v;
//     return (v != QUEUE_EMPTY && v != 0);
// #endif
// }

static __device__ __forceinline__
bool test_dequeue(wf_queue* q, wf_handle* h, uint64_t* out) {
#if DEQ_RET_STATUS
    return wf_dequeue(q, h, out) == WF_OK;
#else
    uint32_t v = wf_dequeue(q, h);
    *out = v;
#if defined(USE_WFQ64)
    return wfq_is_real_index(v);
#else
    return (v != QUEUE_EMPTY && v != 0);
#endif
#endif
}

// ============================================================
// Kernels
// ============================================================

#if HAS_SHARED_BUF
__global__ void balanced_chunk_kernel(wf_queue* q, wf_handle* h, int num_threads, int chunk_ops) {
    extern __shared__ wf_buffer_entry shared_buf[];
    wf_buffer_entry* my_buf = &shared_buf[threadIdx.x * LOCAL_BUFFER_SIZE];
#else
__global__ void balanced_chunk_kernel(wf_queue* q, wf_handle* h, int num_threads, int chunk_ops) {
#endif
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_threads) return;

    wf_handle* my_h = &h[tid];

    unsigned long long local_enq_ok = 0;
    unsigned long long local_enq_fail = 0;
    unsigned long long local_deq_ok = 0;
    unsigned long long local_deq_empty = 0;

    for (int i = 0; i < chunk_ops; i++) {
        // uint64_t val = (((uint64_t)tid) << 32) ^ (uint64_t)(i + 1) ^ VALUE_BASE;
        #if defined(USE_WFQ64)
            // uint64_t val = (uint64_t)(((tid * chunk_ops + i) & 0x3ffffffcU) + 1U);
            uint32_t raw = (uint32_t)(tid * chunk_ops + i + 1);
            uint32_t val32 = 1u + (raw % (WFQ_NOTED_INDEX - 1u)); // 1 .. WFQ_NOTED_INDEX-1
            uint64_t val = (uint64_t)val32;
        #else
            uint64_t val = (((uint64_t)tid) << 32) ^ (uint64_t)(i + 1) ^ VALUE_BASE;
        #endif

#if HAS_SHARED_BUF
        wf_enqueue(q, my_h, val, my_buf);
        local_enq_ok++;
        uint64_t r = wf_dequeue(q, my_h, my_buf);
        if (r != QUEUE_EMPTY && r != 0) local_deq_ok++;
        else                            local_deq_empty++;
#else
        bool ok_enq = test_enqueue(q, my_h, val);
        if (ok_enq) local_enq_ok++;
        else        local_enq_fail++;

        uint64_t out = 0;
        bool ok_deq = test_dequeue(q, my_h, &out);
        if (ok_deq) local_deq_ok++;
        else        local_deq_empty++;
#endif
    }

    atomicAdd(&g_enq_success, local_enq_ok);
    atomicAdd(&g_enq_fail,    local_enq_fail);
    atomicAdd(&g_deq_success, local_deq_ok);
    atomicAdd(&g_deq_empty,   local_deq_empty);
}

#if HAS_SHARED_BUF
__global__ void split_chunk_kernel(wf_queue* q, wf_handle* h, int num_threads, int chunk_ops, int producer_percent) {
    extern __shared__ wf_buffer_entry shared_buf[];
    wf_buffer_entry* my_buf = &shared_buf[threadIdx.x * LOCAL_BUFFER_SIZE];
#else
__global__ void split_chunk_kernel(wf_queue* q, wf_handle* h, int num_threads, int chunk_ops, int producer_percent) {
#endif
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_threads) return;

    wf_handle* my_h = &h[tid];
    bool is_producer = ((tid * 100) / num_threads) < producer_percent;

    unsigned long long local_enq_ok = 0;
    unsigned long long local_enq_fail = 0;
    unsigned long long local_deq_ok = 0;
    unsigned long long local_deq_empty = 0;

    for (int i = 0; i < chunk_ops; i++) {
        if (is_producer) {
            // uint64_t val = (((uint64_t)tid) << 32) ^ (uint64_t)(i + 1) ^ 0x9e3779b97f4a7c15ull;
            #if defined(USE_WFQ64)
                uint64_t val = (uint64_t)(((tid * chunk_ops + i) & 0x3ffffffcU) + 1U);
            #else
                uint64_t val = (((uint64_t)tid) << 32) ^ (uint64_t)(i + 1) ^ 0x9e3779b97f4a7c15ull;
            #endif

#if HAS_SHARED_BUF
            wf_enqueue(q, my_h, val, my_buf);
            local_enq_ok++;
#else
            bool ok_enq = test_enqueue(q, my_h, val);
            if (ok_enq) local_enq_ok++;
            else        local_enq_fail++;
#endif
        } else {
#if HAS_SHARED_BUF
            uint64_t r = wf_dequeue(q, my_h, my_buf);
            if (r != QUEUE_EMPTY && r != 0) local_deq_ok++;
            else                            local_deq_empty++;
#else
            uint64_t out = 0;
            bool ok_deq = test_dequeue(q, my_h, &out);
            if (ok_deq) local_deq_ok++;
            else        local_deq_empty++;
#endif
        }
    }

    atomicAdd(&g_enq_success, local_enq_ok);
    atomicAdd(&g_enq_fail,    local_enq_fail);
    atomicAdd(&g_deq_success, local_deq_ok);
    atomicAdd(&g_deq_empty,   local_deq_empty);
}

// Optional drain kernel if you want a cleanup pass between experiments.
// Usually unnecessary since we re-init the queue each point.
__global__ void noop_kernel() {}

// ============================================================
// Queue init/destroy helpers
// ============================================================

struct QueueInstance {
    wf_queue*  d_q = nullptr;
    wf_handle* d_h = nullptr;
#if NEEDS_RECORDS
    wf_thread_record* d_r = nullptr;
#endif
};

static QueueInstance create_queue(int threads) {
    QueueInstance qi{};
#if NEEDS_RECORDS
    wf_queue_host_init(&qi.d_q, &qi.d_h, &qi.d_r, threads);
#else
    wf_queue_host_init(&qi.d_q, &qi.d_h, threads);
#endif
    return qi;
}

// static void destroy_queue(QueueInstance& qi) {
// #if HAS_DESTROY
//     wf_queue_destroy(qi.d_q, qi.d_h);
// #else
//     wf_queue hq{};
//     HIP_CHECK(hipMemcpy(&hq, qi.d_q, sizeof(wf_queue), hipMemcpyDeviceToHost));
//     if (hq.pool.segments) HIP_CHECK(hipFree(hq.pool.segments));
//     HIP_CHECK(hipFree(qi.d_q));
//     HIP_CHECK(hipFree(qi.d_h));
// #endif

// #if NEEDS_RECORDS
//     if (qi.d_r) HIP_CHECK(hipFree(qi.d_r));
// #endif
// }

static void destroy_queue(QueueInstance& qi) {
#if HAS_DESTROY
    wf_queue_destroy(qi.d_q, qi.d_h);
    qi.d_q = nullptr;
    qi.d_h = nullptr;
#if NEEDS_RECORDS
    qi.d_r = nullptr;
#endif
#else
    wf_queue hq{};
    HIP_CHECK(hipMemcpy(&hq, qi.d_q, sizeof(wf_queue), hipMemcpyDeviceToHost));
    if (hq.pool.segments) HIP_CHECK(hipFree(hq.pool.segments));
    HIP_CHECK(hipFree(qi.d_q));
    HIP_CHECK(hipFree(qi.d_h));
#endif

#if NEEDS_RECORDS
#if !HAS_DESTROY
    if (qi.d_r) HIP_CHECK(hipFree(qi.d_r));
#endif
#endif
}

// static void print_wfq64_debug(const QueueInstance& qi) {
// #if defined(USE_WFQ64)
//     wf_queue hq{};
//     HIP_CHECK(hipMemcpy(&hq, qi.d_q, sizeof(wf_queue), hipMemcpyDeviceToHost));

//     std::cout
//         << "      [dbg] enq_slow=" << hq.enq_slow
//         << " deq_slow=" << hq.deq_slow
//         << " help_given=" << hq.help_given
//         << " proof_fail=" << hq.proof_fail
//         << "\n"
//         << "      [dbg] slow_faa_entries=" << hq.dbg_slow_faa_entries
//         << " fin_exits=" << hq.dbg_slow_faa_fin_exits
//         << " local_cas_fail=" << hq.dbg_slow_faa_local_cas_fail
//         << " gp_cas_fail=" << hq.dbg_slow_faa_gp_cas_fail
//         << "\n"
//         << "      [dbg] phase2_seen=" << hq.dbg_phase2_help_seen
//         << " clear_ok=" << hq.dbg_phase2_clear_success
//         << " clear_fail=" << hq.dbg_phase2_clear_fail
//         << "\n"
//         << "      [dbg] try_enq_false=" << hq.dbg_try_enq_slow_false
//         << " try_deq_false=" << hq.dbg_try_deq_slow_false
//         << " noted=" << hq.dbg_noted_writes
//         << "\n"
//         << "      [dbg] finalize_scans=" << hq.dbg_finalize_scans
//         << " finalize_hits=" << hq.dbg_finalize_hits
//         << " owner_collect_empty=" << hq.dbg_owner_collect_empty
//         << "\n";
// #endif
// }

// ============================================================
// Reporting
// ============================================================

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
              << " | succ=" << std::setprecision(2) << std::setw(9) << succ_mops << " Mops/s"
              << " | enq="  << std::setw(9) << enq_mops
              << " | deq="  << std::setw(9) << deq_mops
              << " | empty="<< std::setw(9) << empty_mops
              << " | fail=" << std::setw(9) << fail_mops
              << "\n";
}

// ============================================================
// Timed runners
// ============================================================

static double run_balanced_timed(QueueInstance& qi, int threads, int block_size, int chunk_ops, int warmup_ms, int run_ms) {
    const int grid = (threads + block_size - 1) / block_size;

#if HAS_SHARED_BUF
    const size_t smem = (size_t)block_size * LOCAL_BUFFER_SIZE * sizeof(wf_buffer_entry);
#endif

    // Warmup
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        while (true) {
#if HAS_SHARED_BUF
            balanced_chunk_kernel<<<grid, block_size, smem>>>(qi.d_q, qi.d_h, threads, chunk_ops);
#else
            balanced_chunk_kernel<<<grid, block_size>>>(qi.d_q, qi.d_h, threads, chunk_ops);
#endif
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
#if HAS_SHARED_BUF
        balanced_chunk_kernel<<<grid, block_size, smem>>>(qi.d_q, qi.d_h, threads, chunk_ops);
#else
        balanced_chunk_kernel<<<grid, block_size>>>(qi.d_q, qi.d_h, threads, chunk_ops);
#endif
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

#if HAS_SHARED_BUF
    const size_t smem = (size_t)block_size * LOCAL_BUFFER_SIZE * sizeof(wf_buffer_entry);
#endif

    // Warmup
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        while (true) {
#if HAS_SHARED_BUF
            split_chunk_kernel<<<grid, block_size, smem>>>(qi.d_q, qi.d_h, threads, chunk_ops, producer_percent);
#else
            split_chunk_kernel<<<grid, block_size>>>(qi.d_q, qi.d_h, threads, chunk_ops, producer_percent);
#endif
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
#if HAS_SHARED_BUF
        split_chunk_kernel<<<grid, block_size, smem>>>(qi.d_q, qi.d_h, threads, chunk_ops, producer_percent);
#else
        split_chunk_kernel<<<grid, block_size>>>(qi.d_q, qi.d_h, threads, chunk_ops, producer_percent);
#endif
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

// ============================================================
// Main
// ============================================================

int main() {
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));

    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Queue: " << QUEUE_NAME << "\n";
    std::cout << "Method: fixed-duration timed throughput harness\n";
    std::cout << "RUN_MS=" << RUN_MS
              << " WARMUP_MS=" << WARMUP_MS
              << " CHUNK_OPS=" << CHUNK_OPS
              << " BLOCK_SIZE=" << BLOCK_SIZE << "\n\n";

    std::vector<int> thread_counts = {512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};

    // Producer splits to make imbalance explicit.
    // 25% matches Scogland's "1 of every 4 threads is producer".
    std::vector<int> producer_percents = {25, 50, 75, 90};

#if HAS_SHARED_BUF
    size_t smem = (size_t)BLOCK_SIZE * LOCAL_BUFFER_SIZE * sizeof(wf_buffer_entry);
    std::cout << "Shared mem/block: " << smem << " bytes\n\n";
#endif

    if (!ONLY_SPLIT) {
        std::cout << "=== Balanced Pairwise Timed Benchmark ===\n";
        std::cout << "Each active thread repeatedly performs enqueue then dequeue.\n\n";

        for (int threads : thread_counts) {
            QueueInstance qi = create_queue(threads);
            HIP_CHECK(hipDeviceSynchronize());

            double ms = run_balanced_timed(qi, threads, BLOCK_SIZE, CHUNK_OPS, WARMUP_MS, RUN_MS);
            HIP_CHECK(hipDeviceSynchronize());
            Stats s = fetch_stats();

            print_result_line("balanced", threads, ms, s);
            // print_wfq64_debug(qi);
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

                double ms = run_split_timed(qi, threads, BLOCK_SIZE, CHUNK_OPS, p, WARMUP_MS, RUN_MS);
                HIP_CHECK(hipDeviceSynchronize());
                Stats s = fetch_stats();

                std::string label = "split-" + std::to_string(p);
                print_result_line(label, threads, ms, s);
                // print_wfq64_debug(qi);
                destroy_queue(qi);
            }
            std::cout << "\n";
        }
    }

    std::cout << "Done.\n";
    return 0;
}