// // End of sfqueue_hip.cpp

#include "sfqueue_hip.hpp"

#include <cstdio>
#include <cstdlib>

// -----------------------------------------------------------------------------
// Device atomics: faithful to the paper's "all atomic_* calls map to the
// corresponding atomic intrinsic" model.
// -----------------------------------------------------------------------------

__device__ __forceinline__ uint32_t sfq_atomic_load_u32(const uint32_t* p) {
    return atomicAdd(reinterpret_cast<unsigned int*>(const_cast<uint32_t*>(p)), 0u);
}

__device__ __forceinline__ void sfq_atomic_store_u32(uint32_t* p, uint32_t v) {
    (void)atomicExch(reinterpret_cast<unsigned int*>(p), v);
}

__device__ __forceinline__ uint32_t sfq_atomic_add_u32(uint32_t* p, uint32_t v) {
    return atomicAdd(reinterpret_cast<unsigned int*>(p), v);
}

__device__ __forceinline__ uint32_t sfq_atomic_cas_u32(uint32_t* p,
                                                       uint32_t expected,
                                                       uint32_t desired) {
    return atomicCAS(reinterpret_cast<unsigned int*>(p), expected, desired);
}

__device__ __forceinline__ uint32_t sfq_next_id(uint32_t id) {
    const uint32_t next = id + 1u;
    return (next == SFQ_MAX_ID) ? 0u : next;
}

__device__ __forceinline__ void sfq_backoff() {
#pragma unroll
    for (int i = 0; i < SFQ_BACKOFF_ITERS; ++i) {
        __asm__ volatile("");
    }
}

// -----------------------------------------------------------------------------
// Core paper-faithful queue operations.
// -----------------------------------------------------------------------------

__device__ int sfq_is_closed(const sfq_queue* q) {
    return static_cast<int>(sfq_atomic_load_u32(&q->closed) != 0u);
}

__device__ void sfq_close(sfq_queue* q) {
    sfq_atomic_store_u32(&q->closed, 1u);
}

__device__ int sfq_enqueue_blocking_u32(sfq_queue* q, uint32_t item) {
    if (sfq_is_closed(q)) return SFQ_CLOSED;

    const uint32_t ticket = sfq_atomic_add_u32(&q->tail, 1u);
    const uint32_t target = ticket % SFQ_QUEUE_LENGTH;
    const uint32_t id     = SFQ_GET_ID(ticket);

    while (sfq_atomic_load_u32(&q->ids[target]) != id) {
        if (sfq_is_closed(q)) return SFQ_CLOSED;
        sfq_backoff();
    }

    sfq_atomic_store_u32(&q->items[target], item);
    sfq_atomic_store_u32(&q->ids[target], sfq_next_id(id));
    return SFQ_SUCCESS;
}

__device__ int sfq_dequeue_blocking_u32(sfq_queue* q, uint32_t* out) {
    if (sfq_is_closed(q)) return SFQ_CLOSED;

    const uint32_t ticket = sfq_atomic_add_u32(&q->head, 1u);
    const uint32_t target = ticket % SFQ_QUEUE_LENGTH;
    const uint32_t id     = SFQ_GET_ID(ticket) + 1u;

    while (sfq_atomic_load_u32(&q->ids[target]) != id) {
        if (sfq_is_closed(q)) return SFQ_CLOSED;
        sfq_backoff();
    }

    *out = sfq_atomic_load_u32(&q->items[target]);
    sfq_atomic_store_u32(&q->ids[target], sfq_next_id(id));
    return SFQ_SUCCESS;
}

__device__ int sfq_enqueue_nb_u32(sfq_queue* q, uint32_t item) {
    if (sfq_is_closed(q)) return SFQ_CLOSED;

    const uint32_t ticket = sfq_atomic_load_u32(&q->tail);
    const uint32_t target = ticket % SFQ_QUEUE_LENGTH;
    const uint32_t id     = SFQ_GET_ID(ticket);

    if (sfq_atomic_load_u32(&q->ids[target]) != id) {
        return SFQ_BUSY;
    }

    if (sfq_atomic_cas_u32(&q->tail, ticket, ticket + 1u) != ticket) {
        return SFQ_BUSY;
    }

    sfq_atomic_store_u32(&q->items[target], item);
    sfq_atomic_store_u32(&q->ids[target], sfq_next_id(id));
    return SFQ_SUCCESS;
}

__device__ int sfq_dequeue_nb_u32(sfq_queue* q, uint32_t* out) {
    if (sfq_is_closed(q)) return SFQ_CLOSED;

    const uint32_t ticket = sfq_atomic_load_u32(&q->head);
    const uint32_t target = ticket % SFQ_QUEUE_LENGTH;
    // Important: dequeue expects the odd phase for the current ticket.
    const uint32_t id     = SFQ_GET_ID(ticket) + 1u;

    if (sfq_atomic_load_u32(&q->ids[target]) != id) {
        return SFQ_BUSY;
    }

    if (sfq_atomic_cas_u32(&q->head, ticket, ticket + 1u) != ticket) {
        return SFQ_BUSY;
    }

    *out = sfq_atomic_load_u32(&q->items[target]);
    sfq_atomic_store_u32(&q->ids[target], sfq_next_id(id));
    return SFQ_SUCCESS;
}

// -----------------------------------------------------------------------------
// Compatibility API matching your harness.
// -----------------------------------------------------------------------------

__device__ void sfq_enqueue(sfq_queue* q, sfq_handle* /*h*/, uint64_t v) {
    uint32_t item = static_cast<uint32_t>(v & 0xFFFFFFFFu);
    if (item == 0u) item = 1u;  // Preserve SFQ_EMPTY == 0 sentinel in wrapper API.
    (void)sfq_enqueue_blocking_u32(q, item);
}

__device__ uint64_t sfq_dequeue(sfq_queue* q, sfq_handle* /*h*/) {
    uint32_t item = 0u;
    const int rc = sfq_dequeue_blocking_u32(q, &item);
    return (rc == SFQ_SUCCESS) ? static_cast<uint64_t>(item) : SFQ_EMPTY;
}

// -----------------------------------------------------------------------------
// Init helpers.
// -----------------------------------------------------------------------------

__device__ void sfq_queue_init(sfq_queue* q, uint32_t nprocs) {
    q->head   = 0u;
    q->tail   = 0u;
    q->closed = 0u;
    q->_pad0  = 0u;
    q->nprocs = nprocs;
}

__device__ void sfq_handle_init(sfq_handle* h, uint32_t tid) {
    h->thread_id = tid;
#pragma unroll
    for (int i = 0; i < 15; ++i) h->dummy[i] = 0u;
}

__global__ void sfq_init_kernel(sfq_queue* q, sfq_handle* handles, int num_threads) {
    const int tid   = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = blockDim.x * gridDim.x;

    if (tid == 0) {
        sfq_queue_init(q, static_cast<uint32_t>(num_threads));
    }

    for (int i = tid; i < static_cast<int>(SFQ_QUEUE_LENGTH); i += total) {
        q->items[i] = 0u;
        q->ids[i]   = 0u;
    }

    for (int i = tid; i < num_threads; i += total) {
        sfq_handle_init(&handles[i], static_cast<uint32_t>(i));
    }
}

// -----------------------------------------------------------------------------
// Test kernels kept compatible with your current harness.
// -----------------------------------------------------------------------------

__global__ void sfq_simple_test_kernel(sfq_queue* q, sfq_handle* handles,
                                       uint64_t* results, int num_threads,
                                       int ops_per_thread) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_threads) return;

    sfq_handle* h = &handles[tid];
    uint64_t successful_ops = 0;

    for (int i = 0; i < ops_per_thread; ++i) {
        const uint64_t value = static_cast<uint64_t>(tid) * 1000000ull + static_cast<uint64_t>(i) + 1ull;
        sfq_enqueue(q, h, value);
        ++successful_ops;

        if ((i % 10) == 0) {
            for (volatile int delay = 0; delay < 100; ++delay) {}
        }

        const uint64_t dequeued = sfq_dequeue(q, h);
        if (dequeued != SFQ_EMPTY) {
            ++successful_ops;
        }
    }

    results[tid] = successful_ops;
}

__global__ void sfq_high_contention_kernel(sfq_queue* q, sfq_handle* handles,
                                           uint64_t* results, int num_threads,
                                           int ops_per_thread) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_threads) return;

    sfq_handle* h = &handles[tid];
    uint64_t successful_ops = 0;
    const bool is_producer = (tid % 10) < 7;

    if (is_producer) {
        for (int i = 0; i < ops_per_thread; ++i) {
            const uint64_t value = (static_cast<uint64_t>(tid) << 32) | static_cast<uint64_t>(i + 1);
            sfq_enqueue(q, h, value);
            ++successful_ops;
        }
    } else {
        for (int i = 0; i < ops_per_thread * 3; ++i) {
            const uint64_t dequeued = sfq_dequeue(q, h);
            if (dequeued != SFQ_EMPTY) {
                ++successful_ops;
            }
            if ((i % 50) == 0) {
                for (volatile int delay = 0; delay < 10; ++delay) {}
            }
        }
    }

    results[tid] = successful_ops;
}

__global__ void sfq_memory_stress_kernel(sfq_queue* q, sfq_handle* handles,
                                         uint64_t* results, int num_threads,
                                         int ops_per_thread) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_threads) return;

    sfq_handle* h = &handles[tid];
    uint64_t successful_ops = 0;

    for (int i = 0; i < ops_per_thread; ++i) {
        const uint64_t value = (static_cast<uint64_t>(tid) << 20) | static_cast<uint64_t>(i + 1);
        sfq_enqueue(q, h, value);
        ++successful_ops;
    }

    __syncthreads();

    for (int i = 0; i < ops_per_thread; ++i) {
        const uint64_t dequeued = sfq_dequeue(q, h);
        if (dequeued != SFQ_EMPTY) {
            ++successful_ops;
        }
    }

    results[tid] = successful_ops;
}

__global__ void sfq_performance_test_kernel(sfq_queue* q, sfq_handle* handles,
                                            uint64_t* results,
                                            int operations_per_thread,
                                            int test_type) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= static_cast<int>(q->nprocs)) return;

    sfq_handle* h = &handles[tid];
    uint64_t local_ops = 0;

    __syncthreads();

    switch (test_type) {
        case 0:
            for (int i = 0; i < operations_per_thread; ++i) {
                const uint64_t val = static_cast<uint64_t>(tid) * 10000ull + static_cast<uint64_t>(i) + 1ull;
                sfq_enqueue(q, h, val);
                ++local_ops;
                const uint64_t dequeued = sfq_dequeue(q, h);
                if (dequeued != SFQ_EMPTY) ++local_ops;
            }
            break;
        case 1:
            for (int i = 0; i < operations_per_thread; ++i) {
                if ((i & 1) == 0) {
                    const uint64_t val = static_cast<uint64_t>(tid) * 10000ull + static_cast<uint64_t>(i) + 1ull;
                    sfq_enqueue(q, h, val);
                } else {
                    (void)sfq_dequeue(q, h);
                }
                ++local_ops;
            }
            break;
        case 2:
            for (int i = 0; i < operations_per_thread; ++i) {
                if ((i % 5) != 0) {
                    const uint64_t val = static_cast<uint64_t>(tid) * 10000ull + static_cast<uint64_t>(i) + 1ull;
                    sfq_enqueue(q, h, val);
                } else {
                    (void)sfq_dequeue(q, h);
                }
                ++local_ops;
            }
            break;
        case 3:
            for (int i = 0; i < operations_per_thread; ++i) {
                if ((i % 5) == 0) {
                    const uint64_t val = static_cast<uint64_t>(tid) * 10000ull + static_cast<uint64_t>(i) + 1ull;
                    sfq_enqueue(q, h, val);
                } else {
                    (void)sfq_dequeue(q, h);
                }
                ++local_ops;
            }
            break;
        default:
            break;
    }

    results[tid] = local_ops;
}

// -----------------------------------------------------------------------------
// Debug helpers.
// -----------------------------------------------------------------------------

__global__ void sfq_validate_kernel(sfq_queue* q, sfq_handle* handles, int num_threads) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        const uint32_t head   = sfq_atomic_load_u32(&q->head);
        const uint32_t tail   = sfq_atomic_load_u32(&q->tail);
        const uint32_t closed = sfq_atomic_load_u32(&q->closed);
        printf("=== SFQ Queue Validation ===\n");
        printf("head=%u tail=%u closed=%u nprocs=%u\n", head, tail, closed, q->nprocs);
        for (int i = 0; i < num_threads && i < 10; ++i) {
            printf("handle[%d].thread_id=%u\n", i, handles[i].thread_id);
        }
        printf("tail >= head (modulo rollover assumption): %s\n", (tail >= head) ? "PASS" : "CHECK_ROLLOVER");
        printf("=== End SFQ Validation ===\n");
    }
}

// -----------------------------------------------------------------------------
// Host helpers.
// -----------------------------------------------------------------------------

void sfq_queue_host_init(sfq_queue** d_q, sfq_handle** d_handles, int num_threads) {
    if (num_threads < 0) {
        std::fprintf(stderr, "sfq_queue_host_init: num_threads must be >= 0\n");
        std::abort();
    }
    if (static_cast<uint32_t>(num_threads) > SFQ_MAX_THREADS) {
        std::fprintf(stderr,
                     "sfq_queue_host_init: num_threads=%d exceeds SFQ_MAX_THREADS=%u\n",
                     num_threads, SFQ_MAX_THREADS);
        std::abort();
    }

    hipError_t err = hipMalloc(reinterpret_cast<void**>(d_q), sizeof(sfq_queue));
    if (err != hipSuccess) {
        std::fprintf(stderr, "hipMalloc(sfq_queue) failed: %s\n", hipGetErrorString(err));
        std::abort();
    }

    err = hipMalloc(reinterpret_cast<void**>(d_handles),
                    static_cast<size_t>(num_threads) * sizeof(sfq_handle));
    if (err != hipSuccess) {
        std::fprintf(stderr, "hipMalloc(sfq_handle[]) failed: %s\n", hipGetErrorString(err));
        std::abort();
    }

    constexpr int threads = 256;
    const int blocks = (num_threads > 0) ? ((num_threads + threads - 1) / threads) : 1;
    hipLaunchKernelGGL(sfq_init_kernel, dim3(blocks), dim3(threads), 0, 0,
                       *d_q, *d_handles, num_threads);
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::fprintf(stderr, "sfq_init_kernel failed: %s\n", hipGetErrorString(err));
        std::abort();
    }
}

void sfq_queue_destroy(sfq_queue* d_q, sfq_handle* d_h) {
    if (d_q) {
        (void)hipFree(d_q);
    }
    if (d_h) {
        (void)hipFree(d_h);
    }
}
