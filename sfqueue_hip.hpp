// #endif // SFQUEUE_HIP_H
#ifndef SFQUEUE_HIP_H
#define SFQUEUE_HIP_H

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <limits.h>

// Faithful to the queue structure/interfaces in Figure 4 of
// Scogland & Feng, "Design and Evaluation of Scalable Concurrent Queues
// for Many-Core Architectures" (ICPE 2015), adapted to HIP.

#ifndef SFQ_QUEUE_LENGTH
#define SFQ_QUEUE_LENGTH 65536u
#endif

#ifndef SFQ_MAX_THREADS
// Must satisfy: MAX_ID >= 2 * (MAX_THREADS + 1)
#define SFQ_MAX_THREADS 65535u
#endif

#ifndef SFQ_BACKOFF_ITERS
#define SFQ_BACKOFF_ITERS 2048
#endif

#define SFQ_EMPTY 0ull

#define SFQ_SUCCESS 0
#define SFQ_BUSY    1
#define SFQ_CLOSED  2

#define SFQ_MAX_DISTANCE (SFQ_QUEUE_LENGTH + SFQ_MAX_THREADS)

// GET_ID from the paper: number of complete passes through the ring, times 2.
#define SFQ_GET_ID(x) (((x) / SFQ_QUEUE_LENGTH) * 2u)

// Chosen so that when 32-bit head/tail roll over to zero, the slot id rolls to zero too.
// For power-of-two queue sizes this becomes 2^(33 - log2(QUEUE_SIZE)).
#define SFQ_MAX_ID (2u * ((UINT32_MAX / SFQ_QUEUE_LENGTH) + 1u))

#if ((SFQ_QUEUE_LENGTH & (SFQ_QUEUE_LENGTH - 1u)) != 0)
#error "SFQ_QUEUE_LENGTH must be a power of two."
#endif

#if (SFQ_MAX_ID < (2u * (SFQ_MAX_THREADS + 1u)))
#error "SFQ_MAX_ID invariant violated: reduce SFQ_MAX_THREADS or queue length."
#endif

#if (SFQ_MAX_DISTANCE >= 0x80000000u)
#error "SFQ_MAX_DISTANCE must be < 2^31 for single-counter rollover detection."
#endif

struct sfq_queue {
    uint32_t head;
    uint32_t tail;
    uint32_t closed;
    uint32_t _pad0;
    uint32_t items[SFQ_QUEUE_LENGTH];
    uint32_t ids[SFQ_QUEUE_LENGTH];
    uint32_t nprocs;
};

struct sfq_handle {
    uint32_t thread_id;
    uint32_t dummy[15];
};

// Main compatibility API used by your harness.
__device__ void     sfq_enqueue(sfq_queue* q, sfq_handle* h, uint64_t v);
__device__ uint64_t sfq_dequeue(sfq_queue* q, sfq_handle* h);

// Faithful paper-style interfaces.
__device__ int sfq_enqueue_blocking_u32(sfq_queue* q, uint32_t item);
__device__ int sfq_dequeue_blocking_u32(sfq_queue* q, uint32_t* out);
__device__ int sfq_enqueue_nb_u32(sfq_queue* q, uint32_t item);
__device__ int sfq_dequeue_nb_u32(sfq_queue* q, uint32_t* out);

// Optional helpers.
__device__ void sfq_close(sfq_queue* q);
__device__ int  sfq_is_closed(const sfq_queue* q);

__device__ void sfq_queue_init(sfq_queue* q, uint32_t nprocs);
__device__ void sfq_handle_init(sfq_handle* h, uint32_t tid);

__global__ void sfq_init_kernel(sfq_queue* q, sfq_handle* handles, int num_threads);
__global__ void sfq_simple_test_kernel(sfq_queue* q, sfq_handle* handles,
                                       uint64_t* results, int num_threads,
                                       int ops_per_thread);
__global__ void sfq_high_contention_kernel(sfq_queue* q, sfq_handle* handles,
                                           uint64_t* results, int num_threads,
                                           int ops_per_thread);
__global__ void sfq_memory_stress_kernel(sfq_queue* q, sfq_handle* handles,
                                         uint64_t* results, int num_threads,
                                         int ops_per_thread);
__global__ void sfq_performance_test_kernel(sfq_queue* q, sfq_handle* handles,
                                            uint64_t* results,
                                            int operations_per_thread,
                                            int test_type);
__global__ void sfq_validate_kernel(sfq_queue* q, sfq_handle* handles, int num_threads);

void sfq_queue_host_init(sfq_queue** d_q, sfq_handle** d_handles, int num_threads);
void sfq_queue_destroy(sfq_queue* d_q, sfq_handle* d_h);

#endif // SFQUEUE_HIP_H
