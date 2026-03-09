#pragma once
#ifndef WFQUEUE_BLFQ_HPP
#define WFQUEUE_BLFQ_HPP

#include <hip/hip_runtime.h>
#include <stdint.h>


/*
 * BLFQ v2: Cache-Aware Lock-Free Queue with Warp-Level FAA Batching
 * 
 * Design Goals:
 *   1. Lock-free (not wait-free) - trades progress guarantee for performance
 *   2. Match or exceed BWFQ throughput
 *   3. Reduce FAA contention via warp-level coordination
 *
 * Key Innovation:
 *   - Warp leader performs FAA(T/H, warp_count) to claim indices for entire warp
 *   - Each lane gets its index via warp shuffle - no additional atomics
 *   - Cell writes are naturally coalesced when warp threads write to adjacent cells
 *
 * Based on bounded ring buffer (like BWFQ) for simplicity and cache efficiency.
 */

/* ===================== Tunables ===================== */
#ifndef BLFQ_PATIENCE
#define BLFQ_PATIENCE 8           // Fast-path retries before giving up
#endif

#ifndef BLFQ_PREALLOC_OPS_PER_THREAD
#define BLFQ_PREALLOC_OPS_PER_THREAD 256
#endif

#ifndef BLFQ_SEGMENT_SAFETY
#define BLFQ_SEGMENT_SAFETY 8
#endif

/* ===================== Warp/Wavefront utilities ===================== */
// AMD CDNA uses 64-wide wavefronts
#define WARP_SIZE 64

// Get lane ID within wavefront
static __device__ __forceinline__ unsigned lane_id() {
    // __lane_id() is the HIP intrinsic
    return __lane_id();
}

// Count bits in mask
static __device__ __forceinline__ unsigned warp_popcount(uint64_t mask) {
    return __popcll(mask);
}

// Get position of this lane among active lanes (exclusive prefix popcount)
static __device__ __forceinline__ unsigned lane_offset(uint64_t mask, unsigned lane) {
    uint64_t lower_mask = (1ULL << lane) - 1;
    return __popcll(mask & lower_mask);
}

// Broadcast value from src_lane to all lanes
// HIP uses __shfl for different widths
static __device__ __forceinline__ uint64_t warp_broadcast(uint64_t val, unsigned src_lane) {
    // For 64-bit values on AMD, we need to shuffle both halves
    unsigned lo = __shfl((unsigned)(val & 0xFFFFFFFFULL), src_lane, WARP_SIZE);
    unsigned hi = __shfl((unsigned)(val >> 32), src_lane, WARP_SIZE);
    return ((uint64_t)hi << 32) | lo;
}

// Get ballot of predicate across wavefront (returns 64-bit mask on AMD)
static __device__ __forceinline__ uint64_t warp_ballot(int predicate) {
    return __ballot(predicate);
}

// Find first set bit (1-indexed, returns 0 if none)
static __device__ __forceinline__ unsigned find_first_lane(uint64_t mask) {
    return __ffsll((unsigned long long)mask);  // Returns 1-indexed position, 0 if no bits set
}

/* ===================== Sentinels ===================== */
#define WF_EMPTY 0ULL

/* ===================== Atomic helpers ===================== */
static __device__ __forceinline__ uint64_t cas64(uint64_t* addr, uint64_t expected, uint64_t desired) {
    return atomicCAS(reinterpret_cast<unsigned long long*>(addr),
                     static_cast<unsigned long long>(expected),
                     static_cast<unsigned long long>(desired));
}

static __device__ __forceinline__ uint64_t faa64(uint64_t* addr, uint64_t inc) {
    return atomicAdd(reinterpret_cast<unsigned long long*>(addr),
                     static_cast<unsigned long long>(inc));
}

static __device__ __forceinline__ uint64_t load64(uint64_t* addr) {
    return atomicAdd(reinterpret_cast<unsigned long long*>(addr), 0ULL);
}

static __device__ __forceinline__ void store64(uint64_t* addr, uint64_t val) {
    atomicExch(reinterpret_cast<unsigned long long*>(addr),
               static_cast<unsigned long long>(val));
}

/* ===================== Ring cell =====================
 * Sequence-number protocol (same as BWFQ):
 * - For slot index p:
 *   * FREE when seq == p
 *   * FULL when seq == p+1
 * - Enq at ticket p: wait seq==p, store val, set seq=p+1
 * - Deq at ticket p: wait seq==p+1, read val, set seq=p+cap
 */
struct __attribute__((aligned(16))) wf_cell {
    uint64_t seq;     // sequence number
    uint64_t val;     // payload (0 == empty)
};

/* ===================== Per-thread handle ===================== */
struct __attribute__((aligned(64))) wf_handle {
    wf_handle* next;  // Ring linkage (for compatibility)
    uint64_t pad[7];
};

/* ===================== Queue ===================== */
struct __attribute__((aligned(64))) wf_queue {
    // Ring buffer
    wf_cell*  ring;
    uint64_t  cap;        // Capacity (power of two)
    uint64_t  mask;       // cap - 1
    
    // Counters (the hot spots we're trying to reduce contention on)
    uint64_t  Tctr;       // Tail counter (enqueue)
    uint64_t  Hctr;       // Head counter (dequeue)
    
    // Metadata
    uint32_t  nprocs;
    uint32_t  _pad32;
    uint64_t  pad[4];
};

/* ===================== Warp-Batched Enqueue =====================
 *
 * Key insight: Instead of each thread doing FAA(T,1), the warp leader
 * does FAA(T, active_count) and distributes indices to lanes.
 *
 * This reduces FAA contention by up to 64x (warp size).
 *
 * IMPORTANT: This works correctly even with partial warp participation.
 * warp_ballot returns mask of ACTIVE lanes, so if only 30 threads call
 * this function, we batch 30 indices, not 64.
 *
 * USE THIS FOR BENCHMARKS where all warp threads call together.
 */
__device__ __forceinline__
void wf_enqueue_batched(wf_queue* q, wf_handle* h, uint64_t val) {
    (void)h;  // Not used in this implementation
    
    const unsigned lane = lane_id();
    
    // Step 1: Warp-level coordination
    // All lanes that want to enqueue vote
    uint64_t active_mask = warp_ballot(1);  // Who is participating?
    
    // Handle edge case: if somehow no lanes active (shouldn't happen)
    if (active_mask == 0) return;
    
    unsigned active_count = warp_popcount(active_mask);
    unsigned my_offset = lane_offset(active_mask, lane);
    
    // Step 2: Leader claims indices for entire warp with SINGLE FAA
    uint64_t base_ticket = 0;
    unsigned leader = find_first_lane(active_mask) - 1;  // Convert to 0-indexed
    
    if (lane == leader) {
        base_ticket = faa64(&q->Tctr, active_count);
    }
    
    // Step 3: Broadcast base ticket to all active lanes
    base_ticket = warp_broadcast(base_ticket, leader);
    
    // Step 4: Each lane computes its own ticket
    uint64_t my_ticket = base_ticket + my_offset;
    
    // Step 5: Write to our cell
    // Because warp threads have adjacent tickets, their cell accesses
    // are likely to be in the same or adjacent cache lines!
    wf_cell* c = &q->ring[my_ticket & q->mask];
    
    // Fast path: cell should be ready (seq == ticket)
    for (int patience = BLFQ_PATIENCE; patience >= 0; --patience) {
        uint64_t seq = load64(&c->seq);
        int64_t diff = (int64_t)seq - (int64_t)my_ticket;
        
        if (diff == 0) {
            // Slot is FREE for our ticket
            // For enqueue: write value, then advance seq
            store64(&c->val, val);
            __threadfence();  // Ensure value visible before advancing seq
            store64(&c->seq, my_ticket + 1);  // Mark as FULL
            return;  // Success!
        } else if (diff > 0) {
            // Already advanced - shouldn't happen, but handle gracefully
            return;
        }
        // diff < 0: Slot not yet free, keep waiting
    }
    
    // Patience exhausted - bounded spin
    while (true) {
        uint64_t seq = load64(&c->seq);
        int64_t diff = (int64_t)seq - (int64_t)my_ticket;
        
        if (diff == 0) {
            store64(&c->val, val);
            __threadfence();
            store64(&c->seq, my_ticket + 1);
            return;
        } else if (diff > 0) {
            return;
        }
        // Lock-free: keep spinning
    }
}

/* ===================== Standard Per-Thread Enqueue =====================
 * USE THIS FOR BFS and other applications where threads loop independently.
 * This is the classic bounded queue protocol without warp batching.
 */
__device__ __forceinline__
void wf_enqueue(wf_queue* q, wf_handle* h, uint64_t val) {
    (void)h;
    
    // Each thread claims its own ticket
    uint64_t my_ticket = faa64(&q->Tctr, 1);
    
    wf_cell* c = &q->ring[my_ticket & q->mask];
    
    // Wait for slot to be free (seq == ticket)
    for (int patience = BLFQ_PATIENCE; patience >= 0; --patience) {
        uint64_t seq = load64(&c->seq);
        int64_t diff = (int64_t)seq - (int64_t)my_ticket;
        
        if (diff == 0) {
            // Slot is FREE - write value and mark FULL
            store64(&c->val, val);
            __threadfence();
            store64(&c->seq, my_ticket + 1);
            return;
        } else if (diff > 0) {
            // Already advanced (shouldn't happen)
            return;
        }
        // diff < 0: Slot not yet free, retry
    }
    
    // Bounded spin for remaining cases
    while (true) {
        uint64_t seq = load64(&c->seq);
        int64_t diff = (int64_t)seq - (int64_t)my_ticket;
        
        if (diff == 0) {
            store64(&c->val, val);
            __threadfence();
            store64(&c->seq, my_ticket + 1);
            return;
        } else if (diff > 0) {
            return;
        }
    }
}

/* ===================== Warp-Batched Dequeue ===================== 
 * USE THIS FOR BENCHMARKS where all warp threads call together
 */
__device__ __forceinline__
uint64_t wf_dequeue_batched(wf_queue* q, wf_handle* h) {
    (void)h;
    
    const unsigned lane = lane_id();
    
    // Step 1: Warp-level coordination
    uint64_t active_mask = warp_ballot(1);
    
    if (active_mask == 0) return WF_EMPTY;
    
    unsigned active_count = warp_popcount(active_mask);
    unsigned my_offset = lane_offset(active_mask, lane);
    
    // Step 2: Leader claims indices
    uint64_t base_ticket = 0;
    unsigned leader = find_first_lane(active_mask) - 1;  // Convert to 0-indexed
    
    if (lane == leader) {
        base_ticket = faa64(&q->Hctr, active_count);
    }
    
    // Step 3: Broadcast and compute individual ticket
    base_ticket = warp_broadcast(base_ticket, leader);
    uint64_t my_ticket = base_ticket + my_offset;
    
    // Step 4: Read from our cell
    wf_cell* c = &q->ring[my_ticket & q->mask];
    
    for (int patience = BLFQ_PATIENCE; patience >= 0; --patience) {
        uint64_t seq = load64(&c->seq);
        int64_t diff = (int64_t)seq - (int64_t)(my_ticket + 1);
        
        if (diff == 0) {
            // Slot is FULL for our ticket - read and clear
            uint64_t val = load64(&c->val);
            store64(&c->val, 0ULL);  // Clear value
            __threadfence();  // Clear visible before freeing slot
            store64(&c->seq, my_ticket + q->cap);  // Mark as FREE for future round
            return val ? val : WF_EMPTY;
        } else if (diff < 0) {
            // Slot not yet full - check if queue is empty
            uint64_t tail = load64(&q->Tctr);
            if (tail <= my_ticket) {
                // Queue is empty relative to our ticket
                return WF_EMPTY;
            }
            // Producer hasn't finished yet - retry
        } else {
            // diff > 0: Already consumed (shouldn't happen)
            return WF_EMPTY;
        }
    }
    
    // Patience exhausted - bounded spin
    for (int spin = 0; spin < 1000; ++spin) {
        uint64_t seq = load64(&c->seq);
        int64_t diff = (int64_t)seq - (int64_t)(my_ticket + 1);
        
        if (diff == 0) {
            uint64_t val = load64(&c->val);
            store64(&c->val, 0ULL);
            __threadfence();
            store64(&c->seq, my_ticket + q->cap);
            return val ? val : WF_EMPTY;
        } else if (diff < 0) {
            uint64_t tail = load64(&q->Tctr);
            if (tail <= my_ticket) {
                return WF_EMPTY;
            }
        } else {
            return WF_EMPTY;
        }
    }
    
    return WF_EMPTY;
}

/* ===================== Standard Per-Thread Dequeue ===================== 
 * USE THIS FOR BFS and other applications where threads loop independently.
 * This is the classic bounded queue protocol without warp batching.
 */
__device__ __forceinline__
uint64_t wf_dequeue(wf_queue* q, wf_handle* h) {
    (void)h;
    
    // Each thread claims its own ticket
    uint64_t my_ticket = faa64(&q->Hctr, 1);
    
    wf_cell* c = &q->ring[my_ticket & q->mask];
    
    for (int patience = BLFQ_PATIENCE; patience >= 0; --patience) {
        uint64_t seq = load64(&c->seq);
        int64_t diff = (int64_t)seq - (int64_t)(my_ticket + 1);
        
        if (diff == 0) {
            // Slot is FULL - read and clear
            uint64_t val = load64(&c->val);
            store64(&c->val, 0ULL);
            __threadfence();
            store64(&c->seq, my_ticket + q->cap);  // FREE for next round
            return val ? val : WF_EMPTY;
        } else if (diff < 0) {
            // Slot not yet full - check if queue is empty
            uint64_t tail = load64(&q->Tctr);
            if (tail <= my_ticket) {
                return WF_EMPTY;  // Queue is empty
            }
            // Producer slow, retry
        } else {
            // Already consumed (shouldn't happen)
            return WF_EMPTY;
        }
    }
    
    // Bounded spin
    for (int spin = 0; spin < 100; ++spin) {
        uint64_t seq = load64(&c->seq);
        int64_t diff = (int64_t)seq - (int64_t)(my_ticket + 1);
        
        if (diff == 0) {
            uint64_t val = load64(&c->val);
            store64(&c->val, 0ULL);
            __threadfence();
            store64(&c->seq, my_ticket + q->cap);
            return val ? val : WF_EMPTY;
        } else if (diff < 0) {
            uint64_t tail = load64(&q->Tctr);
            if (tail <= my_ticket) {
                return WF_EMPTY;
            }
        } else {
            return WF_EMPTY;
        }
    }
    
    return WF_EMPTY;
}

/* ===================== Initialization Kernel ===================== */
__global__ void wf_init_kernel(wf_queue* q, wf_handle* h, int nthreads) {
    const uint64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    
    // Initialize counters (single thread)
    if (gid == 0) {
        q->Tctr = 0;
        q->Hctr = 0;
        q->nprocs = nthreads;
    }
    
    // Initialize handles
    for (uint64_t i = gid; i < (uint64_t)nthreads; i += stride) {
        h[i].next = &h[(i + 1) % nthreads];
    }
    
    // Initialize ring cells
    for (uint64_t p = gid; p < q->cap; p += stride) {
        q->ring[p].seq = p;      // FREE state
        q->ring[p].val = 0ULL;   // Empty value
    }
}

/* ===================== Host Initialization ===================== */
static inline uint64_t round_up_pow2(uint64_t x) {
    if (x <= 1) return 1;
    --x;
    x |= x >> 1;  x |= x >> 2;  x |= x >> 4;
    x |= x >> 8;  x |= x >> 16; x |= x >> 32;
    return x + 1;
}

inline void wf_queue_host_init(wf_queue** d_q, wf_handle** d_h, int num_threads) {
    // Calculate ring capacity
    uint64_t need = (uint64_t)num_threads * BLFQ_PREALLOC_OPS_PER_THREAD
                    + (uint64_t)(BLFQ_SEGMENT_SAFETY * 1024ULL);
    uint64_t cap = round_up_pow2(need);
    if (cap < 1024) cap = 1024;
    
    // Allocate queue and handles
    hipMalloc(reinterpret_cast<void**>(d_q), sizeof(wf_queue));
    hipMalloc(reinterpret_cast<void**>(d_h), (size_t)num_threads * sizeof(wf_handle));
    
    // Allocate ring buffer
    wf_cell* d_ring = nullptr;
    hipMalloc(reinterpret_cast<void**>(&d_ring), (size_t)cap * sizeof(wf_cell));
    
    // Initialize queue header
    wf_queue hq{};
    hq.ring = d_ring;
    hq.cap = cap;
    hq.mask = cap - 1;
    hq.Tctr = 0;
    hq.Hctr = 0;
    hq.nprocs = num_threads;
    
    hipMemcpy(*d_q, &hq, sizeof(wf_queue), hipMemcpyHostToDevice);
    
    // Run initialization kernel
    const int block = 256;
    int grid = (num_threads + block - 1) / block;
    if (grid < 80) grid = 80;
    wf_init_kernel<<<grid, block>>>(*d_q, *d_h, num_threads);
    hipDeviceSynchronize();
}


// reset kernel for BFS
// BFS-optimised reset: only clears slots actually used this level.
// used_slots = Tctr value (items enqueued this level) — caller knows this.
__global__ void wf_bfs_reset_kernel(wf_queue* q, wf_handle* h,
                                     uint64_t used_slots, int nthreads) {
    const uint64_t gid    = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride = (uint64_t)gridDim.x  * blockDim.x;

    if (gid == 0) {
        q->Tctr = 0;
        q->Hctr = 0;
    }

    // Reset handles (always small — nthreads, not cap)
    for (uint64_t i = gid; i < (uint64_t)nthreads; i += stride)
        h[i].next = &h[(i + 1) % (uint64_t)nthreads];

    // Only clear slots that were actually written this level
    for (uint64_t p = gid; p < used_slots; p += stride) {
        q->ring[p].seq = p;
        q->ring[p].val = 0ULL;
    }
}

inline void wf_bfs_reset(wf_queue* d_q, wf_handle* d_h,
                          int num_threads, uint64_t used_slots) {
    if (used_slots == 0) used_slots = 1;

    uint64_t work = (used_slots > (uint64_t)num_threads) ? used_slots : (uint64_t)num_threads;
    const int block = 256;
    int grid = (int)((work + block - 1) / block);
    if (grid < 4)     grid = 4;
    if (grid > 65535) grid = 65535;

    wf_bfs_reset_kernel<<<grid, block>>>(d_q, d_h, used_slots, num_threads);
    // hipDeviceSynchronize();
}


/* ===================== Reset for BFS (preserves ring allocation) ===================== */





__global__ void wf_reset_kernel(wf_queue* q, wf_handle* h, int nthreads) {
    const uint64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    
    // Reset counters (single thread)
    if (gid == 0) {
        q->Tctr = 0;
        q->Hctr = 0;
    }
    
    // Reset handles
    for (uint64_t i = gid; i < (uint64_t)nthreads; i += stride) {
        h[i].next = &h[(i + 1) % nthreads];
    }
    
    // Reset ring cells
    for (uint64_t p = gid; p < q->cap; p += stride) {
        q->ring[p].seq = p;      // FREE state
        q->ring[p].val = 0ULL;   // Empty value
    }
}

inline void wf_queue_reset(wf_queue* d_q, wf_handle* d_h, int num_threads) {
    // Read capacity from device
    wf_queue hq{};
    hipMemcpy(&hq, d_q, sizeof(wf_queue), hipMemcpyDeviceToHost);
    
    const int block = 256;
    uint64_t work = (hq.cap > (uint64_t)num_threads) ? hq.cap : (uint64_t)num_threads;
    int grid = (int)((work + block - 1) / block);
    if (grid < 80) grid = 80;
    if (grid > 65535) grid = 65535;
    
    wf_reset_kernel<<<grid, block>>>(d_q, d_h, num_threads);
    hipDeviceSynchronize();
}

/* ===================== Cleanup ===================== */
inline void wf_queue_destroy(wf_queue* d_q, wf_handle* d_h) {
    wf_queue hq{};
    hipMemcpy(&hq, d_q, sizeof(wf_queue), hipMemcpyDeviceToHost);
    
    if (hq.ring) {
        hipFree(hq.ring);
    }
    hipFree(d_q);
    hipFree(d_h);
}

#endif // WFQUEUE_BLFQ_HPP