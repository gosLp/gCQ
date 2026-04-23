/**
 * fast_queue_gpu.h — High-Throughput Lock-Free MPMC FIFO Queue for GPU
 *
 * Design philosophy: THROUGHPUT FIRST.
 *   - Lock-free (not wait-free) — simpler, faster common path
 *   - Warp-batched FAA on Head/Tail — GPU-native Aggregating Funnels
 *   - 64-bit packed entries — modular cycle from our wCQ work
 *   - No slow path, no helper descriptors, no phase2 machinery
 *
 * The key insight: GPU wavefront/warp intrinsics (__ballot_sync, popc,
 * prefix sum, __shfl_sync) implement Aggregating Funnels (PPoPP'25) with
 * zero software overhead. 64 threads (AMD) or 32 threads (NVIDIA) get
 * unique tickets from ONE atomicAdd.
 *
 * Based on: SCQ (Nikolaev, DISC'19) for the ring buffer algorithm
 *           wCQ (Nikolaev & Ravindran, SPAA'22) for 64-bit packing
 *           Aggregating Funnels (Roh et al., PPoPP'25) for batching insight
 *
 * Progress: Lock-free (at least one thread always makes progress)
 * Memory:   Bounded (static ring buffer, no dynamic allocation)
 * FIFO:     Linearizable
 */

#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

/*============================================================================
 * CONFIGURATION
 *============================================================================*/

#ifndef FQ_N
#define FQ_N 32768           /* Buffer capacity. Must be power of 2.        */
#endif

#ifndef FQ_CYCLE_BITS
#define FQ_CYCLE_BITS 8     /* Modular cycle width. 8 bits = safe to 127. */
#endif

/*============================================================================
 * CONSTANTS
 *============================================================================*/

static constexpr uint32_t FQ_SLOTS       = 2 * FQ_N;
static constexpr int32_t  FQ_THRESH_MAX  = 3 * (int32_t)FQ_N - 1;
static constexpr uint32_t FQ_CYCLE_MASK  = (1u << FQ_CYCLE_BITS) - 1;
static constexpr uint32_t FQ_HALF_CYCLE  = 1u << (FQ_CYCLE_BITS - 1);
static constexpr uint32_t FQ_BOT         = 0xFFFFFFFEu;  /* ⊥  : empty    */
static constexpr uint32_t FQ_BOT_C       = 0xFFFFFFFFu;  /* ⊥c : consumed */
static constexpr uint32_t FQ_CACHELINE   = 64;
static constexpr uint32_t FQ_ENTRIES_PER_CL = FQ_CACHELINE / sizeof(uint64_t);

/*============================================================================
 * ENTRY — 64-bit packed (simplified from wCQ: no Note, no Enq bit)
 *
 *   [55..48]  Cycle    8 bits  (modular)
 *   [47]      IsSafe   1 bit
 *   [46..32]  reserved 15 bits (must be 0 for OR correctness)
 *   [31..0]   Index    32 bits
 *
 * No Note field: there's no slow path with cooperative helpers.
 * No Enq bit: no two-step insertion protocol.
 * This is 16 bits simpler than wCQ entries.
 *============================================================================*/

namespace ent {

static constexpr int CYCLE_SHIFT  = 48;
static constexpr int SAFE_SHIFT   = 47;
static constexpr uint64_t CYCLE_FIELD = (uint64_t)FQ_CYCLE_MASK << CYCLE_SHIFT;
static constexpr uint64_t SAFE_BIT    = 1ULL << SAFE_SHIFT;
static constexpr uint64_t INDEX_MASK  = 0xFFFFFFFFULL;

/* OR mask for consume: sets Index = ⊥c, preserves Cycle and IsSafe. */
static constexpr uint64_t CONSUME_MASK = (uint64_t)FQ_BOT_C;

__device__ __forceinline__ uint32_t cycle(uint64_t e) { return (uint32_t)((e >> CYCLE_SHIFT) & FQ_CYCLE_MASK); }
__device__ __forceinline__ bool     safe(uint64_t e)  { return (e >> SAFE_SHIFT) & 1; }
__device__ __forceinline__ uint32_t index(uint64_t e) { return (uint32_t)(e & INDEX_MASK); }

__device__ __forceinline__
uint64_t pack(uint32_t cyc, bool is_safe, uint32_t idx) {
    return ((uint64_t)(cyc & FQ_CYCLE_MASK) << CYCLE_SHIFT)
         | ((uint64_t)is_safe << SAFE_SHIFT)
         | (uint64_t)idx;
}

static constexpr uint64_t INIT = SAFE_BIT | (uint64_t)FQ_BOT;  /* Cycle=0, Safe=1, Index=⊥ */

} /* namespace ent */

/*============================================================================
 * CYCLE COMPARISON — Modular signed arithmetic
 *============================================================================*/

__device__ __forceinline__
uint32_t cycle_of(uint64_t ticket) {
    return (uint32_t)((ticket / FQ_SLOTS) & FQ_CYCLE_MASK);
}

__device__ __forceinline__
uint32_t slot_of(uint64_t ticket) {
    return (uint32_t)(ticket & (FQ_SLOTS - 1));
}

__device__ __forceinline__
int cycle_cmp(uint32_t a, uint32_t b) {
    uint32_t d = (a - b) & FQ_CYCLE_MASK;
    return d == 0 ? 0 : d < FQ_HALF_CYCLE ? 1 : -1;
}

/*============================================================================
 * CACHE REMAP — Anti-false-sharing permutation
 *============================================================================*/

__device__ __forceinline__
uint32_t cache_remap(uint32_t pos) {
    return ((pos & (FQ_ENTRIES_PER_CL - 1)) * (FQ_SLOTS / FQ_ENTRIES_PER_CL))
         + (pos / FQ_ENTRIES_PER_CL);
}

/*============================================================================
 * QUEUE STRUCT — Plain counters (no ThrIdx packing, no slow path)
 *============================================================================*/

struct fq_ring_t {
    unsigned long long Tail;              /* Plain 64-bit counter */
    unsigned long long Head;              /* Plain 64-bit counter */
    int32_t            Threshold;
    int32_t            _pad;
    unsigned long long Entry[FQ_SLOTS];   /* 64-bit packed entries */
};

struct fq_mpmc_t {
    fq_ring_t aq;                         /* Allocated queue */
    fq_ring_t fq;                         /* Free queue */
    uint64_t  data[FQ_N];                 /* Payload array */
};

/*============================================================================
 * ATOMIC HELPERS
 *============================================================================*/

__device__ __forceinline__
uint64_t aload(unsigned long long* p) { return (uint64_t)atomicAdd(p, 0ULL); }

__device__ __forceinline__
int32_t aload32(int32_t* p) { return atomicAdd(p, 0); }

__device__ __forceinline__
bool cas(unsigned long long* p, uint64_t exp, uint64_t des) {
    return atomicCAS(p, (unsigned long long)exp, (unsigned long long)des)
           == (unsigned long long)exp;
}

__device__ __forceinline__
bool cas_upd(unsigned long long* p, uint64_t* exp, uint64_t des) {
    uint64_t old = (uint64_t)atomicCAS(p, (unsigned long long)*exp,
                                       (unsigned long long)des);
    bool ok = (old == *exp);
    *exp = old;
    return ok;
}

/*============================================================================
 * WAVEFRONT / WARP SIZE DETECTION
 *
 * AMD GCN/CDNA/RDNA: 64-wide wavefronts (RDNA can be 32, but 64 is default)
 * NVIDIA: 32-wide warps
 *
 * HIP requires 64-bit masks for __ballot_sync, __shfl_sync, etc.
 * CUDA uses 32-bit masks.
 *
 * We detect this at compile time and use the correct types.
 *============================================================================*/

 #ifdef __HIP_PLATFORM_AMD__
  using wave_mask_t = uint64_t;
  #define WAVE_SIZE      64
  #define WAVE_FULL_MASK 0xFFFFFFFFFFFFFFFFULL
  #define WAVE_LANE_MASK 63

  __device__ __forceinline__
  uint32_t wave_popc(wave_mask_t m) {
      return (uint32_t)__popcll((unsigned long long)m);
  }

  __device__ __forceinline__
  uint32_t wave_ffs(wave_mask_t m) {
      return (uint32_t)__ffsll((unsigned long long)m);
  }

  __device__ __forceinline__
  wave_mask_t wave_ballot(bool pred) {
      return (wave_mask_t)__ballot((int)pred);
  }

  __device__ __forceinline__
  wave_mask_t wave_activemask() {
      return (wave_mask_t)__ballot(1);
  }

  __device__ __forceinline__
  uint32_t wave_lane_id() {
      return threadIdx.x & WAVE_LANE_MASK;
  }

  __device__ __forceinline__
  uint32_t wave_shfl_u32(uint32_t v, int src_lane) {
      return __shfl(v, src_lane);
  }

#else
  using wave_mask_t = uint32_t;
  #define WAVE_SIZE      32
  #define WAVE_FULL_MASK 0xFFFFFFFFu
  #define WAVE_LANE_MASK 31

  __device__ __forceinline__
  uint32_t wave_popc(wave_mask_t m) {
      return (uint32_t)__popc((unsigned int)m);
  }

  __device__ __forceinline__
  uint32_t wave_ffs(wave_mask_t m) {
      return (uint32_t)__ffs((unsigned int)m);
  }

  __device__ __forceinline__
  wave_mask_t wave_ballot(bool pred) {
      return (wave_mask_t)__ballot_sync(WAVE_FULL_MASK, pred);
  }

  __device__ __forceinline__
  wave_mask_t wave_activemask() {
      return (wave_mask_t)__activemask();
  }

  __device__ __forceinline__
  uint32_t wave_lane_id() {
      return threadIdx.x & WAVE_LANE_MASK;
  }

  __device__ __forceinline__
  uint32_t wave_shfl_u32(uint32_t v, int src_lane) {
      return __shfl_sync(WAVE_FULL_MASK, v, src_lane);
  }
#endif

__device__ __forceinline__
uint64_t wave_shfl_u64(uint64_t v, int src_lane) {
    uint32_t lo = (uint32_t)v;
    uint32_t hi = (uint32_t)(v >> 32);
    lo = wave_shfl_u32(lo, src_lane);
    hi = wave_shfl_u32(hi, src_lane);
    return ((uint64_t)hi << 32) | lo;
}



// #ifdef __HIP_PLATFORM_AMD__
//   /* AMD: 64-wide wavefronts, 64-bit masks required by HIP API */
//   using wave_mask_t = uint64_t;
//   #define WAVE_SIZE        64
//   #define WAVE_FULL_MASK   0xFFFFFFFFFFFFFFFFULL
//   #define WAVE_LANE_MASK   63

//   __device__ __forceinline__
//   uint32_t wave_popc(wave_mask_t m)  { return __popcll(m); }

//   __device__ __forceinline__
//   uint32_t wave_ffs(wave_mask_t m)   { return (uint32_t)__ffsll((unsigned long long)m); }  /* returns 1-indexed, 0 if none */

// #else
//   /* NVIDIA: 32-wide warps, 32-bit masks */
//   using wave_mask_t = uint32_t;
//   #define WAVE_SIZE        32
//   #define WAVE_FULL_MASK   0xFFFFFFFFu
//   #define WAVE_LANE_MASK   31

//   __device__ __forceinline__
//   uint32_t wave_popc(wave_mask_t m)  { return __popc(m); }

//   __device__ __forceinline__
//   uint32_t wave_ffs(wave_mask_t m)   { return __ffs(m); }

// #endif

/*============================================================================
 * WARP/WAVEFRONT-BATCHED FAA 
 *
 * This is the performance-critical innovation. Instead of every thread
 * doing atomicAdd(&Tail, 1), the wavefront does:
 *
 *   1. __ballot_sync to find which lanes want to participate
 *   2. popc to count participants
 *   3. ONE leader atomicAdd(&Tail, count)
 *   4. __shfl_sync to broadcast the base ticket
 *   5. Wavefront prefix sum to assign unique offsets
 *
 * Result: up to 64 unique tickets from ONE atomic (AMD).
 *         up to 32 unique tickets from ONE atomic (NVIDIA).
 *
 *============================================================================*/

/**
 * Wavefront-batched atomic increment.
 *
 * @param counter  Pointer to the global counter (Tail or Head)
 * @param active   true if this lane wants a ticket
 * @return         This lane's unique ticket (raw counter value).
 *                 Only valid if active==true.
 *
 * All lanes in the wavefront MUST call this (even inactive ones), since
 * __ballot_sync requires full wavefront participation.
 */
// __device__ __forceinline__
// uint64_t wave_batched_faa(unsigned long long* counter, bool active) {
//     /* 1. Which lanes want tickets? */
//     wave_mask_t mask = __ballot_sync(WAVE_FULL_MASK, active);

//     if (mask == 0) return 0;

//     /* 2. How many tickets does the wavefront need? */
//     uint32_t count = wave_popc(mask);

//     /* 3. Leader (lowest active lane) does ONE atomicAdd. */
//     uint32_t leader = wave_ffs(mask) - 1;
//     uint32_t lane = threadIdx.x & WAVE_LANE_MASK;
//     uint64_t base = 0;
//     if (lane == leader) {
//         base = (uint64_t)atomicAdd(counter, (unsigned long long)count);
//     }

//     /* 4. Broadcast base to all lanes. */
//     base = __shfl_sync(mask, base, leader);

//     /* 5. Each active lane's offset = popcount of lower active bits.
//      *    For 64-bit mask: use uint64_t shift to avoid UB. */
//     // wave_mask_t lower_mask = mask & (((wave_mask_t)1 << lane) - 1);
//     wave_mask_t lower_mask = (lane == 0)
//         ? (wave_mask_t)0
//         : (mask & ((((wave_mask_t)1) << lane) - 1));
//     uint32_t offset = wave_popc(lower_mask);

//     return base + offset;
// }

// /**
//  * Variant: wavefront-batched FAA for divergent control flow.
//  * Uses __activemask() instead of requiring all lanes.
//  */
// __device__ __forceinline__
// uint64_t wave_batched_faa_divergent(unsigned long long* counter) {
//     wave_mask_t mask = __activemask();
//     uint32_t count = wave_popc(mask);
//     uint32_t leader = wave_ffs(mask) - 1;
//     uint32_t lane = threadIdx.x & WAVE_LANE_MASK;
//     uint64_t base = 0;
//     if (lane == leader)
//         base = (uint64_t)atomicAdd(counter, (unsigned long long)count);
//     base = __shfl_sync(mask, base, leader);
//     wave_mask_t lower_mask = mask & (((wave_mask_t)1 << lane) - 1);
//     uint32_t offset = wave_popc(lower_mask);
//     return base + offset;
// }

__device__ __forceinline__
uint64_t wave_batched_faa(unsigned long long* counter, bool active) {
    wave_mask_t mask = wave_ballot(active);
    if (mask == 0) return 0;

    uint32_t count  = wave_popc(mask);
    uint32_t leader = wave_ffs(mask) - 1;
    uint32_t lane   = wave_lane_id();

    uint64_t base = 0;
    if (lane == leader) {
        base = (uint64_t)atomicAdd(counter, (unsigned long long)count);
    }

    base = wave_shfl_u64(base, leader);

    wave_mask_t lower_mask = (lane == 0)
        ? (wave_mask_t)0
        : (mask & ((((wave_mask_t)1) << lane) - 1));

    uint32_t offset = wave_popc(lower_mask);
    return active ? (base + offset) : 0;
}

__device__ __forceinline__
uint64_t wave_batched_faa_divergent(unsigned long long* counter) {
    wave_mask_t mask = wave_activemask();
    uint32_t count  = wave_popc(mask);
    uint32_t leader = wave_ffs(mask) - 1;
    uint32_t lane   = wave_lane_id();

    uint64_t base = 0;
    if (lane == leader) {
        base = (uint64_t)atomicAdd(counter, (unsigned long long)count);
    }

    base = wave_shfl_u64(base, leader);

    wave_mask_t lower_mask = (lane == 0)
        ? (wave_mask_t)0
        : (mask & ((((wave_mask_t)1) << lane) - 1));

    uint32_t offset = wave_popc(lower_mask);
    return base + offset;
}

/*============================================================================
 * INIT
 *============================================================================*/

__device__ void fq_ring_init_empty(fq_ring_t* q) {
    q->Tail = (unsigned long long)FQ_SLOTS;
    q->Head = (unsigned long long)FQ_SLOTS;
    q->Threshold = -1;
    for (uint32_t i = 0; i < FQ_SLOTS; i++)
        q->Entry[i] = (unsigned long long)ent::INIT;
}

__device__ void fq_ring_init_full(fq_ring_t* q) {
    fq_ring_init_empty(q);
    for (uint32_t i = 0; i < FQ_N; i++) {
        uint64_t ticket = FQ_SLOTS + i;
        uint32_t j = cache_remap(slot_of(ticket));
        q->Entry[j] = (unsigned long long)ent::pack(cycle_of(ticket), true, i);
    }
    q->Tail = (unsigned long long)(FQ_SLOTS + FQ_N);
    q->Threshold = FQ_THRESH_MAX;
}

__device__ void fq_init(fq_mpmc_t* q) {
    fq_ring_init_empty(&q->aq);
    fq_ring_init_full(&q->fq);
}

/*============================================================================
 * CATCHUP — Advance Tail to Head when Tail falls behind
 * Paper Figure 3 lines 5-10. Bounded for lock-freedom.
 *============================================================================*/

__device__ __forceinline__
void catchup(fq_ring_t* q, uint64_t tail, uint64_t head) {
    for (int i = 0; i < 32; i++) {
        if (cas(&q->Tail, tail, head)) break;
        head = aload(&q->Head);
        tail = aload(&q->Tail);
        if (tail >= head) break;
    }
}


/*============================================================================
 * ENQUEUE — Warp-batched, lock-free
 *
 * Up to 32 threads get unique slots in ONE atomic operation.
 * Each thread then independently CAS's its assigned entry.
 * Since entries are in different cache lines (via cache_remap),
 * the per-entry CAS operations run in parallel with minimal contention.
 *============================================================================*/

__device__ void fq_enqueue(fq_ring_t* q, uint32_t index) {
    // __shared__ BlockTicketDispenser s_tail_disp;
    // __shared__ uint32_t s_tail_init;

    // init_dispenser(&s_tail_disp, &s_tail_init);

    /* Warp-batched FAA on Tail. */
    while (true) {
        uint64_t T = wave_batched_faa_divergent(&q->Tail);
        // uint64_t T = block_batched_faa<512>(&q->Tail, &s_tail_disp);
        uint32_t tcyc = cycle_of(T);
        uint32_t j = cache_remap(slot_of(T));

        uint64_t E = aload(&q->Entry[j]);
    retry:
        uint32_t ecyc = ent::cycle(E);
        uint32_t eidx = ent::index(E);

        /* SCQ enqueue condition: entry cycle behind ours, slot is empty. */
        if (cycle_cmp(ecyc, tcyc) < 0
            && (ent::safe(E) || aload(&q->Head) <= T)
            && (eidx == FQ_BOT || eidx == FQ_BOT_C))
        {
            uint64_t New = ent::pack(tcyc, true, index);
            if (!cas_upd(&q->Entry[j], &E, New))
                goto retry;
            /* Success. Reset threshold. */
            if (aload32(&q->Threshold) != FQ_THRESH_MAX)
                atomicExch(&q->Threshold, FQ_THRESH_MAX);
            return;
        }
        /* Slot unavailable — loop gets a new ticket via warp-batched FAA. */
    }
}

/*============================================================================
 * DEQUEUE — Warp-batched, lock-free
 *
 * Same warp-batching pattern. Additionally uses warp ballot to
 * collectively detect empty queue (if ALL lanes in the warp fail,
 * the queue is likely empty).
 *============================================================================*/

__device__ uint32_t fq_dequeue(fq_ring_t* q) {
    /* Quick empty check. */
    if (aload32(&q->Threshold) < 0)
        return FQ_BOT;

    while (true) {
        uint64_t H = wave_batched_faa_divergent(&q->Head);
        uint32_t hcyc = cycle_of(H);
        uint32_t j = cache_remap(slot_of(H));

        uint64_t E = aload(&q->Entry[j]);
    retry_inner:;
        uint32_t ecyc = ent::cycle(E);
        uint32_t eidx = ent::index(E);

        /* Entry ready: cycle matches and has a real index. */
        if (cycle_cmp(ecyc, hcyc) == 0
            && eidx != FQ_BOT && eidx != FQ_BOT_C)
        {
            /* Consume via OR: sets Index=⊥c, preserves Cycle and IsSafe. */
            atomicOr(&q->Entry[j], (unsigned long long)ent::CONSUME_MASK);
            return eidx;
        }

        /* Determine replacement. */
        uint64_t New;
        if (eidx == FQ_BOT || eidx == FQ_BOT_C) {
            /* Dequeue arrived before enqueue — mark with our cycle. */
            New = ent::pack(hcyc, ent::safe(E), FQ_BOT);
        } else {
            /* Clear IsSafe. */
            New = ent::pack(ecyc, false, eidx);
        }

        if (cycle_cmp(ecyc, hcyc) < 0) {
            if (!cas_upd(&q->Entry[j], &E, New))
                goto retry_inner;
        }

        /* Empty check. */
        uint64_t T = aload(&q->Tail);
        if (T <= H + 1) {
            catchup(q, T, H + 1);
            atomicSub(&q->Threshold, 1);
            return FQ_BOT;
        }
        if (atomicSub(&q->Threshold, 1) <= 0)
            return FQ_BOT;
        /* Retry — warp-batched FAA gets new tickets. */
    }
}

/*============================================================================
 * INDIRECTION LAYER — 64-bit payloads via index shuttle
 *============================================================================*/

__device__ bool fq_enqueue_ptr(fq_mpmc_t* q, uint64_t value) {
    uint32_t idx = fq_dequeue(&q->fq);
    if (idx == FQ_BOT) return false;
    q->data[idx] = value;
    __threadfence();
    fq_enqueue(&q->aq, idx);
    return true;
}

__device__ bool fq_dequeue_ptr(fq_mpmc_t* q, uint64_t* out) {
    uint32_t idx = fq_dequeue(&q->aq);
    if (idx == FQ_BOT) return false;
    *out = q->data[idx];
    __threadfence();
    fq_enqueue(&q->fq, idx);
    return true;
}
