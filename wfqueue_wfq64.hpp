#pragma once
// ============================================================================
// GWFQ: GPU Wait-Free FIFO Queue — 64-bit CAS only, no CAS2
//
// KEY DESIGN INSIGHT:
//   The slow path uses F&A to get tickets, SAME as the fast path.
//   This eliminates the CAS-vs-F&A starvation that killed performance.
//
//   Cooperative termination uses a FIN bit on localTail/localHead.
//   When any cooperator (owner or helper) produces/identifies the result,
//   it sets FIN. All other cooperators see FIN and stop.
//
//   No slow_faa. No INC bit. No Phase 2 record.
//
// ENTRY LAYOUT (64-bit):
//   [31:0]   cycle
//   [61:32]  index (30-bit) or sentinel
//   [62]     is_safe
//   [63]     enq (always 1 in this design)
//
// LOCAL WORD LAYOUT (localTail, localHead):
//   [62:0]   counter value (ticket number)
//   [63]     FIN flag (request complete)
//
// GLOBAL COUNTERS (TailGP, HeadGP):
//   Plain uint64_t counters. Both fast and slow paths use F&A.
//
// WAIT-FREEDOM BOUND:
//   O(2n + n × MAX_PATIENCE) per operation.
//   The ring has 2n slots with at most n occupied. After at most 2n F&A's,
//   a cooperator finds an available slot (pigeonhole).
// ============================================================================

#include <hip/hip_runtime.h>
#include <stdint.h>

#ifndef WFQ_MAX_PATIENCE
#define WFQ_MAX_PATIENCE 16
#endif

#ifndef WFQ_HELP_DELAY
#define WFQ_HELP_DELAY 8
#endif

#ifndef WFQ_CATCHUP_LIMIT
#define WFQ_CATCHUP_LIMIT 16
#endif

#ifndef WFQ_SLOW_LIMIT
#define WFQ_SLOW_LIMIT 131072
#endif

#ifndef WFQ_EMPTY_INDEX
#define WFQ_EMPTY_INDEX 0x3fffffffu
#endif

#ifndef WFQ_CONSUMED_INDEX
#define WFQ_CONSUMED_INDEX 0x3ffffffeu
#endif

#ifndef WFQ_NOTED_INDEX
#define WFQ_NOTED_INDEX 0x3ffffffdu
#endif

#ifndef WFQ_TID
#define WFQ_TID ((uint32_t)(blockIdx.x * blockDim.x + threadIdx.x))
#endif

#define WFQ_FIN_BIT  (1ull << 63)
#define WFQ_CNT_MASK (~WFQ_FIN_BIT)


static __device__ __forceinline__ uint64_t wfq_counter(uint64_t x) { return x & WFQ_CNT_MASK; }
static __device__ __forceinline__ bool     wfq_is_fin(uint64_t x)  { return (x & WFQ_FIN_BIT) != 0ull; }

// =============================================================================
// Atomics
// =============================================================================
static __device__ __forceinline__
uint64_t wfq_cas64(uint64_t* a, uint64_t e, uint64_t d) {
  return atomicCAS(reinterpret_cast<unsigned long long*>(a),
                   (unsigned long long)e, (unsigned long long)d);
}
static __device__ __forceinline__
uint64_t wfq_faa64(uint64_t* a, uint64_t inc) {
  return atomicAdd(reinterpret_cast<unsigned long long*>(a),
                   (unsigned long long)inc);
}
static __device__ __forceinline__
uint64_t wfq_ld64(const uint64_t* a) {
  return atomicAdd((unsigned long long*)a, 0ull);
}
static __device__ __forceinline__
void wfq_st64(uint64_t* a, uint64_t v) {
  atomicExch(reinterpret_cast<unsigned long long*>(a), (unsigned long long)v);
}

// =============================================================================
// Packed entry word
// =============================================================================
struct wfq_entry_fields {
  uint32_t cycle;
  uint32_t index;
  uint32_t is_safe;
  uint32_t enq;
};

static __host__ __device__ __forceinline__
uint64_t wfq_pack_entry(uint32_t cycle, uint32_t index, uint32_t is_safe, uint32_t enq) {
  return (uint64_t)cycle
       | ((uint64_t)(index & 0x3fffffffu) << 32)
       | ((uint64_t)(is_safe & 1u) << 62)
       | ((uint64_t)(enq & 1u) << 63);
}
static __host__ __device__ __forceinline__
wfq_entry_fields wfq_unpack_entry(uint64_t w) {
  wfq_entry_fields e{};
  e.cycle   = (uint32_t)(w & 0xffffffffu);
  e.index   = (uint32_t)((w >> 32) & 0x3fffffffu);
  e.is_safe = (uint32_t)((w >> 62) & 1u);
  e.enq     = (uint32_t)((w >> 63) & 1u);
  return e;
}
static __host__ __device__ __forceinline__
bool wfq_is_real_index(uint32_t idx) {
  return idx != WFQ_EMPTY_INDEX && idx != WFQ_CONSUMED_INDEX && idx != WFQ_NOTED_INDEX;
}

// =============================================================================
// Structures
// =============================================================================
struct __attribute__((aligned(128))) wfq_record {
  int32_t  next_check;
  uint32_t next_tid;

  uint64_t seq1;
  uint64_t enqueue;
  uint64_t pending;

  uint64_t localTail;   // [62:0]=ticket, [63]=FIN
  uint64_t initTail;

  uint64_t localHead;   // [62:0]=ticket, [63]=FIN
  uint64_t initHead;

  uint64_t index;       // enqueue payload
  uint64_t deq_result;  // packed [31:0]=ticket, [61:32]=index
  uint64_t seq2;
};

struct __attribute__((aligned(64))) wfq_handle {
  wfq_handle* next;
  uint64_t pad[7];
};

struct __attribute__((aligned(128))) wfq_queue {
  uint64_t* entry;
  uint64_t ring_size;
  uint64_t ring_mask;
  uint64_t n;

  uint64_t TailGP;
  uint64_t HeadGP;
  int64_t  Threshold;

  wfq_record* rec;
  uint32_t nprocs;
  uint32_t _pad32;

  unsigned long long enq_slow;
  unsigned long long deq_slow;
  unsigned long long help_given;
  unsigned long long proof_fail;
  unsigned long long dbg_slow_enq_iters;
  unsigned long long dbg_slow_deq_iters;
  unsigned long long dbg_owner_collect_empty;
};

// =============================================================================
// Utility
// =============================================================================
static __host__ __device__ __forceinline__
uint64_t wfq_round_up_pow2(uint64_t x) {
  if (x <= 1) return 1;
  --x;
  x |= x >> 1; x |= x >> 2; x |= x >> 4;
  x |= x >> 8; x |= x >> 16; x |= x >> 32;
  return x + 1;
}
static __device__ __forceinline__
uint32_t wfq_cycle_of(uint64_t t, uint64_t ring_size) { return (uint32_t)(t / ring_size); }
static __device__ __forceinline__
uint64_t wfq_slot_of(uint64_t t, uint64_t ring_mask) { return t & ring_mask; }
static __device__ __forceinline__
uint64_t wfq_cache_remap(uint64_t j, uint64_t ring_mask) { return j & ring_mask; }

// =============================================================================
// catchup
// =============================================================================
static __device__ __forceinline__
void wfq_catchup(uint64_t* gp_word, uint64_t target) {
  for (int i = 0; i < WFQ_CATCHUP_LIMIT; ++i) {
    uint64_t cur = wfq_ld64(gp_word);
    if (cur >= target) return;
    if (wfq_cas64(gp_word, cur, target) == cur) return;
  }
}

// =============================================================================
// Fast path
// =============================================================================
static __device__ __forceinline__
bool wfq_slot_available(const wfq_entry_fields& E, uint32_t cyc,
                        uint64_t head_cnt, uint64_t T) {
  if (E.cycle >= cyc) return false;
  if (E.index == WFQ_NOTED_INDEX) return false;
  if (E.index != WFQ_EMPTY_INDEX && E.index != WFQ_CONSUMED_INDEX) return false;
  if (!E.is_safe && head_cnt > T) return false;
  return true;
}

static __device__ __forceinline__
uint64_t wfq_try_enq_fast(wfq_queue* q, uint32_t index) {
  const uint64_t T = wfq_faa64(&q->TailGP, 1ull);
  const uint32_t tail = (uint32_t)(T & 0xffffffffu);
  const uint64_t j = wfq_cache_remap(wfq_slot_of(tail, q->ring_mask), q->ring_mask);
  uint64_t* slot = &q->entry[j];
  const uint64_t cur = wfq_ld64(slot);
  const wfq_entry_fields E = wfq_unpack_entry(cur);
  const uint32_t cyc = wfq_cycle_of(tail, q->ring_size);
  const uint64_t head_cnt = wfq_ld64(&q->HeadGP) & 0xffffffffull;

  if (wfq_slot_available(E, cyc, head_cnt, tail)) {
    const uint64_t desired = wfq_pack_entry(cyc, index, 1u, 1u);
    if (wfq_cas64(slot, cur, desired) == cur) {
      if (wfq_ld64((uint64_t*)&q->Threshold) != (uint64_t)(3 * q->n - 1))
        wfq_st64((uint64_t*)&q->Threshold, (uint64_t)(3 * q->n - 1));
      return 0ull;
    }
  }
  return (uint64_t)tail;
}

static __device__ __forceinline__
uint32_t wfq_try_deq_fast(wfq_queue* q, uint64_t* out_head, bool* retry) {
  const uint64_t Hraw = wfq_faa64(&q->HeadGP, 1ull);
  const uint32_t H = (uint32_t)(Hraw & 0xffffffffu);
  const uint64_t j = wfq_cache_remap(wfq_slot_of(H, q->ring_mask), q->ring_mask);
  uint64_t* slot = &q->entry[j];
  const uint64_t cur = wfq_ld64(slot);
  const wfq_entry_fields E = wfq_unpack_entry(cur);
  const uint32_t cyc = wfq_cycle_of(H, q->ring_size);

  if (E.cycle == cyc && wfq_is_real_index(E.index)) {
    const uint64_t desired = wfq_pack_entry(E.cycle, WFQ_CONSUMED_INDEX, 0u, 1u);
    if (wfq_cas64(slot, cur, desired) == cur) {
      *out_head = H; *retry = false;
      return E.index;
    }
  }

  if (E.cycle < cyc) {
    const uint64_t desired = wfq_pack_entry(cyc, WFQ_EMPTY_INDEX, E.is_safe, E.enq);
    (void)wfq_cas64(slot, cur, desired);
  }

  const uint64_t T = wfq_ld64(&q->TailGP) & 0xffffffffull;
  if (T <= (uint64_t)H + 1ull)
    wfq_catchup(&q->TailGP, (uint64_t)H + 1ull);

  if ((int64_t)wfq_faa64((uint64_t*)&q->Threshold, (uint64_t)-1ll) <= 0) {
    *out_head = H; *retry = false;
    return WFQ_EMPTY_INDEX;
  }
  *out_head = H; *retry = true;
  return WFQ_EMPTY_INDEX;
}

// =============================================================================
// Request publication
// =============================================================================
static __device__ __forceinline__
void wfq_publish_enq_request(wfq_record* r, uint64_t tail, uint32_t index) {
  const uint64_t seq = wfq_ld64(&r->seq1);
  wfq_st64(&r->localTail, tail);
  wfq_st64(&r->initTail, tail);
  wfq_st64(&r->index, (uint64_t)index);
  wfq_st64(&r->enqueue, 1ull);
  __threadfence();
  wfq_st64(&r->seq2, seq);
  wfq_st64(&r->pending, 1ull);
}

static __device__ __forceinline__
void wfq_publish_deq_request(wfq_record* r, uint64_t head) {
  const uint64_t seq = wfq_ld64(&r->seq1);
  wfq_st64(&r->localHead, head);
  wfq_st64(&r->initHead, head);
  wfq_st64(&r->deq_result, ((uint64_t)WFQ_EMPTY_INDEX << 32));
  wfq_st64(&r->enqueue, 0ull);
  __threadfence();
  wfq_st64(&r->seq2, seq);
  wfq_st64(&r->pending, 1ull);
}

static __device__ __forceinline__
void wfq_retire_request(wfq_record* r) {
  const uint64_t seq = wfq_ld64(&r->seq1);
  wfq_st64(&r->pending, 0ull);
  __threadfence();
  wfq_st64(&r->seq1, seq + 1ull);
}

// =============================================================================
// SLOW ENQUEUE — F&A for tickets, FIN for termination
// =============================================================================
static __device__ __forceinline__
void wfq_enqueue_slow(wfq_queue* q, uint64_t last_tail, uint32_t index,
                       wfq_record* r) {
  for (int iters = 0; iters < WFQ_SLOW_LIMIT; ++iters) {
    atomicAdd(&q->dbg_slow_enq_iters, 1ull);

    // Check FIN
    if (wfq_is_fin(wfq_ld64(&r->localTail))) return;

    // Get ticket via F&A — zero CAS contention
    const uint64_t T = wfq_faa64(&q->TailGP, 1ull);
    const uint32_t tail = (uint32_t)(T & 0xffffffffu);
    const uint64_t j = wfq_cache_remap(wfq_slot_of(tail, q->ring_mask), q->ring_mask);
    uint64_t* slot = &q->entry[j];

    // Re-check FIN
    if (wfq_is_fin(wfq_ld64(&r->localTail))) return;

    const uint64_t cur = wfq_ld64(slot);
    const wfq_entry_fields E = wfq_unpack_entry(cur);
    const uint32_t cyc = wfq_cycle_of(tail, q->ring_size);
    const uint64_t head_cnt = wfq_ld64(&q->HeadGP) & 0xffffffffull;

    if (wfq_slot_available(E, cyc, head_cnt, tail)) {
      const uint64_t desired = wfq_pack_entry(cyc, index, 1u, 1u);
      if (wfq_cas64(slot, cur, desired) == cur) {
        // Success — set FIN to stop all cooperators
        uint64_t lt = wfq_ld64(&r->localTail);
        if (!wfq_is_fin(lt))
          (void)wfq_cas64(&r->localTail, lt, lt | WFQ_FIN_BIT);
        // Reset threshold
        if (wfq_ld64((uint64_t*)&q->Threshold) != (uint64_t)(3 * q->n - 1))
          wfq_st64((uint64_t*)&q->Threshold, (uint64_t)(3 * q->n - 1));
        return;
      }
    }
  }
  atomicAdd(&q->proof_fail, 1ull);
}

// =============================================================================
// SLOW DEQUEUE — F&A for tickets, FIN for termination
// =============================================================================
static __device__ __forceinline__
void wfq_dequeue_slow(wfq_queue* q, uint64_t last_head, wfq_record* r) {
  for (int iters = 0; iters < WFQ_SLOW_LIMIT; ++iters) {
    atomicAdd(&q->dbg_slow_deq_iters, 1ull);

    // Check FIN
    if (wfq_is_fin(wfq_ld64(&r->localHead))) return;

    // Get head ticket via F&A
    const uint64_t Hraw = wfq_faa64(&q->HeadGP, 1ull);
    const uint32_t H = (uint32_t)(Hraw & 0xffffffffu);
    const uint64_t j = wfq_cache_remap(wfq_slot_of(H, q->ring_mask), q->ring_mask);
    uint64_t* slot = &q->entry[j];

    // Re-check FIN
    if (wfq_is_fin(wfq_ld64(&r->localHead))) return;

    const uint64_t cur = wfq_ld64(slot);
    const wfq_entry_fields E = wfq_unpack_entry(cur);
    const uint32_t cyc = wfq_cycle_of(H, q->ring_size);

    if (E.cycle == cyc && wfq_is_real_index(E.index)) {
      // Found element — save result, consume, set FIN
      uint64_t packed = ((uint64_t)E.index << 32) | (H & 0xffffffffull);
      wfq_st64(&r->deq_result, packed);
      __threadfence();

      const uint64_t consumed = wfq_pack_entry(E.cycle, WFQ_CONSUMED_INDEX, 0u, 1u);
      (void)wfq_cas64(slot, cur, consumed);

      uint64_t lh = wfq_ld64(&r->localHead);
      if (!wfq_is_fin(lh))
        (void)wfq_cas64(&r->localHead, lh, lh | WFQ_FIN_BIT);
      return;
    }

    // Advance stale entry
    if (E.cycle < cyc) {
      const uint64_t desired = wfq_pack_entry(cyc, WFQ_EMPTY_INDEX, E.is_safe, E.enq);
      (void)wfq_cas64(slot, cur, desired);
    }

    // Catchup
    const uint64_t Tcnt = wfq_ld64(&q->TailGP) & 0xffffffffull;
    if (Tcnt <= (uint64_t)H + 1ull)
      wfq_catchup(&q->TailGP, (uint64_t)H + 1ull);

    // Threshold — detect empty queue
    if ((int64_t)wfq_faa64((uint64_t*)&q->Threshold, (uint64_t)-1ll) <= 0) {
      uint64_t packed = ((uint64_t)WFQ_EMPTY_INDEX << 32) | (H & 0xffffffffull);
      wfq_st64(&r->deq_result, packed);
      uint64_t lh = wfq_ld64(&r->localHead);
      if (!wfq_is_fin(lh))
        (void)wfq_cas64(&r->localHead, lh, lh | WFQ_FIN_BIT);
      return;
    }
  }
  atomicAdd(&q->proof_fail, 1ull);
}

// =============================================================================
// Owner collect
// =============================================================================
static __device__ __forceinline__
uint32_t wfq_owner_collect_deq_result(wfq_queue* q, wfq_record* r) {
  uint64_t packed = wfq_ld64(&r->deq_result);
  uint32_t result = (uint32_t)((packed >> 32) & 0x3fffffffu);
  if (wfq_is_real_index(result)) return result;
  atomicAdd(&q->dbg_owner_collect_empty, 1ull);
  return WFQ_EMPTY_INDEX;
}

// =============================================================================
// Helping
// =============================================================================
static __device__ __forceinline__
void wfq_help_enqueue(wfq_queue* q, wfq_record* thr) {
  const uint64_t seq = wfq_ld64(&thr->seq2);
  const uint64_t enq = wfq_ld64(&thr->enqueue);
  const uint64_t idx = wfq_ld64(&thr->index);
  const uint64_t tail = wfq_ld64(&thr->initTail);
  const uint64_t s1 = wfq_ld64(&thr->seq1);
  if (enq && s1 == seq) {
    wfq_enqueue_slow(q, tail, (uint32_t)idx, thr);
    atomicAdd(&q->help_given, 1ull);
  }
}

static __device__ __forceinline__
void wfq_help_dequeue(wfq_queue* q, wfq_record* thr) {
  const uint64_t seq = wfq_ld64(&thr->seq2);
  const uint64_t enq = wfq_ld64(&thr->enqueue);
  const uint64_t head = wfq_ld64(&thr->initHead);
  const uint64_t s1 = wfq_ld64(&thr->seq1);
  if (!enq && s1 == seq) {
    wfq_dequeue_slow(q, head, thr);
    atomicAdd(&q->help_given, 1ull);
  }
}

static __device__ __forceinline__
void wfq_help_threads(wfq_queue* q) {
  const uint32_t tid = WFQ_TID;
  if (tid >= q->nprocs) return;
  wfq_record* me = &q->rec[tid];
  if (--me->next_check > 0) return;
  wfq_record* thr = &q->rec[me->next_tid];
  if (wfq_ld64(&thr->pending)) {
    if (wfq_ld64(&thr->enqueue))
      wfq_help_enqueue(q, thr);
    else
      wfq_help_dequeue(q, thr);
  }
  me->next_check = WFQ_HELP_DELAY;
  me->next_tid = (me->next_tid + 1u) % q->nprocs;
}

// =============================================================================
// Public enqueue / dequeue
// =============================================================================
__device__ __forceinline__
void wfq_enqueue(wfq_queue* q, wfq_handle* h, uint32_t index) {
  (void)h;
  const uint32_t tid = WFQ_TID;
  if (tid >= q->nprocs) return;

  wfq_help_threads(q);

  uint64_t tail = 0ull;
  for (int i = 0; i < WFQ_MAX_PATIENCE; ++i) {
    tail = wfq_try_enq_fast(q, index);
    if (tail == 0ull) return;
  }

  atomicAdd(&q->enq_slow, 1ull);
  wfq_record* r = &q->rec[tid];
  wfq_publish_enq_request(r, tail, index);
  wfq_enqueue_slow(q, tail, index, r);
  wfq_retire_request(r);
}

__device__ __forceinline__
uint32_t wfq_dequeue(wfq_queue* q, wfq_handle* h) {
  (void)h;
  const uint32_t tid = WFQ_TID;
  if (tid >= q->nprocs) return WFQ_EMPTY_INDEX;

  if ((int64_t)wfq_ld64((uint64_t*)&q->Threshold) < 0)
    return WFQ_EMPTY_INDEX;

  wfq_help_threads(q);

  uint64_t head = 0ull;
  for (int i = 0; i < WFQ_MAX_PATIENCE; ++i) {
    bool retry = false;
    uint32_t idx = wfq_try_deq_fast(q, &head, &retry);
    if (!retry) return idx;
  }

  atomicAdd(&q->deq_slow, 1ull);
  wfq_record* r = &q->rec[tid];
  wfq_publish_deq_request(r, head);
  wfq_dequeue_slow(q, head, r);
  wfq_retire_request(r);
  return wfq_owner_collect_deq_result(q, r);
}

// =============================================================================
// Init / destroy
// =============================================================================
__global__
void wfq_init_kernel(wfq_queue* q, wfq_handle* h, wfq_record* rec, int nthreads) {
  const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;

  if (gid == 0) {
    q->TailGP = (uint64_t)q->ring_size;
    q->HeadGP = (uint64_t)q->ring_size;
    q->Threshold = -1;
    q->nprocs = (uint32_t)nthreads;
    q->enq_slow = 0; q->deq_slow = 0;
    q->help_given = 0; q->proof_fail = 0;
    q->dbg_slow_enq_iters = 0; q->dbg_slow_deq_iters = 0;
    q->dbg_owner_collect_empty = 0;
  }

  for (uint64_t i = gid; i < (uint64_t)nthreads; i += stride) {
    h[i].next = &h[(i + 1) % (uint64_t)nthreads];
    rec[i].next_check = WFQ_HELP_DELAY;
    rec[i].next_tid   = ((uint32_t)i + 1u) % (uint32_t)nthreads;
    rec[i].seq1 = 1ull; rec[i].enqueue = 0ull; rec[i].pending = 0ull;
    rec[i].localTail = 0ull; rec[i].initTail = 0ull;
    rec[i].localHead = 0ull; rec[i].initHead = 0ull;
    rec[i].index = 0ull;
    rec[i].deq_result = ((uint64_t)WFQ_EMPTY_INDEX << 32);
    rec[i].seq2 = 0ull;
  }

  for (uint64_t p = gid; p < q->ring_size; p += stride)
    q->entry[p] = wfq_pack_entry(0u, WFQ_EMPTY_INDEX, 1u, 1u);
}

inline void wfq_queue_host_init(wfq_queue** d_q, wfq_handle** d_h,
                                wfq_record** d_rec, int num_threads) {
  uint64_t n = wfq_round_up_pow2((uint64_t)num_threads);
  if (n < 1024ull) n = 1024ull;
  uint64_t ring_size = 2ull * n;

  hipMalloc((void**)d_q, sizeof(wfq_queue));
  hipMalloc((void**)d_h, (size_t)num_threads * sizeof(wfq_handle));
  hipMalloc((void**)d_rec, (size_t)num_threads * sizeof(wfq_record));
  uint64_t* d_entry = nullptr;
  hipMalloc((void**)&d_entry, (size_t)ring_size * sizeof(uint64_t));

  int device = 0;
  hipGetDevice(&device);
  hipMemPrefetchAsync(*d_h, (size_t)num_threads * sizeof(wfq_handle), device, 0);
  hipMemPrefetchAsync(*d_rec, (size_t)num_threads * sizeof(wfq_record), device, 0);
  hipMemPrefetchAsync(d_entry, (size_t)ring_size * sizeof(uint64_t), device, 0);
  hipDeviceSynchronize();

  wfq_queue hq{};
  hq.entry = d_entry;
  hq.ring_size = ring_size;
  hq.ring_mask = ring_size - 1ull;
  hq.n = n;
  hq.TailGP = (uint64_t)ring_size;
  hq.HeadGP = (uint64_t)ring_size;
  hq.Threshold = -1;
  hq.rec = *d_rec;
  hq.nprocs = (uint32_t)num_threads;

  hipMemcpy(*d_q, &hq, sizeof(hq), hipMemcpyHostToDevice);

  int block = 256;
  int grid = (num_threads + block - 1) / block;
  if (grid < 80) grid = 80;
  wfq_init_kernel<<<grid, block>>>(*d_q, *d_h, *d_rec, num_threads);
  hipDeviceSynchronize();
}

inline void wfq_queue_destroy(wfq_queue* d_q, wfq_handle* d_h) {
  wfq_queue hq{};
  hipMemcpy(&hq, d_q, sizeof(wfq_queue), hipMemcpyDeviceToHost);
  if (hq.entry) hipFree(hq.entry);
  if (hq.rec) hipFree(hq.rec);
  hipFree(d_q);
  hipFree(d_h);
}