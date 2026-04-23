/**
 * wf-GPU Implementation
 * 
 * Include this AFTER wf_gpu.h.  All functions are templates so they
 * must live in a header visible at compile time.
 *
 * CONVENTIONS:
 *   - "raw ticket" = the logical sequence number (0, 1, 2, ...)
 *   - "local format" = (raw_ticket << 2) | INC | FIN  [used in localTail/Head and *v]
 *   - "global word" = (raw_ticket << THRIDX_BITS) | thridx  [used in q->Tail/Head]
 *   - Conversions: raw_to_local(r) = r << 2
 *                  local_to_raw(l) = l >> 2
 *                  global_to_raw(g) = Config::unpack_counter(g)
 *
 * MEMORY ORDERING:
 *   atomicCAS/atomicAdd/atomicOr provide seq_cst on NVIDIA/AMD GPUs.
 *   __threadfence() is used for publish/acquire patterns on non-atomic fields.
 */

#pragma once
#include "gwf_queue.hpp"

// #define GWF_PROFILE 1


#ifdef GWF_PROFILE
extern __device__ unsigned long long g_prof_finalize_calls;
extern __device__ unsigned long long g_prof_finalize_scan_iters;
extern __device__ unsigned long long g_prof_slow_enq_entries;
extern __device__ unsigned long long g_prof_slow_deq_entries;
extern __device__ unsigned long long g_prof_slowfaa_loops;
extern __device__ unsigned long long g_prof_slowfaa_gp_nonnull;
#define GWF_PROF_ADD(sym, x) atomicAdd(&(sym), (unsigned long long)(x))
#else
#define GWF_PROF_ADD(sym, x) ((void)0)
#endif

/*============================================================================
 * HELPERS
 *============================================================================*/

// slot idx helper with turning on Cache Remaping or not
__device__ __forceinline__
uint32_t slot_idx(uint32_t pos, uint32_t num_slots) {
#ifdef CACHE_REMAP
    return cache_remap(pos, num_slots);
#else
    return pos & (num_slots - 1);
#endif    
}

/** Global thread ID across all blocks. */
#ifdef WF_TID_EXPR
__device__ __forceinline__
uint32_t wf_tid() {
    return (uint32_t)(WF_TID_EXPR);
}
#else
__device__ __forceinline__
uint32_t wf_tid() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}
#endif
// __device__ __forceinline__
// uint32_t wf_tid() {
//     return blockIdx.x * blockDim.x + threadIdx.x;
// }

/** Volatile load — prevents compiler from caching the read. */
__device__ __forceinline__
uint64_t vload(unsigned long long* p) {
    // return (uint64_t)atomicAdd(p, 0ULL);  /* atomicAdd(x,0) = atomic load */
    // __ATOMIC_RELAXED compiles to a standard load that skips register 
    // caching but freely utilizes the massive read bandwidth of L1/L2 cache.
    return (uint64_t)__atomic_load_n((const unsigned long long*)p, __ATOMIC_RELAXED);
}

__device__ __forceinline__
int32_t vload32(int32_t* p) {
    // return atomicAdd(p, 0);
    return (int32_t)__atomic_load_n((const int*)p, __ATOMIC_RELAXED);
}

/** CAS returning bool (true = success). */
__device__ __forceinline__
bool cas64(unsigned long long* ptr, uint64_t expected, uint64_t desired) {
    return atomicCAS(ptr, (unsigned long long)expected,
                     (unsigned long long)desired) == (unsigned long long)expected;
}

/** CAS that updates *expected on failure (like C11 atomic_compare_exchange). */
__device__ __forceinline__
bool cas64_update(unsigned long long* ptr, uint64_t* expected, uint64_t desired) {
    uint64_t old = (uint64_t)atomicCAS(ptr, (unsigned long long)*expected,
                                       (unsigned long long)desired);
    bool ok = (old == *expected);
    *expected = old;
    return ok;
}

/** Convert raw ticket to local format. */
__device__ __forceinline__
uint64_t raw_to_local(uint64_t raw) { return raw << 2; }

/** Extract raw ticket from local format (strips INC/FIN). */
__device__ __forceinline__
uint64_t local_to_raw(uint64_t local_val) { return local_val >> 2; }


__device__ __forceinline__
uint32_t aload_u32(const unsigned int* p) {
    // return atomicAdd((unsigned int*)p, 0u);
    return (uint32_t)__atomic_load_n(p, __ATOMIC_RELAXED);
}

__device__ __forceinline__
void astore_u32(unsigned int* p, uint32_t v) {
    // atomicExch(p, v);
    __atomic_store_n(p, (unsigned int)v, __ATOMIC_RELAXED);
}

__device__ __forceinline__
uint64_t aload_u64(const unsigned long long* p){
    // return (uint64_t)atomicAdd((unsigned long long*)p, 0ULL);
    return (uint64_t)__atomic_load_n(p, __ATOMIC_RELAXED);
}

__device__ __forceinline__
void astore_u64(unsigned long long* p, uint64_t v){
    // atomicExch(p,(unsigned long long)v);
    __atomic_store_n(p, (unsigned long long)v, __ATOMIC_RELAXED);
}

/*----------------------------------------------------------------------------
+ * FAST-PATH BATCHED FAA (safe only for fast path)
+ *
+ * This reserves a consecutive block of logical counters while preserving the
+ * low ThrIdx bits by adding a multiple of Config::COUNTER_INC.
+ *
+ * DO NOT use this in slow_faa(): the slow path proof relies on single-request
+ * ownership through ThrIdx and one successful cooperative increment at a time.
+ *----------------------------------------------------------------------------*/

#ifndef GWF_FASTPATH_BATCH
#define GWF_FASTPATH_BATCH 1
#endif

#ifdef __HIP_PLATFORM_AMD__
using gwf_wave_mask_t = uint64_t;

__device__ __forceinline__ gwf_wave_mask_t gwf_wave_activemask() {
    return (gwf_wave_mask_t)__ballot(1);
}

__device__ __forceinline__ uint32_t gwf_wave_popc(gwf_wave_mask_t m) {
    return (uint32_t)__popcll((unsigned long long)m);
}

__device__ __forceinline__ uint32_t gwf_wave_ffs(gwf_wave_mask_t m) {
    return (uint32_t)__ffsll((unsigned long long)m);
}

__device__ __forceinline__ uint32_t gwf_wave_lane_id() {
    return (uint32_t)(threadIdx.x & 63);
}

__device__ __forceinline__ uint32_t gwf_wave_shfl_u32(uint32_t v, int src_lane) {
    return __shfl(v, src_lane);
}
#else
using gwf_wave_mask_t = uint32_t;

__device__ __forceinline__ gwf_wave_activemask() {
    return (gwf_wave_mask_t)__activemask();
}

__device__ __forceinline__ uint32_t gwf_wave_popc(gwf_wave_mask_t m) {
    return (uint32_t)__popc((unsigned int)m);
}

__device__ __forceinline__ uint32_t gwf_wave_ffs(gwf_wave_mask_t m) {
    return (uint32_t)__ffs((unsigned int)m);
}

__device__ __forceinline__ uint32_t gwf_wave_lane_id() {
    return (uint32_t)(threadIdx.x & 31);
}

__device__ __forceinline__ uint32_t gwf_wave_shfl_u32(uint32_t v, int src_lane) {
    return __shfl_sync(0xFFFFFFFFu, v, src_lane);
}
#endif

__device__ __forceinline__ uint64_t gwf_wave_shfl_u64(uint64_t v, int src_lane) {
    uint32_t lo = (uint32_t)v;
    uint32_t hi = (uint32_t)(v >> 32);
    lo = gwf_wave_shfl_u32(lo, src_lane);
    hi = gwf_wave_shfl_u32(hi, src_lane);
    return ((uint64_t)hi << 32) | lo;
}

template <typename Config>
__device__ __forceinline__ uint64_t gwf_fastpath_batched_faa(unsigned long long* globalp) {
#if GWF_FASTPATH_BATCH
    gwf_wave_mask_t mask = gwf_wave_activemask();
    uint32_t lane = gwf_wave_lane_id();
    uint32_t leader = gwf_wave_ffs(mask) - 1u;
    uint32_t count = gwf_wave_popc(mask);
    gwf_wave_mask_t lower = (lane == 0) ? (gwf_wave_mask_t)0
                                        : (mask & ((((gwf_wave_mask_t)1) << lane) - 1));
    uint32_t rank = gwf_wave_popc(lower);

    uint64_t old_global = 0;
    if (lane == leader) {
        old_global = (uint64_t)atomicAdd(globalp,
            (unsigned long long)((uint64_t)count * (uint64_t)Config::COUNTER_INC));
    }
    old_global = gwf_wave_shfl_u64(old_global, leader);
    return old_global + (uint64_t)rank * (uint64_t)Config::COUNTER_INC;
#else
    return (uint64_t)atomicAdd(globalp, (unsigned long long)Config::COUNTER_INC);
#endif
}

/*============================================================================
 * INITIALIZATION
 *============================================================================*/

template <typename Config, uint32_t N_PARAM>
__device__ void wf_init(wf_t<Config, N_PARAM>* q) {
    using Q = wf_t<Config, N_PARAM>;
    /* Head and Tail start at SLOTS (= 2n), same as original SCQ.
     * This means the first cycle accessed is cycle 1, not 0.
     * Entries are initialized with Cycle=0, so the first threads see
     * their cycle (1) > entry cycle (0) and can proceed. */
    uint64_t init_counter = Q::SLOTS;
    q->Tail = (unsigned long long)Config::pack_global(init_counter, (uint32_t)Config::NULL_TID);
    q->Head = (unsigned long long)Config::pack_global(init_counter, (uint32_t)Config::NULL_TID);
    q->Threshold = -1;  /* Empty */

    for (uint32_t i = 0; i < Q::SLOTS; i++)
        q->Entry[i] = (unsigned long long)wf_entry::INIT;

    for (uint32_t i = 0; i < Config::MAX_THREADS; i++) {
        thrdrec_t* r = &q->Record[i];
        r->nextCheck = WF_HELP_DELAY;
        r->nextTid = i;
        // r->phase2.seq1 = 1;
        // r->phase2.local = nullptr;
        // r->phase2.cnt = 0;
        // r->phase2.seq2 = 0;
        // r->seq1 = 1;
        // r->enqueue = 0;
        // r->pending = 0;
        // r->localTail = 0;
        // r->initTail = 0;
        // r->localHead = 0;
        // r->initHead = 0;
        // r->index = 0;
        // r->seq2 = 0;
        astore_u32(&r->phase2.seq1, 1);
        astore_u64(&r->phase2.local, 0);
        astore_u64(&r->phase2.cnt, 0);
        astore_u32(&r->phase2.seq2, 0);

        astore_u32(&r->seq1, 1);
        astore_u32(&r->enqueue, 0);
        astore_u32(&r->pending, 0);
        astore_u64(&r->localTail, 0);
        astore_u64(&r->initTail, 0);
        astore_u64(&r->localHead, 0);
        astore_u64(&r->initHead, 0);
        astore_u32(&r->index, 0);
        astore_u32(&r->seq2, 0);
    }
}

template <typename Config, uint32_t N_PARAM>
__device__ void wf_mpmc_init(wf_mpmc_t<Config, N_PARAM>* q) {
    using Q = wf_t<Config, N_PARAM>;
    /* Initialize aq as empty. */
    wf_init(&q->aq);
    /* Initialize fq as full (pre-loaded with indices [0, N-1]). */
    wf_init(&q->fq);
    /* Overwrite fq entries to contain indices. */
    uint64_t init_counter = Q::SLOTS;
    for (uint32_t i = 0; i < N_PARAM; i++) {
        uint64_t ticket = init_counter + i;
        // uint32_t j = cache_remap(slot_of(ticket, Q::SLOTS), Q::SLOTS);
        uint32_t j = slot_idx(ticket, Q::SLOTS);
        uint32_t cyc = cycle_of(ticket, Q::SLOTS);
        q->fq.Entry[j] = (unsigned long long)wf_entry::pack(0xFF, cyc, true, true, i);
    }
    /* fq.Tail advances past the pre-loaded entries. */
    q->fq.Tail = (unsigned long long)Config::pack_global(init_counter + N_PARAM,
                                                          (uint32_t)Config::NULL_TID);
    q->fq.Threshold = Q::THRESHOLD_MAX;
}

/*============================================================================
 * CONSUME, FINALIZE_REQUEST, CATCHUP  (Figure 5 lines 1-17)
 *============================================================================*/

/**
 * finalize_request: Set FIN on the slow-path enqueuer whose localTail
 * matches the head ticket h_raw. Scans thread records.
 * Paper Figure 5, lines 4-11.
 */
template <typename Config, uint32_t N_PARAM>
__device__ void finalize_request(wf_t<Config, N_PARAM>* q, uint64_t h_raw) {
    #if defined(GWF_PROFILE)
    GWF_PROF_ADD(g_prof_finalize_calls, 1);
    #endif
    uint32_t my_tid = wf_tid();
    uint64_t h_local = raw_to_local(h_raw);
    uint32_t i = (my_tid + 1) % Config::MAX_THREADS;
    while (i != my_tid) {
        uint64_t ltail = vload(&q->Record[i].localTail);
        #if defined(GWF_PROFILE)
        GWF_PROF_ADD(g_prof_finalize_scan_iters, 1);
        #endif
        if (wf_local_ptr::counter(ltail) == h_local) {
            cas64(&q->Record[i].localTail, h_local, h_local | wf_local_ptr::FIN_BIT);
            return;
        }
        i = (i + 1) % Config::MAX_THREADS;
    }
}

/**
 * consume: Mark entry as consumed via atomicOr. If entry was produced by
 * slow path (Enq=0), help finalize the enqueuer's request first.
 * Paper Figure 5, lines 1-3.
 */
template <typename Config, uint32_t N_PARAM>
__device__ void consume(wf_t<Config, N_PARAM>* q,
                        uint64_t h_raw, uint32_t j, uint64_t ent) {
    if (!wf_entry::enq(ent))
        finalize_request(q, h_raw);
    atomicOr(&q->Entry[j], (unsigned long long)wf_entry::CONSUME_MASK);
}

/**
 * catchup: Advance Tail to at least Head when Tail has fallen behind.
 * Bounded iterations for wait-freedom.
 * Paper Figure 3, lines 13-17.
 */
template <typename Config, uint32_t N_PARAM>
__device__ void catchup(wf_t<Config, N_PARAM>* q,
                        uint64_t tail_raw, uint64_t head_raw) {
    constexpr int CATCHUP_LIMIT = 32;
    for (int i = 0; i < CATCHUP_LIMIT; i++) {
        /* CAS Tail from tail_raw to head_raw, preserving ThrIdx. */
        uint64_t cur = vload(&q->Tail);
        uint64_t cur_cnt = Config::unpack_counter(cur);
        uint32_t cur_tid = Config::unpack_thridx(cur);
        if (cur_cnt >= head_raw) break;
        if (cur_cnt != tail_raw) {
            /* Counter changed, re-read. */
            tail_raw = cur_cnt;
            head_raw = Config::unpack_counter(vload(&q->Head));
            if (tail_raw >= head_raw) break;
            continue;
        }
        uint64_t desired = Config::pack_global(head_raw, cur_tid);
        if (cas64(&q->Tail, cur, desired)) break;
        head_raw = Config::unpack_counter(vload(&q->Head));
        tail_raw = Config::unpack_counter(vload(&q->Tail));
        if (tail_raw >= head_raw) break;
    }
}

/*============================================================================
 * FAST PATH  (Figure 3)
 *============================================================================*/

/**
 * try_enq: Attempt one enqueue on the fast path.
 * Returns 0 on success, otherwise the raw ticket that was tried.
 * Paper Figure 3, lines 18-29.
 */
template <typename Config, uint32_t N_PARAM>
__device__ uint64_t try_enq(wf_t<Config, N_PARAM>* q, uint32_t index) {
    using Q = wf_t<Config, N_PARAM>;
    /* F&A on Tail. */
    // uint64_t old_global = (uint64_t)atomicAdd(&q->Tail, (unsigned long long)Config::COUNTER_INC);
    uint64_t old_global = gwf_fastpath_batched_faa<Config>(&q->Tail);
    uint64_t T = Config::unpack_counter(old_global);
    uint32_t tcyc = cycle_of(T, Q::SLOTS);
    // uint32_t j = cache_remap(slot_of(T, Q::SLOTS), Q::SLOTS);
    uint32_t j = slot_idx(T, Q::SLOTS);


    uint64_t E = vload(&q->Entry[j]);
retry:
    uint32_t ecyc = wf_entry::cycle(E);
    uint32_t eidx = wf_entry::index(E);
    /* E.Cycle < Cycle(T) */
    if (cycle_cmp(ecyc, tcyc) < 0
        && (wf_entry::issafe(E)
            || Config::unpack_counter(vload(&q->Head)) <= T)
        && (eidx == wf_entry::BOT || eidx == wf_entry::BOT_C))
    {
        /* Build new entry: preserve Note, set Cycle=tcyc, IsSafe=1, Enq=1 */
        uint64_t New = wf_entry::pack(wf_entry::note(E), tcyc, true, true, index);
        if (!cas64_update(&q->Entry[j], &E, New))
            goto retry;
        /* Success — reset threshold. */
        if (vload32(&q->Threshold) != Q::THRESHOLD_MAX)
            atomicExch(&q->Threshold, Q::THRESHOLD_MAX);
        return 0;  /* OK */
    }
    return T;  /* Try again */
}

/**
 * try_deq: Attempt one dequeue on the fast path.
 * Returns 0 on success (index written to *out), otherwise the raw ticket tried.
 * Paper Figure 3, lines 30-52.
 */
template <typename Config, uint32_t N_PARAM>
__device__ uint64_t try_deq(wf_t<Config, N_PARAM>* q, uint32_t* out) {
    using Q = wf_t<Config, N_PARAM>;
    /* F&A on Head. */
    // uint64_t old_global = (uint64_t)atomicAdd(&q->Head, (unsigned long long)Config::COUNTER_INC);
    uint64_t old_global = gwf_fastpath_batched_faa<Config>(&q->Head);
    uint64_t H = Config::unpack_counter(old_global);
    uint32_t hcyc = cycle_of(H, Q::SLOTS);
    // uint32_t j = cache_remap(slot_of(H, Q::SLOTS), Q::SLOTS);
    uint32_t j = slot_idx(H, Q::SLOTS);

    uint64_t E = vload(&q->Entry[j]);
retry_inner:;
    uint32_t ecyc = wf_entry::cycle(E);
    uint32_t eidx = wf_entry::index(E);

    /* Case 1: Entry is ready — consume it. */
    if (cycle_cmp(ecyc, hcyc) == 0 && eidx != wf_entry::BOT && eidx != wf_entry::BOT_C) {
        consume(q, H, j, E);
        *out = eidx;
        return 0;  /* OK */
    }

    /* Build replacement entry depending on what we found. */
    uint64_t New;
    if (eidx == wf_entry::BOT || eidx == wf_entry::BOT_C) {
        /* Dequeue arrived before enqueue — place ⊥ with our cycle. */
        New = wf_entry::pack(wf_entry::note(E), hcyc, wf_entry::issafe(E), true, wf_entry::BOT);
    } else {
        /* Clear IsSafe. */
        New = wf_entry::pack(wf_entry::note(E), ecyc, false, wf_entry::enq(E), eidx);
    }

    if (cycle_cmp(ecyc, hcyc) < 0) {
        if (!cas64_update(&q->Entry[j], &E, New))
            goto retry_inner;
    }

    /* Check if queue is empty. */
    uint64_t T = Config::unpack_counter(vload(&q->Tail));
    if (T <= H + 1) {
        catchup(q, T, H + 1);
        atomicSub(&q->Threshold, 1);
        *out = wf_entry::BOT;
        return 0;  /* Empty */
    }
    if (atomicSub(&q->Threshold, 1) <= 0) {
        *out = wf_entry::BOT;
        return 0;  /* Empty */
    }
    return H;  /* Try again */
}

/*============================================================================
 * SLOW PATH HELPERS  (Figure 7 lines 21-42, 77-88)
 *============================================================================*/

/**
 * prepare_phase2: Publish a Phase 2 help request.
 * Paper Figure 7, lines 38-42.
 *
 * @param phase2  This thread's phase2 record
 * @param local   Pointer to localTail or localHead
 * @param cnt_local  Counter value in LOCAL format (raw << 2)
 */
__device__ void prepare_phase2(phase2rec_t* phase2,
                               unsigned long long* local,
                               uint64_t cnt_local) {
    // uint32_t seq = phase2->seq1 + 1;
    // phase2->seq1 = seq;
    // __threadfence();       /* seq1 visible before fields */
    // phase2->local = local;
    // phase2->cnt = cnt_local;
    // __threadfence();       /* fields visible before seq2 */
    // phase2->seq2 = seq;
    uint32_t seq = aload_u32(&phase2->seq1) + 1;
    astore_u32(&phase2->seq1, seq);

    __threadfence();  // start publish of payload
    astore_u64(&phase2->local, (uint64_t)(uintptr_t)local);
    astore_u64(&phase2->cnt,   cnt_local);
    __threadfence();  // payload before seq2
    astore_u32(&phase2->seq2, seq);
}

/* load_global_help_phase2 is inlined into slow_faa (which has access to
 * q->Record[] needed to look up phase2 by ThrIdx). See slow_faa below. */

/**
 * slow_faa: Cooperative fetch-and-add.
 * Increments the global counter exactly once across all cooperative threads.
 * Uses a two-phase protocol: Phase 1 sets INC on local, Phase 2 clears it.
 *
 * Paper Figure 7, lines 21-37.
 *
 * @param q       Queue (needed to access Record[] for phase2 helping)
 * @param globalp Pointer to q->Tail or q->Head
 * @param local   Pointer to r->localTail or r->localHead
 * @param v       In/out: current counter in LOCAL FORMAT (raw<<2 | flags)
 * @param thld    Pointer to q->Threshold if dequeueing, else nullptr
 * @return true if incremented (continue), false if FIN detected (stop)
 */
template <typename Config, uint32_t N_PARAM>
__device__ bool slow_faa(wf_t<Config, N_PARAM>* q,
                         unsigned long long* globalp,
                         unsigned long long* local,
                         uint64_t* v,
                         int32_t* thld) {
    uint32_t my_tid = wf_tid();
    phase2rec_t* my_phase2 = &q->Record[my_tid].phase2;
    uint64_t cnt_local = 0; /* Counter in local format (raw << 2) */
    uint64_t cnt_raw = 0;   /* Raw counter for global operations */
    bool global_cas_ok = false;

    #if defined(GWF_PROFILE)
    GWF_PROF_ADD(g_prof_slowfaa_loops, 1);
    #endif
    do {
        /* ── Check FIN ── */
        uint64_t local_cur = vload(local);
        if (local_cur & wf_local_ptr::FIN_BIT) {
            *v = local_cur;
            return false;
        }

        /* ── Load global & help pending Phase 2 (Paper Fig 7 lines 77-88) ── */
        uint64_t gp = vload(globalp);
        uint32_t gp_tid = Config::unpack_thridx(gp);

        if (gp_tid != (uint32_t)Config::NULL_TID) {
            #if defined(GWF_PROFILE)
            GWF_PROF_ADD(g_prof_slowfaa_gp_nonnull, 1);
            #endif
            /* A slow-path thread has a Phase 2 request in the global word.
             * Help it by clearing INC on its local value, then clear the
             * ThrIdx from the global word. */
            thrdrec_t* helper_rec = &q->Record[gp_tid];
            // phase2rec_t* p2 = &helper_rec->phase2;
            // uint32_t seq = p2->seq2;
            // __threadfence();
            // unsigned long long* p2_local = p2->local;
            // uint64_t p2_cnt = p2->cnt;
            // __threadfence();
            // if (p2->seq1 == seq) {
            //     cas64(p2_local, p2_cnt | wf_local_ptr::INC_BIT, p2_cnt);
            // }
            phase2rec_t* p2 = &helper_rec->phase2;

            uint32_t seq = aload_u32(&p2->seq2);
            __threadfence();  // acquire payload after seq2

            unsigned long long* p2_local =
                (unsigned long long*)(uintptr_t)aload_u64(&p2->local);
            uint64_t p2_cnt = aload_u64(&p2->cnt);

            __threadfence();
            uint32_t seq1 = aload_u32(&p2->seq1);

            if (seq1 == seq) {
                cas64(p2_local, p2_cnt | wf_local_ptr::INC_BIT, p2_cnt);
            }
            /* Clear ThrIdx from global. CAS may fail if another thread
             * already cleared it or the counter advanced — that's fine,
             * we just loop and re-read. */
            uint64_t gp_clean = Config::pack_global(
                Config::unpack_counter(gp), (uint32_t)Config::NULL_TID);
            if (!cas64(globalp, gp, gp_clean))
                continue;
            gp = gp_clean;
        }

        cnt_raw = Config::unpack_counter(gp);
        cnt_local = raw_to_local(cnt_raw);

        /* ── Phase 1: CAS local from *v to cnt_local|INC (Paper line 25) ── */
        uint64_t v_copy = *v;
        if (!cas64_update(local, &v_copy, cnt_local | wf_local_ptr::INC_BIT)) {
            *v = v_copy;
            if (v_copy & wf_local_ptr::FIN_BIT) return false;
            if (!(v_copy & wf_local_ptr::INC_BIT)) return true;
            /* INC already set by another helper — extract their cnt. */
            cnt_local = wf_local_ptr::counter(v_copy);
            cnt_raw = cnt_local >> 2;
        } else {
            *v = cnt_local | wf_local_ptr::INC_BIT;
        }

        /* ── Prepare Phase 2 request (Paper line 31) ── */
        prepare_phase2(my_phase2, local, cnt_local);

        /* ── CAS global: {cnt, NULL_TID} → {cnt+1, my_tid} (Paper line 32) ──
         * This is the actual increment. If it fails, the outer loop retries
         * from the top (re-reading global, re-helping phase2, etc.).
         * All cooperative threads eventually converge, so at least one
         * succeeds — bounding this loop (Lemma 5.5). */
        global_cas_ok = cas64(
            globalp,
            Config::pack_global(cnt_raw, (uint32_t)Config::NULL_TID),
            Config::pack_global(cnt_raw + 1, my_tid));

    } while (!global_cas_ok);

    /* ── Success: global counter incremented ── */

    /* Decrement threshold if dequeueing (Paper line 33). */
    if (thld)
        atomicSub(thld, 1);

    /* Phase 2: Clear INC on local (Paper line 34). */
    cas64(local, cnt_local | wf_local_ptr::INC_BIT, cnt_local);

    /* Clear our phase2 ThrIdx from the global word (Paper line 35). */
    cas64(globalp,
          Config::pack_global(cnt_raw + 1, my_tid),
          Config::pack_global(cnt_raw + 1, (uint32_t)Config::NULL_TID));

    *v = cnt_local;  /* Clean counter in local format. */
    return true;
}

/*============================================================================
 * SLOW PATH ENTRY OPERATIONS  (Figure 7 lines 1-20, 43-69)
 *============================================================================*/

/**
 * try_enq_slow: One slow-path enqueue attempt at raw ticket T_raw.
 * Paper Figure 7, lines 1-20.
 * Returns true on success or if already inserted by another helper.
 */
template <typename Config, uint32_t N_PARAM>
__device__ bool try_enq_slow(wf_t<Config, N_PARAM>* q, uint64_t T_raw,
                             uint32_t index, thrdrec_t* r) {
    using Q = wf_t<Config, N_PARAM>;
    uint32_t tcyc = cycle_of(T_raw, Q::SLOTS);
    // uint32_t j = cache_remap(slot_of(T_raw, Q::SLOTS), Q::SLOTS);
    uint32_t j = slot_idx(T_raw, Q::SLOTS);

    uint64_t Pair = vload(&q->Entry[j]);
retry:;
    uint32_t ecyc  = wf_entry::cycle(Pair);
    uint32_t enote = wf_entry::note(Pair);
    uint32_t eidx  = wf_entry::index(Pair);

    if (cycle_cmp(ecyc, tcyc) < 0 && cycle_cmp(enote, tcyc) < 0) {
        /* Entry is from an older cycle and Note hasn't been advanced yet. */

        if ((!wf_entry::issafe(Pair)
             && Config::unpack_counter(vload(&q->Head)) > T_raw)
            || (eidx != wf_entry::BOT && eidx != wf_entry::BOT_C))
        {
            /* Skip this wf_entry. Advance Note to prevent later helpers. */
            uint64_t N = wf_entry::pack(tcyc, ecyc, wf_entry::issafe(Pair),
                                     wf_entry::enq(Pair), eidx);
            if (!cas64_update(&q->Entry[j], &Pair, N))
                goto retry;
            return false;  /* Try again with next slot. */
        }

        /* Produce entry with Enq=0 (two-step insertion). */
        uint64_t N = wf_entry::pack(enote, tcyc, true, false, index);
        if (!cas64_update(&q->Entry[j], &Pair, N))
            goto retry;

        /* Finalize: set FIN on localTail. */
        uint64_t T_local = raw_to_local(T_raw);
        if (cas64(&r->localTail, T_local, T_local | wf_local_ptr::FIN_BIT)) {
            /* We finalized. Now flip Enq to 1. */
            uint64_t with_enq0 = N;
            uint64_t with_enq1 = wf_entry::pack(enote, tcyc, true, true, index);
            cas64(&q->Entry[j], with_enq0, with_enq1);
        }

        if (vload32(&q->Threshold) != Q::THRESHOLD_MAX)
            atomicExch(&q->Threshold, Q::THRESHOLD_MAX);
        return true;

    } else if (cycle_cmp(ecyc, tcyc) != 0) {
        /* Entry's cycle doesn't match ours but isn't behind us.
         * Might not yet be inserted by another helper — or already consumed. */
        return false;
    }

    /* ecyc == tcyc: already produced (by us or another helper). */
    return true;
}

/**
 * try_deq_slow: One slow-path dequeue attempt at raw ticket H_raw.
 * Paper Figure 7, lines 43-69.
 * Returns true on success or if empty.
 */
template <typename Config, uint32_t N_PARAM>
__device__ bool try_deq_slow(wf_t<Config, N_PARAM>* q,
                             uint64_t H_raw, thrdrec_t* r) {
    using Q = wf_t<Config, N_PARAM>;
    uint32_t hcyc = cycle_of(H_raw, Q::SLOTS);
    // uint32_t j = cache_remap(slot_of(H_raw, Q::SLOTS), Q::SLOTS);
    uint32_t j = slot_idx(H_raw, Q::SLOTS);

    uint64_t Pair = vload(&q->Entry[j]);
retry:;
    uint32_t ecyc  = wf_entry::cycle(Pair);
    uint32_t enote = wf_entry::note(Pair);
    uint32_t eidx  = wf_entry::index(Pair);

    /* Ready or consumed by helper (cycle matches and has a real index). */
    if (cycle_cmp(ecyc, hcyc) == 0 && eidx != wf_entry::BOT && eidx != wf_entry::BOT_C) {
        /* Terminate helpers. */
        uint64_t H_local = raw_to_local(H_raw);
        cas64(&r->localHead, H_local, H_local | wf_local_ptr::FIN_BIT);
        return true;
    }

    /* Determine replacement value. */
    uint64_t N_note = enote;
    uint64_t Val = wf_entry::pack(N_note, hcyc, wf_entry::issafe(Pair), true, wf_entry::BOT);

    if (eidx != wf_entry::BOT && eidx != wf_entry::BOT_C) {
        /* Entry has a real index from a different cycle. */
        if (cycle_cmp(ecyc, hcyc) < 0 && cycle_cmp(enote, hcyc) < 0) {
            /* Advance Note to prevent helper dequeuers from using it. */
            uint64_t N = wf_entry::pack(hcyc, ecyc, wf_entry::issafe(Pair),
                                     wf_entry::enq(Pair), eidx);
            if (!cas64_update(&q->Entry[j], &Pair, N))
                goto retry;
            N_note = hcyc;
        }
        /* Clear IsSafe. */
        Val = wf_entry::pack(N_note, ecyc, false, wf_entry::enq(Pair), eidx);
    }

    if (cycle_cmp(ecyc, hcyc) < 0) {
        uint64_t New = Val;
        if (!cas64_update(&q->Entry[j], &Pair, New))
            goto retry;
    }

    /* Check if queue is empty. */
    uint64_t T = Config::unpack_counter(vload(&q->Tail));
    if (T <= H_raw + 1) {
        catchup(q, T, H_raw + 1);
    }
    if (vload32(&q->Threshold) < 0) {
        uint64_t H_local = raw_to_local(H_raw);
        cas64(&r->localHead, H_local, H_local | wf_local_ptr::FIN_BIT);
        return true;  /* Empty. */
    }
    return false;  /* Try again. */
}

/*============================================================================
 * SLOW PATH DRIVERS  (Figure 7 lines 70-76)
 *============================================================================*/

template <typename Config, uint32_t N_PARAM>
__device__ void enqueue_slow(wf_t<Config, N_PARAM>* q, uint64_t v,
                             uint32_t index, thrdrec_t* r) {

    #if defined(GWF_PROFILE) // OPTIONALLY PROFILE SLOW ENQUEUE CALLS
    GWF_PROF_ADD(g_prof_slow_enq_entries, 1);
    #endif
    /* v is in local format. slow_faa updates it. */
    while (slow_faa(q, &q->Tail, &r->localTail, &v, nullptr)) {
        uint64_t T_raw = local_to_raw(v);
        if (try_enq_slow(q, T_raw, index, r))
            break;
    }
}

template <typename Config, uint32_t N_PARAM>
__device__ void dequeue_slow(wf_t<Config, N_PARAM>* q,
                             uint64_t v, thrdrec_t* r) {
    #if defined(GWF_PROFILE) // OPTIONALLY PROFILE SLOW DEQUEUE CALLS
    GWF_PROF_ADD(g_prof_slow_deq_entries, 1);
    #endif
    /* v is in local format. slow_faa updates it. */
    while (slow_faa(q, &q->Head, &r->localHead, &v, &q->Threshold)) {
        uint64_t H_raw = local_to_raw(v);
        if (try_deq_slow(q, H_raw, r))
            break;
    }
}

/*============================================================================
 * HELPING PROCEDURES  (Figure 6)
 *============================================================================*/

template <typename Config, uint32_t N_PARAM>
__device__ void help_enqueue(wf_t<Config, N_PARAM>* q, thrdrec_t* thr) {
    // uint32_t seq = thr->seq2;
    uint32_t seq = aload_u32(&thr->seq2);
    __threadfence();
    // uint32_t is_enq = thr->enqueue;
    // uint32_t idx = thr->index;
    // uint64_t init_tail = thr->initTail;  /* In local format. */
    uint32_t is_enq    = aload_u32(&thr->enqueue);
    uint32_t idx       = aload_u32(&thr->index);
    uint64_t init_tail = aload_u64(&thr->initTail);
    __threadfence();
    uint32_t seq1 = aload_u32(&thr->seq1);
    // if (is_enq && thr->seq1 == seq) {
    if (is_enq && seq1 == seq) {
        enqueue_slow(q, init_tail, idx, thr);
    }
}

template <typename Config, uint32_t N_PARAM>
__device__ void help_dequeue(wf_t<Config, N_PARAM>* q, thrdrec_t* thr) {
    // uint32_t seq = thr->seq2;
    uint32_t seq = aload_u32(&thr->seq2);
    __threadfence();
    // uint32_t is_enq = thr->enqueue;
    // uint64_t init_head = thr->initHead;  /* In local format. */
    uint32_t is_enq = aload_u32(&thr->enqueue);
    uint64_t init_head  = aload_u64(&thr->initHead);
    __threadfence();
    uint32_t seq1 = aload_u32(&thr->seq1);
    if (!is_enq && seq1 == seq) {
        dequeue_slow(q, init_head, thr);
    }
}

template <typename Config, uint32_t N_PARAM>
__device__ void help_threads(wf_t<Config, N_PARAM>* q) {
    uint32_t my_tid = wf_tid();
    thrdrec_t* my_rec = &q->Record[my_tid];

    if (--my_rec->nextCheck != 0)
        return;

    thrdrec_t* thr = &q->Record[my_rec->nextTid];
    uint32_t pend = aload_u32(&thr->pending);
    if (pend) {
        uint32_t is_enq = aload_u32(&thr->enqueue);
        if (is_enq) help_enqueue(q, thr);
        else        help_dequeue(q, thr);
    }

    my_rec->nextCheck = WF_HELP_DELAY;
    my_rec->nextTid = (my_rec->nextTid + 1) % Config::MAX_THREADS;
}


/**
 * LAYER 5: GPU-SPECIFIC WARP OPTIMIZATIONS
 *
 * Insert this section in gwf_ring.hpp BETWEEN:
 *   - "HELPING PROCEDURES (Figure 6)" section  (help_threads, etc.)
 *   - "TOP LEVEL: Enqueue_wf / Dequeue_wf (Figure 5)" section
 *
 * PROOF ALIGNMENT (Section 6.2 of correctness proof):
 *
 *   Asterisk 1 (Intra-warp): All lanes execute in lockstep.
 *   __ballot gives O(1) detection of pending requests across the
 *   entire wavefront. No scheduling hazard exists — if any lane
 *   runs, all run. Wait-freedom is unconditional.
 *
 *   The cooperative slow_faa (Figure 7) is ALREADY designed for
 *   multiple threads operating on the same {local, global} state:
 *     - Each helper uses its OWN phase2 record (my_tid)
 *     - All helpers CAS on the VICTIM's localTail/localHead
 *     - Exactly one CAS succeeds per global increment
 *     - FIN propagation terminates all helpers (Lemma 5.4)
 *
 *   Adding more helpers (64 lanes vs 1) increases CAS contention
 *   on the global word, but REDUCES the wall-clock time for the
 *   victim's request to complete. The bound from Lemma 6.2 becomes
 *   O(n) instead of O(k*n) because all helpers are co-scheduled.
 *
 * MEMORY ORDERING:
 *   Within a warp/wavefront, all lanes see the same memory state
 *   at each instruction boundary (SIMT). __threadfence() is still
 *   needed for global memory visibility to OTHER warps, but
 *   intra-warp reads of the victim's record are consistent as long
 *   as the publishing thread (victim) used __threadfence() before
 *   setting pending=1 — which Enqueue_wf/Dequeue_wf already do.
 */

/*============================================================================
 * WARP SIZE DETECTION
 *============================================================================*/

#ifndef WF_WARP_SIZE
  #if defined(__AMDGCN_WAVEFRONT_SIZE)
    #define WF_WARP_SIZE __AMDGCN_WAVEFRONT_SIZE   /* 64 on MI210/MI300A */
  #elif defined(__CUDA_ARCH__)
    #define WF_WARP_SIZE 32
  #else
    #define WF_WARP_SIZE 64
  #endif
#endif

/*============================================================================
 * WARP PRIMITIVES
 *============================================================================*/

/** Lane index within the current warp/wavefront. */
__device__ __forceinline__
uint32_t wf_lane_id() {
    return threadIdx.x & (WF_WARP_SIZE - 1);
}

/**
 * 64-bit warp shuffle.  HIP/CUDA __shfl is 32-bit, so we split.
 * Cost: 2 register shuffles (sub-cycle on GCN/CDNA).
 */
__device__ __forceinline__
uint64_t wf_shfl64(uint64_t val, int source_lane) {
    uint32_t lo = (uint32_t)(val);
    uint32_t hi = (uint32_t)(val >> 32);
    lo = __shfl(lo, source_lane);
    hi = __shfl(hi, source_lane);
    return ((uint64_t)hi << 32) | (uint64_t)lo;
}

/*============================================================================
 * warp_help_scan  (Layer 5)
 *
 * Single-instruction detection of pending slow-path requests across
 * the entire wavefront via __ballot.
 *
 * Returns: lane index [0, WF_WARP_SIZE) of first pending thread,
 *          or -1 if no warp-mate is pending.
 *
 * Cost: 1 global load (own pending flag) + 1 ballot + 1 ffs.
 * Compare to help_threads: countdown + global load of peer record.
 *============================================================================*/

template <typename Config, uint32_t N_PARAM>
__device__ int warp_help_scan(wf_t<Config, N_PARAM>* q) {
    uint32_t my_tid = wf_tid();

    /* Each lane loads its own pending flag — already in L1 from
     * the thread's own record access patterns. */
    uint32_t my_pending = q->Record[my_tid].pending;

    /* Ballot: one instruction, all lanes contribute simultaneously.
     * Inactive lanes (tid >= MAX_THREADS) have pending=0 from init. */
#if WF_WARP_SIZE == 64
    unsigned long long pending_mask = __ballot(my_pending);
    if (pending_mask == 0ULL) return -1;
    return __ffsll((long long)pending_mask) - 1;   /* ffsll is 1-indexed */
#else
    unsigned int pending_mask = __ballot_sync(0xFFFFFFFF, my_pending);
    if (pending_mask == 0u) return -1;
    return __ffs((int)pending_mask) - 1;            /* ffs is 1-indexed  */
#endif
}

/*============================================================================
 * warp_broadcast_request  (Layer 5)
 *
 * Once warp_help_scan identifies a victim lane, only THAT lane reads
 * the victim's help-request fields from global memory. All other lanes
 * receive the values via register shuffles — zero global memory traffic
 * for (WF_WARP_SIZE - 1) lanes.
 *
 * On GCN/CDNA, each __shfl is a single ds_swizzle or v_readlane.
 *============================================================================*/

__device__ void warp_broadcast_request(uint64_t* init_val,
                                       uint32_t* index,
                                       uint32_t* is_enqueue,
                                       int source_lane) {
    *init_val   = wf_shfl64(*init_val, source_lane);
    *index      = __shfl(*index,      source_lane);
    *is_enqueue = __shfl(*is_enqueue, source_lane);
}

/*============================================================================
 * warp_help_threads  (Layer 5 — main entry point)
 *
 * Full warp-cooperative helping sequence:
 *   1. Ballot scan to find a victim                  [O(1)]
 *   2. Single-lane read of victim's record           [1 global load]
 *   3. Register broadcast to all lanes               [3 shuffles]
 *   4. Sequence consistency check                    [compare]
 *   5. All lanes enter cooperative slow path         [slow_faa]
 *
 * This replaces the per-thread round-robin for INTRA-WARP helping.
 * Cross-warp helping still uses the existing help_threads().
 *
 * PROOF: Lemma 6.3 guarantees convergence. Within a warp, convergence
 * is IMMEDIATE (one ballot). FIN propagation (INV-7) ensures all
 * helpers exit once the request completes.
 *============================================================================*/

template <typename Config, uint32_t N_PARAM>
__device__ void warp_help_threads(wf_t<Config, N_PARAM>* q) {
    int victim_lane = warp_help_scan(q);
    if (victim_lane < 0) return;  /* No warp-mate needs help. */

    uint32_t my_lane = wf_lane_id();

    /* Compute victim's global thread ID. */
    uint32_t warp_base_tid = wf_tid() - my_lane;
    uint32_t victim_tid = warp_base_tid + (uint32_t)victim_lane;

    /* Safety: if the victim lane maps beyond the thread space, bail.
     * This handles partial warps at the tail of the last block. */
    if (victim_tid >= Config::MAX_THREADS) return;

    thrdrec_t* victim_rec = &q->Record[victim_tid];

    /* ── Single-lane global read + broadcast ── */
    uint64_t init_val   = 0;
    uint32_t index      = 0;
    uint32_t is_enqueue = 0;
    uint32_t seq1_val   = 0;
    uint32_t seq2_val   = 0;

    if (my_lane == (uint32_t)victim_lane) {
        /*
         * Read the victim's published request.
         * The victim wrote these fields with __threadfence() before
         * setting pending=1 (see Enqueue_wf/Dequeue_wf below).
         *
         * Ordering: seq2 → fields → seq1 (seqlock pattern).
         */
        seq2_val   = victim_rec->seq2;
        __threadfence();  /* acquire: seq2 before fields */
        is_enqueue = victim_rec->enqueue;
        index      = victim_rec->index;
        init_val   = is_enqueue
                         ? victim_rec->initTail
                         : victim_rec->initHead;
        __threadfence();  /* acquire: fields before seq1 */
        seq1_val   = victim_rec->seq1;
    }

    /* Broadcast to all lanes via register shuffles.
     * Cost: 5 shuffles (3 in warp_broadcast_request + 2 for seqs).
     * Saves (WF_WARP_SIZE - 1) global memory loads. */
    warp_broadcast_request(&init_val, &index, &is_enqueue, victim_lane);
    seq1_val = __shfl(seq1_val, victim_lane);
    seq2_val = __shfl(seq2_val, victim_lane);

    /* Seqlock consistency: if seq1 != seq2, the victim was still
     * writing its request — stale data, don't help. */
    if (seq1_val != seq2_val) return;

    /* ── All lanes cooperatively execute the slow path ──
     *
     * Inside slow_faa, each lane uses:
     *   - Its OWN phase2 record (q->Record[wf_tid()].phase2)
     *   - The VICTIM's localTail/localHead (passed via victim_rec)
     *   - The shared global Head or Tail
     *
     * This is exactly the cooperative model from Figure 7.
     * With WF_WARP_SIZE helpers instead of 1, the expected number
     * of slow_faa iterations to succeed drops by up to WF_WARP_SIZE×.
     *
     * FIN propagation (INV-7): once ANY helper succeeds, FIN is set
     * on the victim's localTail/localHead. All other helpers see FIN
     * via the `*v & FIN_BIT` check in slow_faa and exit. */
    if (is_enqueue) {
        enqueue_slow(q, init_val, index, victim_rec);
    } else {
        dequeue_slow(q, init_val, victim_rec);
    }
}

// /*============================================================================
//  * TOP LEVEL: Enqueue_wf / Dequeue_wf  (Figure 5) — WITH WARP OPTS
// *============================================================================*/

// template <typename Config, uint32_t N_PARAM>
// __device__ void Enqueue_wf(wf_t<Config, N_PARAM>* q, uint32_t index) {
//     using Q = wf_t<Config, N_PARAM>;
//     uint32_t my_tid = wf_tid();
 
//     /* [WARP-OPT] Warp-level cooperative helping: O(1) detection via ballot.
//      * If any warp-mate is stuck on the slow path, all lanes help it
//      * BEFORE starting their own operation. This dramatically reduces
//      * slow-path latency under contention.
//      *
//      * Cost when no one is pending: 1 global load + 1 ballot + 1 branch.
//      * Cost when someone is pending: full cooperative slow_faa execution. */
//     warp_help_threads(q);
 
//     /* Cross-warp round-robin helping (unchanged).
//      * Covers the case where the stuck thread is in a DIFFERENT warp.
//      * Proof Section 6.2, Asterisk 2: needed for inter-warp progress. */
//     help_threads(q);
 
//     /* ── Fast path (SCQ) ── */
//     uint64_t tail_raw = 0;
//     int count = WF_ENQ_PATIENCE;
//     while (--count != 0) {
//         tail_raw = try_enq(q, index);
//         if (tail_raw == 0) return;  /* Success. */
//     }
 
//     /* ── Slow path (wf) ── */
//     thrdrec_t* r = &q->Record[my_tid];
//     uint32_t seq = r->seq1;
//     uint64_t tail_local = raw_to_local(tail_raw);
 
//     /* Publish help request.  Order matters:
//      *   1. Write fields (localTail, initTail, index, enqueue)
//      *   2. __threadfence()  — make fields visible to all threads
//      *   3. Write seq2        — seqlock commit
//      *   4. __threadfence()  — seq2 visible before pending
//      *   5. Write pending=1   — triggers ballot detection in other warps
//      */
//     r->localTail = (unsigned long long)tail_local;
//     r->initTail = tail_local;
//     r->index = index;
//     r->enqueue = 1;
//     __threadfence();
//     r->seq2 = seq;
//     __threadfence();
//     r->pending = 1;
 
//     /* Execute slow path.  Other warps' warp_help_threads() will
//      * detect our pending=1 via ballot and join cooperatively. */
//     enqueue_slow(q, tail_local, index, r);
 
//     /* Clear pending BEFORE updating seq1.
//      * This prevents a helper from reading stale fields with a new seq. */
//     r->pending = 0;
//     __threadfence();
//     r->seq1 = seq + 1;
// }
 
// template <typename Config, uint32_t N_PARAM>
// __device__ uint32_t Dequeue_wf(wf_t<Config, N_PARAM>* q) {
//     using Q = wf_t<Config, N_PARAM>;
//     uint32_t my_tid = wf_tid();
 
//     /* Early exit if empty. */
//     if (vload32(&q->Threshold) < 0)
//         return wf_entry::BOT;
 
//     /* [WARP-OPT] Warp-level cooperative helping. */
//     warp_help_threads(q);
 
//     /* Cross-warp round-robin helping. */
//     help_threads(q);
 
//     /* ── Fast path (SCQ) ── */
//     uint32_t out_idx;
//     uint64_t head_raw = 0;
//     int count = WF_DEQ_PATIENCE;
//     while (--count != 0) {
//         head_raw = try_deq(q, &out_idx);
//         if (head_raw == 0) return out_idx;
//     }
 
//     /* ── Slow path (wf) ── */
//     thrdrec_t* r = &q->Record[my_tid];
//     uint32_t seq = r->seq1;
//     uint64_t head_local = raw_to_local(head_raw);
 
//     r->localHead = (unsigned long long)head_local;
//     r->initHead = head_local;
//     r->enqueue = 0;
//     __threadfence();
//     r->seq2 = seq;
//     __threadfence();
//     r->pending = 1;
 
//     dequeue_slow(q, head_local, r);
 
//     r->pending = 0;
//     __threadfence();
//     r->seq1 = seq + 1;
 
//     /* ── Gather slow-path result ── */
//     uint64_t h_local = vload(&r->localHead);
//     uint64_t h_raw = local_to_raw(h_local);
//     uint32_t hcyc = cycle_of(h_raw, Q::SLOTS);
//     uint32_t j = cache_remap(slot_of(h_raw, Q::SLOTS), Q::SLOTS);
//     uint64_t Ent = vload(&q->Entry[j]);
 
//     if (cycle_cmp(wf_entry::cycle(Ent), hcyc) == 0 && 
//         wf_entry::index(Ent) != wf_entry::BOT && 
//         wf_entry::index(Ent) != wf_entry::BOT_C)
//     {
//         consume(q, h_raw, j, Ent);
//         return wf_entry::index(Ent);
//     }
//     return wf_entry::BOT;  /* Empty. */
// }
 
// /*============================================================================
//  * INDIRECTION LAYER — unchanged, included for completeness
//  *============================================================================*/
 
// template <typename Config, uint32_t N_PARAM>
// __device__ bool Enqueue_Ptr(wf_mpmc_t<Config, N_PARAM>* q, uint64_t value) {
//     uint32_t idx = Dequeue_wf(&q->fq);
//     if (idx == wf_entry::BOT) return false;  /* Full. */
//     q->data[idx] = value;
//     __threadfence();
//     Enqueue_wf(&q->aq, idx);
//     return true;
// }
 
// template <typename Config, uint32_t N_PARAM>
// __device__ bool Dequeue_Ptr(wf_mpmc_t<Config, N_PARAM>* q, uint64_t* out) {
//     uint32_t idx = Dequeue_wf(&q->aq);
//     if (idx == wf_entry::BOT) return false;  /* Empty. */
//     *out = q->data[idx];
//     __threadfence();
//     Enqueue_wf(&q->fq, idx);
//     return true;
// }

/*============================================================================
 * TOP LEVEL: Enqueue_wf / Dequeue_wf  (Figure 5)
 *============================================================================*/

template <typename Config, uint32_t N_PARAM>
__device__ void Enqueue_wf(wf_t<Config, N_PARAM>* q, uint32_t index) {
    using Q = wf_t<Config, N_PARAM>;
    uint32_t my_tid = wf_tid();

    help_threads(q);

    /* ── Fast path (SCQ) ── */
    uint64_t tail_raw = 0;
    int count = WF_ENQ_PATIENCE;
    while (--count != 0) {
        tail_raw = try_enq(q, index);
        if (tail_raw == 0) return;  /* Success. */
    }

    /* ── Slow path (wf) ── */
    thrdrec_t* r = &q->Record[my_tid];
    uint32_t seq = r->seq1;
    uint64_t tail_local = raw_to_local(tail_raw);

    // r->localTail = (unsigned long long)tail_local;
    astore_u64(&r->localTail, tail_local); // We switch to atomic visibility
    // r->initTail = tail_local;
    astore_u64(&r->initTail, tail_local); // set init local to tail
    // r->index = index;
    astore_u32(&r->index, index);
    // r->enqueue = 1;
    astore_u32(&r->enqueue, 1u);

    // release mechanism before commiting
    __threadfence();
    // r->seq2 = seq;
    astore_u32(&r->seq2, seq);
    __threadfence(); // Release commit before publishing
    // r->pending = 1;
    astore_u32(&r->pending, 1u);

    // profile start of slow path OPTIONAL
    #if defined(GWF_PROFILE)
    GWF_PROF_ADD(g_prof_slow_enq_entries, 1);
    #endif

    enqueue_slow(q, tail_local, index, r);

    // r->pending = 0;
    // __threadfence();
    // r->seq1 = seq + 1;
    astore_u32(&r->pending, 0);
    __threadfence();
    astore_u32(&r->seq1, seq + 1);
}

template <typename Config, uint32_t N_PARAM>
__device__ uint32_t Dequeue_wf(wf_t<Config, N_PARAM>* q) {
    using Q = wf_t<Config, N_PARAM>;
    uint32_t my_tid = wf_tid();

    /* Early exit if empty. */
    if (vload32(&q->Threshold) < 0)
        return wf_entry::BOT;

    help_threads(q);

    /* ── Fast path (SCQ) ── */
    uint32_t out_idx;
    uint64_t head_raw = 0;
    int count = WF_DEQ_PATIENCE;
    while (--count != 0) {
        head_raw = try_deq(q, &out_idx);
        if (head_raw == 0) return out_idx;
    }

    /* ── Slow path (wf) ── */
    thrdrec_t* r = &q->Record[my_tid];
    uint32_t seq = r->seq1;
    uint64_t head_local = raw_to_local(head_raw);

    // r->localHead = (unsigned long long)head_local;
    // r->initHead = head_local;
    // r->enqueue = 0;
    // __threadfence();
    // r->seq2 = seq;
    // __threadfence();
    // r->pending = 1;

    astore_u64(&r->localHead, head_local);
    astore_u64(&r->initHead,  head_local);
    astore_u32(&r->enqueue,   0);

    __threadfence();
    astore_u32(&r->seq2, seq);
    __threadfence();
    astore_u32(&r->pending, 1);

    #if defined(GWF_PROFILE) // profile start of slow path OPTIONAL
    GWF_PROF_ADD(g_prof_slow_deq_entries, 1);
    #endif
    dequeue_slow(q, head_local, r);

    // r->pending = 0;
    // __threadfence();
    // r->seq1 = seq + 1;
    astore_u32(&r->pending, 0);
    __threadfence();
    astore_u32(&r->seq1, seq + 1);

    /* ── Gather slow-path result ── */
    uint64_t h_local = vload(&r->localHead);
    uint64_t h_raw = local_to_raw(h_local);
    uint32_t hcyc = cycle_of(h_raw, Q::SLOTS);
    // uint32_t j = cache_remap(slot_of(h_raw, Q::SLOTS), Q::SLOTS);
    uint32_t j = slot_idx(h_raw, Q::SLOTS);
    uint64_t Ent = vload(&q->Entry[j]);

    if (cycle_cmp(wf_entry::cycle(Ent), hcyc) == 0 && 
        wf_entry::index(Ent) != wf_entry::BOT
        && wf_entry::index(Ent) != wf_entry::BOT_C
    ){
        consume(q, h_raw, j, Ent);
        return wf_entry::index(Ent);
    }
    return wf_entry::BOT;  /* Empty. */
}

/*============================================================================
 * INDIRECTION LAYER  (Figure 2)
 *============================================================================*/

template <typename Config, uint32_t N_PARAM>
__device__ bool Enqueue_Ptr(wf_mpmc_t<Config, N_PARAM>* q, uint64_t value) {
    uint32_t idx = Dequeue_wf(&q->fq);
    if (idx == wf_entry::BOT) return false;  /* Full. */
    q->data[idx] = value;
    __threadfence();  /* Ensure data is visible before index is published. */
    Enqueue_wf(&q->aq, idx);
    return true;
}

template <typename Config, uint32_t N_PARAM>
__device__ bool Dequeue_Ptr(wf_mpmc_t<Config, N_PARAM>* q, uint64_t* out) {
    uint32_t idx = Dequeue_wf(&q->aq);
    if (idx == wf_entry::BOT) return false;  /* Empty. */
    *out = q->data[idx];
    __threadfence();  /* Ensure data is read before index is recycled. */
    Enqueue_wf(&q->fq, idx);
    return true;
}