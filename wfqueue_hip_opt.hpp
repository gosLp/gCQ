#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>

/*
 * GPU port of "A Wait-free Queue as Fast as Fetch-and-Add" (Yang & Mellor-Crummey, PPoPP'16)
 *
 * This version fixes the main issues we discussed:
 *   1) deq_fast never returns WF_TOP as data
 *   2) help_deq never treats WF_TOP as a valid dequeue candidate
 *   3) help_deq advances H while scanning (matching the CPU algorithm's intent)
 *   4) help_enq rotates peers correctly instead of getting stuck
 *   5) request publication/observation uses acquire-release ordering
 *   6) cell value publication in enq_commit is atomic
 *
 * Note:
 *   - The request synchronization word is state.
 *   - val/id are written before state is release-published.
 *   - Helpers acquire-load state before reading val/id.
 *   - val/id are accessed through atomic wrappers too, for safety/clarity.
 */

/*************** Tunables ****************/
#ifndef WF_SEGMENT_SIZE
#define WF_SEGMENT_SIZE 1024u
#endif

#ifndef WF_PATIENCE
#define WF_PATIENCE 16
#endif

#ifndef WF_PREALLOC_OPS_PER_THREAD
#define WF_PREALLOC_OPS_PER_THREAD 512
#endif

#ifndef WF_SEGMENT_SAFETY
#define WF_SEGMENT_SAFETY 8
#endif

/*************** Sentinels & helpers ****************/
static __device__ __constant__ uint64_t WF_BOTTOM = 0ull;                    // ⊥
static __device__ __constant__ uint64_t WF_TOP    = 0xFFFFFFFFFFFFFFFFull;   // >
#ifndef WF_EMPTY
#define WF_EMPTY 0ull
#endif
static __device__ __constant__ uint64_t WF_RETRY  = 0xFFFFFFFFFFFFFFFEull;

struct EnqReq;
struct DeqReq;
struct Segment;
struct wf_queue;
struct wf_handle;

static __device__ __constant__ EnqReq* ENQ_NULL = (EnqReq*)0ull;
static __device__ __constant__ EnqReq* ENQ_TOPP = (EnqReq*)1ull;
static __device__ __constant__ DeqReq* DEQ_NULL = (DeqReq*)0ull;
static __device__ __constant__ DeqReq* DEQ_TOPP = (DeqReq*)1ull;

/*************** Low-level atomic helpers ****************/

// SC-ish RMW helpers used throughout the queue
static __device__ __forceinline__ uint64_t cas64(uint64_t* addr, uint64_t expected, uint64_t desired) {
    return atomicCAS(reinterpret_cast<unsigned long long*>(addr),
                     static_cast<unsigned long long>(expected),
                     static_cast<unsigned long long>(desired));
}

static __device__ __forceinline__ uint64_t faa64(uint64_t* addr, uint64_t inc) {
    return atomicAdd(reinterpret_cast<unsigned long long*>(addr),
                     static_cast<unsigned long long>(inc));
}

static __device__ __forceinline__ uint64_t exch64(uint64_t* addr, uint64_t value) {
    return atomicExch(reinterpret_cast<unsigned long long*>(addr),
                      static_cast<unsigned long long>(value));
}

static __device__ __forceinline__ uint64_t atomic_load_u64(uint64_t* addr) {
    return atomicAdd(reinterpret_cast<unsigned long long*>(addr), 0ull);
}

// clang/hip acquire-release wrappers for request publication.
// These are only used for request payload/state synchronization.
static __device__ __forceinline__ uint64_t load_rlx_u64(const uint64_t* addr) {
    return (uint64_t)__atomic_load_n(reinterpret_cast<const unsigned long long*>(addr), __ATOMIC_RELAXED);
}

static __device__ __forceinline__ uint64_t load_acq_u64(const uint64_t* addr) {
    return (uint64_t)__atomic_load_n(reinterpret_cast<const unsigned long long*>(addr), __ATOMIC_ACQUIRE);
}

static __device__ __forceinline__ void store_rlx_u64(uint64_t* addr, uint64_t v) {
    __atomic_store_n(reinterpret_cast<unsigned long long*>(addr),
                     (unsigned long long)v, __ATOMIC_RELAXED);
}

static __device__ __forceinline__ void store_rel_u64(uint64_t* addr, uint64_t v) {
    __atomic_store_n(reinterpret_cast<unsigned long long*>(addr),
                     (unsigned long long)v, __ATOMIC_RELEASE);
}

template <typename T>
static __device__ __forceinline__ T* atomic_load_ptr(T** addr) {
    uint64_t raw = atomicCAS(reinterpret_cast<unsigned long long*>(addr), 0ull, 0ull);
    return reinterpret_cast<T*>(raw);
}

static __device__ __forceinline__ EnqReq* cas_enq(EnqReq** addr, EnqReq* expected, EnqReq* desired) {
    uint64_t old = atomicCAS(reinterpret_cast<unsigned long long*>(addr),
                             reinterpret_cast<unsigned long long>(expected),
                             reinterpret_cast<unsigned long long>(desired));
    return reinterpret_cast<EnqReq*>(old);
}

static __device__ __forceinline__ DeqReq* cas_deq(DeqReq** addr, DeqReq* expected, DeqReq* desired) {
    uint64_t old = atomicCAS(reinterpret_cast<unsigned long long*>(addr),
                             reinterpret_cast<unsigned long long>(expected),
                             reinterpret_cast<unsigned long long>(desired));
    return reinterpret_cast<DeqReq*>(old);
}

/*************** Bit packing for request state ****************/
// Layout: state = (pending<<63) | low63
static __device__ __forceinline__ uint64_t pack_state(int pending, uint64_t low63) {
    return (static_cast<uint64_t>(pending) << 63) | (low63 & ((1ull << 63) - 1));
}
static __device__ __forceinline__ int state_pending(uint64_t st)  { return static_cast<int>(st >> 63); }
static __device__ __forceinline__ uint64_t state_low(uint64_t st) { return st & ((1ull << 63) - 1); }

/*************** Core structs ****************/
struct EnqReq {
    uint64_t val;      // published before state
    uint64_t state;    // (pending, original-id or claimed-cell-id)
};

struct DeqReq {
    uint64_t id;       // published before state
    uint64_t state;    // (pending, announced candidate index)
};

struct Cell {
    uint64_t val;      // data, WF_BOTTOM, or WF_TOP
    EnqReq*  enq;      // ENQ_NULL, request*, or ENQ_TOPP
    DeqReq*  deq;      // DEQ_NULL, request*, or DEQ_TOPP
};

struct Segment {
    int64_t  id;
    Segment* next;
    Cell     cells[WF_SEGMENT_SIZE];
};

struct SegmentPool {
    Segment* segments;
    uint32_t capacity;
    uint32_t segment_size;
    uint32_t active_segments;
};

struct wf_handle {
    Segment*   tail;
    Segment*   head;
    wf_handle* next;

    struct {
        EnqReq     req;
        wf_handle* peer;
        uint64_t   id_last;
    } enq;

    struct {
        DeqReq     req;
        wf_handle* peer;
    } deq;

    Segment* hzdp;
};

struct wf_queue {
    Segment* Q;
    uint64_t T;
    uint64_t H;
    int64_t  I;
    SegmentPool pool;
#ifdef WF_BOUNDS_CHECK
    unsigned int bounds_oob;
#endif
};

/*************** Utility ****************/
static __device__ __forceinline__ void advance_end_for_linearizability(uint64_t* E, uint64_t cid) {
    while (true) {
        uint64_t e = atomic_load_u64(E);
        if (e >= cid) break;
        uint64_t old = cas64(E, e, cid);
        if (old == e) break;
    }
}

static __device__ __forceinline__ void init_cell(Cell& c) {
    c.val = WF_BOTTOM;
    c.enq = ENQ_NULL;
    c.deq = DEQ_NULL;
}

/*************** Segment access ****************/
static __device__ __forceinline__ Cell* find_cell(wf_queue* q, Segment** sp, int64_t cell_id) {
    if (cell_id < 0) {
        *sp = nullptr;
        return nullptr;
    }

    uint32_t seg_idx = static_cast<uint32_t>(cell_id / static_cast<int64_t>(WF_SEGMENT_SIZE));
    uint32_t cap = q->pool.capacity;

    if (seg_idx >= cap) {
        *sp = nullptr;
        return nullptr;
    }

#ifdef WF_BOUNDS_CHECK
    if (__builtin_expect(seg_idx >= cap, 0)) {
        atomicExch(reinterpret_cast<unsigned int*>(&q->bounds_oob), 1u);
        seg_idx = cap - 1;
    }
#endif

    if (seg_idx < cap) {
        unsigned int* act = reinterpret_cast<unsigned int*>(&q->pool.active_segments);
        atomicMax(act, seg_idx + 1);
    }

    Segment* s = &q->pool.segments[seg_idx];
    *sp = s;
    return &s->cells[static_cast<uint32_t>(cell_id) & (WF_SEGMENT_SIZE - 1)];
}

/*************** Request-state helpers ****************/
static __device__ __forceinline__ uint64_t enq_req_state_load(const EnqReq* r) {
    return load_acq_u64(&r->state);
}

static __device__ __forceinline__ uint64_t deq_req_state_load(const DeqReq* r) {
    return load_acq_u64(&r->state);
}

static __device__ __forceinline__ bool try_to_claim_req(uint64_t* state_addr, uint64_t expected_id, uint64_t cell_id) {
    uint64_t expected = pack_state(1, expected_id);
    uint64_t desired  = pack_state(0, cell_id);
    return cas64(state_addr, expected, desired) == expected;
}

/*************** Enqueue fast/slow + helpers ****************/

// Forward decl for help_enq used by dequeuers
static __device__ uint64_t help_enq(wf_queue* q, wf_handle* h, Cell* c, uint64_t i);

static __device__ __forceinline__ void enq_commit(wf_queue* q, Cell* c, uint64_t v, uint64_t cid) {
    // Match the paper's linearization discipline: once a value is published in Q[cid],
    // T must already cover cid+1.
    advance_end_for_linearizability(&q->T, cid + 1);

    // Atomic store of the actual value to replace WF_TOP.
    // This is only done by the thread/helper that owns the request-cell assignment.
    exch64(&c->val, v);
}

static __device__ __forceinline__ bool enq_fast(wf_queue* q, wf_handle* h, uint64_t v, uint64_t* cid_out) {
    uint64_t i = faa64(&q->T, 1ull);
    Cell* c = find_cell(q, &h->tail, static_cast<int64_t>(i));
    if (!c) {
        *cid_out = i;
        return false;
    }

    if (cas64(&c->val, WF_BOTTOM, v) == WF_BOTTOM) return true;

    *cid_out = i;
    return false;
}

static __device__ __forceinline__ bool enq_slow(wf_queue* q, wf_handle* h, uint64_t v, uint64_t cell_id) {
    EnqReq* r = &h->enq.req;

    // Publish payload first, then release-publish state.
    store_rlx_u64(&r->val, v);
    store_rel_u64(&r->state, pack_state(1, cell_id));

    Segment* tmp_tail = h->tail;

    while (true) {
        uint64_t i = faa64(&q->T, 1ull);
        Cell* c = find_cell(q, &tmp_tail, static_cast<int64_t>(i));
        if (!c) break;

        if (cas_enq(&c->enq, ENQ_NULL, r) == ENQ_NULL) {
            uint64_t cur = atomic_load_u64(&c->val);
            if (cur == WF_BOTTOM || cur == WF_TOP) {
                (void)try_to_claim_req(&r->state, cell_id, i);
                break;
            }
        }

        uint64_t s = enq_req_state_load(r);
        if (!state_pending(s)) break;
    }

    uint64_t final_s  = enq_req_state_load(r);
    uint64_t final_id = state_low(final_s);
    Cell* dest = find_cell(q, &h->tail, static_cast<int64_t>(final_id));
    if (dest) {
        uint64_t cur = atomic_load_u64(&dest->val);
        if (cur == WF_TOP) enq_commit(q, dest, v, final_id);
        return true; // Success enqueued
    }
    return false; // FAILED: Dropped ray due to pool exhuastion
}

static __device__ uint64_t help_enq(wf_queue* q, wf_handle* h, Cell* c, uint64_t i) {
    // First, try to move the cell out of WF_BOTTOM so it becomes observable.
    uint64_t old = cas64(&c->val, WF_BOTTOM, WF_TOP);
    if (old != WF_BOTTOM) {
        if (old != WF_TOP) return old; // real value already present
    }

    EnqReq* cell_enq = atomic_load_ptr(&c->enq);

    // If the cell still has no enqueue request bound to it, search peers.
    if (cell_enq == ENQ_NULL) {
        wf_handle* p = h->enq.peer ? h->enq.peer : h;
        EnqReq* r = &p->enq.req;
        uint64_t s = enq_req_state_load(r);
        uint64_t rid = state_low(s);
        int pend = state_pending(s);

        // If we were tracking this peer's request and it completed or changed, move on
        if (h->enq.id_last != 0 && h->enq.id_last != rid) {
            h->enq.id_last = 0;
            p = p->next;
            h->enq.peer = p;

            r = &p->enq.req;
            s = enq_req_state_load(r);
            rid = state_low(s);
            pend = state_pending(s);
        }

        // Try to help this one peer
        if (pend && rid > 0 && rid <= i) {
            EnqReq* prev = cas_enq(&c->enq, ENQ_NULL, r);
            
            // Did we fail to bind because a DIFFERENT request beat us?
            if (prev != ENQ_NULL && prev != r) {
                cell_enq = prev; 
                h->enq.id_last = rid; // Keep tracking our peer, it STILL needs a cell!
            } else {
                // We succeeded (prev == ENQ_NULL) OR someone else helped our peer (prev == r)
                cell_enq = r;
                h->enq.id_last = 0; 
                h->enq.peer = p->next; // Our peer has a cell now, move on to the next peer!
            }
        } else {
            // Peer doesn't need help, advance pointer for the next time
            h->enq.id_last = 0;
            h->enq.peer = p->next;
        }

        // if still null, mark as TOPP
        if (cell_enq == ENQ_NULL) {
            EnqReq* prev = cas_enq(&c->enq, ENQ_NULL, ENQ_TOPP);
            cell_enq = (prev == ENQ_NULL) ? ENQ_TOPP : prev;
        }
    }
    
    if (cell_enq == ENQ_TOPP) {
        uint64_t t = atomic_load_u64(&q->T);
        return (t <= i ? WF_EMPTY : WF_RETRY);
    }

    EnqReq* r = cell_enq;
    uint64_t s = enq_req_state_load(r);
    int pend = state_pending(s);
    uint64_t rid = state_low(s);
    uint64_t v = load_rlx_u64(&r->val);

    // If the request is already completed, it only belongs to this cell if low(state)==i.
    if (!pend) {
        if (rid == i) {
            if (atomic_load_u64(&c->val) == WF_TOP) {
                enq_commit(q, c, v, i);
            }
            return atomic_load_u64(&c->val);
        }
        uint64_t t = atomic_load_u64(&q->T);
        return (t <= i ? WF_EMPTY : WF_RETRY);
    }

    // Pending but unsuitable for this cell
    if (rid > i) {
        uint64_t t = atomic_load_u64(&q->T);
        return (t <= i ? WF_EMPTY : WF_RETRY);
    }

    // Try to bind this request to this cell, or finish a request already claimed for this cell.
    if (try_to_claim_req(&r->state, rid, i) || enq_req_state_load(r) == pack_state(0, i)) {
        if (atomic_load_u64(&c->val) == WF_TOP) {
            enq_commit(q, c, v, i);
        }
    }

    return atomic_load_u64(&c->val);
}

/*************** Dequeue fast/slow + helpers ****************/
static __device__ __forceinline__ uint64_t deq_fast(wf_queue* q, wf_handle* h, uint64_t* id_out) {
    uint64_t i = faa64(&q->H, 1ull);
    Cell* c = find_cell(q, &h->head, static_cast<int64_t>(i));
    if (!c) {
        *id_out = i;
        return WF_EMPTY;
    }

    uint64_t v = help_enq(q, h, c, i);
    if (v == WF_EMPTY) return WF_EMPTY;

    // Critical fix: never return WF_TOP as user data.
    if (v != WF_TOP && v != WF_RETRY) {
        if (cas_deq(&c->deq, DEQ_NULL, DEQ_TOPP) == DEQ_NULL) return v;
    }

    *id_out = i;
    return WF_RETRY;
}

static __device__ __forceinline__ void help_deq(wf_queue* q, wf_handle* h, wf_handle* helpee) {
    DeqReq* r = &helpee->deq.req;
    uint64_t s = deq_req_state_load(r);
    uint64_t id = load_rlx_u64(&r->id);

    if (!state_pending(s) || state_low(s) < id) return;

    Segment* ha = helpee->head;
    h->hzdp = ha;

    uint64_t prior = id;
    uint64_t i = id;
    uint64_t cand = 0;

    while (true) {
        s = deq_req_state_load(r);
        if (!state_pending(s) || load_rlx_u64(&r->id) != id) return;

        // Search while the announced candidate is still 'prior'
        while (!cand && state_pending(s) && state_low(s) == prior) {
            Cell* c = find_cell(q, &ha, static_cast<int64_t>(++i));
            if (!c) break;

            // Critical fix: advance H during the scan so visited cells are covered.
            advance_end_for_linearizability(&q->H, i + 1);

            uint64_t v = help_enq(q, h, c, i);

            // Critical fix: WF_TOP is not a dequeueable value.
            if (v == WF_EMPTY || (v != WF_TOP && v != WF_RETRY &&
                                  atomic_load_ptr(&c->deq) == DEQ_NULL)) {
                cand = i;
                break;
            }

            s = deq_req_state_load(r);
            if (!state_pending(s) || load_rlx_u64(&r->id) != id) return;
        }

        if (cand) {
            uint64_t exp = pack_state(1, prior);
            uint64_t des = pack_state(1, cand);
            (void)cas64(&r->state, exp, des);
            cand = 0;
        }

        s = deq_req_state_load(r);
        if (!state_pending(s) || load_rlx_u64(&r->id) != id) return;

        uint64_t aidx = state_low(s);
        Cell* c = find_cell(q, &ha, static_cast<int64_t>(aidx));
        if (!c) return;

        DeqReq* cur_deq = atomic_load_ptr(&c->deq);

        if (atomic_load_u64(&c->val) == WF_TOP ||
            cas_deq(&c->deq, DEQ_NULL, &helpee->deq.req) == DEQ_NULL ||
            cur_deq == &helpee->deq.req) {
            (void)cas64(&r->state, s, pack_state(0, aidx));
            return;
        }

        prior = aidx;
        if (aidx >= i) i = aidx;
    }
}

static __device__ __forceinline__ uint64_t deq_slow(wf_queue* q, wf_handle* h, uint64_t cid) {
    DeqReq* r = &h->deq.req;

    // Publish id first, then release-publish state.
    store_rlx_u64(&r->id, cid);
    store_rel_u64(&r->state, pack_state(1, cid));

    help_deq(q, h, h);

    uint64_t s   = deq_req_state_load(r);
    uint64_t idx = state_low(s);
    Cell* c = find_cell(q, &h->head, static_cast<int64_t>(idx));
    if (!c) {
        advance_end_for_linearizability(&q->H, idx + 1);
        return WF_EMPTY;
    }

    uint64_t v = atomic_load_u64(&c->val);
    advance_end_for_linearizability(&q->H, idx + 1);

    return (v == WF_TOP ? WF_EMPTY : v);
}

/*************** Public device API ****************/
__device__ __forceinline__ bool wf_enqueue(wf_queue* q, wf_handle* h, uint64_t value) {
    for (int p = WF_PATIENCE; p >= 0; --p) {
        uint64_t cid = 0;
        if (enq_fast(q, h, value, &cid)) return true;
        if (p == 0) {
            return enq_slow(q, h, value, cid);
        }
    }
    return false;
}

__device__ __forceinline__ uint64_t wf_dequeue(wf_queue* q, wf_handle* h) {
    uint64_t v = WF_RETRY;
    uint64_t cid = 0;

    for (int p = WF_PATIENCE; p >= 0; --p) {
        v = deq_fast(q, h, &cid);
        if (v != WF_RETRY) break;
    }

    if (v == WF_RETRY) v = deq_slow(q, h, cid);

    if (v != WF_EMPTY) {
        help_deq(q, h, h->deq.peer);
        h->deq.peer = h->deq.peer->next;
    }

    return v;
}

__device__ __forceinline__ uint64_t wf_dequeue_bfs(wf_queue* q, wf_handle* h) {
    uint64_t cid = 0;
    for (int p = WF_PATIENCE; p >= 0; --p) {
        uint64_t v = deq_fast(q, h, &cid);
        if (v != WF_RETRY) return v;

        uint64_t t = atomic_load_u64(&q->T);
        uint64_t hcur = atomic_load_u64(&q->H);
        if (t <= hcur + 1) return WF_EMPTY;
    }
    return WF_EMPTY;
}

/*************** Init kernel ****************/
__global__ void wf_init_kernel(wf_queue* q, wf_handle* handles, int num_threads) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_threads) return;

    wf_handle* h = &handles[tid];
    h->tail = q->Q;
    h->head = q->Q;
    h->next = &handles[(tid + 1) % num_threads];

    h->enq.peer = h->next;
    h->deq.peer = h->next;

    h->enq.req.val = 0;
    h->enq.req.state = pack_state(0, 0);

    h->deq.req.id = 0;
    h->deq.req.state = pack_state(0, 0);

    h->enq.id_last = 0;
    h->hzdp = nullptr;
}

/*************** Host-side initialization ****************/
static inline void wf_queue_host_init_ex(wf_queue** d_q, wf_handle** d_handles, int num_threads, int ops_per_thread) {
    const int SAFETY_CELLS = 128;
    uint64_t cells_per_thread = (uint64_t)(3 * ops_per_thread + SAFETY_CELLS);
    const uint64_t total_cells = (uint64_t)num_threads * cells_per_thread;
    uint32_t seg_needed = (uint32_t)((total_cells + WF_SEGMENT_SIZE - 1) / WF_SEGMENT_SIZE + WF_SEGMENT_SAFETY);
    if (seg_needed < 2) seg_needed = 2;

    hipMalloc((void**)d_q, sizeof(wf_queue));
    hipMalloc((void**)d_handles, sizeof(wf_handle) * (size_t)num_threads);

    wf_queue hq_host{};
    Segment* d_segments = nullptr;
    size_t seg_bytes = sizeof(Segment) * (size_t)seg_needed;
    hipMallocManaged((void**)&d_segments, seg_bytes);

    for (uint32_t i = 0; i < seg_needed; ++i) {
        Segment* s = &d_segments[i];
        s->id = (int64_t)i;
        s->next = (i + 1 < seg_needed) ? &d_segments[i + 1] : nullptr;
        for (uint32_t j = 0; j < WF_SEGMENT_SIZE; ++j) {
            s->cells[j].val = 0ull;
            s->cells[j].enq = (EnqReq*)0ull;
            s->cells[j].deq = (DeqReq*)0ull;
        }
    }

    hq_host.Q = &d_segments[0];
    hq_host.T = 0ull;
    hq_host.H = 0ull;
    hq_host.I = 0;

    hq_host.pool.segments = d_segments;
    hq_host.pool.capacity = seg_needed;
    hq_host.pool.segment_size = WF_SEGMENT_SIZE;
    hq_host.pool.active_segments = 1u;

#ifdef WF_BOUNDS_CHECK
    hq_host.bounds_oob = 0u;
#endif

    hipMemcpy(*d_q, &hq_host, sizeof(wf_queue), hipMemcpyHostToDevice);

    int block = 256;
    int grid = (num_threads + block - 1) / block;
    wf_init_kernel<<<grid, block>>>(*d_q, *d_handles, num_threads);
    hipDeviceSynchronize();

    int dev = 0;
    hipGetDevice(&dev);
    hipMemPrefetchAsync(d_segments, seg_bytes, dev, 0);
}

__global__ void wf_clear_pool_kernel(wf_queue* q) {
    Segment* base = q->pool.segments;
    uint32_t segs = q->pool.active_segments;
    if (!base || segs == 0) return;

    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total_cells = (uint64_t)segs * (uint64_t)WF_SEGMENT_SIZE;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;

    for (uint64_t c = gid; c < total_cells; c += stride) {
        uint32_t seg = (uint32_t)(c / WF_SEGMENT_SIZE);
        uint32_t off = (uint32_t)(c & (WF_SEGMENT_SIZE - 1));
        Cell& cell = base[seg].cells[off];
        cell.val = WF_BOTTOM;
        cell.enq = ENQ_NULL;
        cell.deq = DEQ_NULL;
    }
}

static inline void wf_queue_reset_for_bfs(wf_queue* d_q, wf_handle* d_handles, int num_threads) {
    wf_queue hq;
    hipMemcpy(&hq, d_q, sizeof(wf_queue), hipMemcpyDeviceToHost);

    const int block = 256;
    uint64_t total_cells = (uint64_t)hq.pool.active_segments * (uint64_t)WF_SEGMENT_SIZE;
    int grid = (int)((total_cells + block - 1) / block);
    if (grid < 1) grid = 1;
    if (grid > 65535) grid = 65535;

    wf_clear_pool_kernel<<<grid, block>>>(d_q);
    hipDeviceSynchronize();

    hq.T = 0;
    hq.H = 0;
    hq.I = 0;
    hq.pool.active_segments = 1;

#ifdef WF_BOUNDS_CHECK
    hq.bounds_oob = 0;
#endif

    hipMemcpy(d_q, &hq, sizeof(wf_queue), hipMemcpyHostToDevice);

    int grid2 = (num_threads + block - 1) / block;
    wf_init_kernel<<<grid2, block>>>(d_q, d_handles, num_threads);
    hipDeviceSynchronize();
}

static inline void wf_queue_host_init_for_bfs(
    wf_queue** d_q, wf_handle** d_handles,
    int num_threads, long long graph_edges, int graph_vertices, int chunk_size) {

    printf("DEBUG: graph_edges=%lld graph_vertices=%d chunk_size=%d\n",
           graph_edges, graph_vertices, chunk_size);

    uint64_t work_items = (graph_edges + chunk_size - 1) / chunk_size;
    uint64_t min_cells_needed = (uint64_t)graph_vertices + work_items;
    min_cells_needed = (uint64_t)(min_cells_needed * 1.20) + 10000;

    uint32_t seg_needed = (uint32_t)((min_cells_needed + WF_SEGMENT_SIZE - 1) / WF_SEGMENT_SIZE);
    seg_needed += WF_SEGMENT_SAFETY;
    if (seg_needed < 2) seg_needed = 2;

    hipMalloc((void**)d_q, sizeof(wf_queue));
    hipMalloc((void**)d_handles, sizeof(wf_handle) * (size_t)num_threads);

    wf_queue hq_host{};
    Segment* d_segments = nullptr;
    size_t seg_bytes = sizeof(Segment) * (size_t)seg_needed;

    hipError_t err = hipMallocManaged((void**)&d_segments, seg_bytes);
    if (err != hipSuccess) {
        fprintf(stderr, "Failed to allocate %zu bytes for segment pool\n", seg_bytes);
        return;
    }

    for (uint32_t i = 0; i < seg_needed; ++i) {
        Segment* s = &d_segments[i];
        s->id = (int64_t)i;
        s->next = (i + 1 < seg_needed) ? &d_segments[i + 1] : nullptr;
        for (uint32_t j = 0; j < WF_SEGMENT_SIZE; ++j) {
            s->cells[j].val = 0ull;
            s->cells[j].enq = (EnqReq*)0ull;
            s->cells[j].deq = (DeqReq*)0ull;
        }
    }

    hq_host.Q = &d_segments[0];
    hq_host.T = 0ull;
    hq_host.H = 0ull;
    hq_host.I = 0;

    hq_host.pool.segments = d_segments;
    hq_host.pool.capacity = seg_needed;
    hq_host.pool.segment_size = WF_SEGMENT_SIZE;
    hq_host.pool.active_segments = 1u;

#ifdef WF_BOUNDS_CHECK
    hq_host.bounds_oob = 0u;
#endif

    hipMemcpy(*d_q, &hq_host, sizeof(wf_queue), hipMemcpyHostToDevice);

    int block = 256;
    int grid = (num_threads + block - 1) / block;
    wf_init_kernel<<<grid, block>>>(*d_q, *d_handles, num_threads);
    hipDeviceSynchronize();

    int dev = 0;
    hipGetDevice(&dev);
    hipMemPrefetchAsync(d_segments, seg_bytes, dev, 0);

    printf("DEBUG: Allocated %u segments = %llu cells for %d vertices\n",
           seg_needed, (unsigned long long)seg_needed * WF_SEGMENT_SIZE, graph_vertices);
}

static inline void wf_queue_host_init(wf_queue** d_q, wf_handle** d_handles, int num_threads) {
    wf_queue_host_init_ex(d_q, d_handles, num_threads, WF_PREALLOC_OPS_PER_THREAD);
}