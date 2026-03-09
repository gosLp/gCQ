#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>

/*
 * GPU port of "A Wait-free Queue as Fast as Fetch-and-Add" (Yang & Mellor-Crummey, PPoPP'16)
 * Fast-path/slow-path FAA queue with request helping. This header is self-contained
 * and compatible with the user's test harness under -DUSE_OPT.
 *
 * Design choices for GPU:
 *  - Preallocate and prelink a contiguous array of segments on the host.
 *    No device malloc/new anywhere. find_cell() uses arithmetic to map
 *    cell_id -> segment index; we atomically grow an "active_segments" watermark.
 *  - 64-bit atomics (FAA/CAS) on MI300A are used for T/H and cell state.
 *  - Request words are 64-bit: [63]=pending bit, [62:0]=id (cell index) or idx.
 *  - Sentinels follow the paper: val: ⊥=0, >=WF_TOP (0xFFFFFFFFFFFFFFFF),
 *    enq/deq pointers: ⊥e/⊥d=nullptr, >e/>d=(EnqReq*)1/(DeqReq*)1.
 *  - Memory reclamation: we preallocate a very large linear segment array.
 *    We do not free memory during kernels; segments are logically retired when
 *    H/T advance. This preserves correctness invariants and avoids device malloc.
 *    (The host frees the pool after each test.)
 */

/*************** Tunables ****************/
#ifndef WF_SEGMENT_SIZE
#define WF_SEGMENT_SIZE 1024u   // cells per segment (power of two is nice)
#endif

#ifndef WF_PATIENCE
#define WF_PATIENCE 16          // fast-path attempts before slow-path
#endif

#ifndef WF_PREALLOC_OPS_PER_THREAD
#define WF_PREALLOC_OPS_PER_THREAD 1024  // conservative upper bound used for pool sizing
#endif

#ifndef WF_SEGMENT_SAFETY
#define WF_SEGMENT_SAFETY 8     // extra segments to absorb rounding/skew
#endif

/*************** Sentinels & helpers ****************/
static __device__ __constant__ uint64_t WF_BOTTOM = 0ull;             // ⊥
static __device__ __constant__ uint64_t WF_TOP    = 0xFFFFFFFFFFFFFFFFull; // >

// public empty value returned by dequeue when queue is empty
#ifndef WF_EMPTY
#define WF_EMPTY 0ull
#endif

// internal retry marker (never exposed to user); any value > UINT64_MAX-2 is fine
static __device__ __constant__ uint64_t WF_RETRY  = 0xFFFFFFFFFFFFFFFEull;

struct EnqReq; struct DeqReq; struct Segment; struct wf_queue; struct wf_handle;

// reserved pointer sentinels for enq/deq fields
static __device__ __constant__ EnqReq* ENQ_NULL = (EnqReq*)0ull;   // ⊥e
static __device__ __constant__ EnqReq* ENQ_TOPP = (EnqReq*)1ull;   // >e
static __device__ __constant__ DeqReq* DEQ_NULL = (DeqReq*)0ull;   // ⊥d
static __device__ __constant__ DeqReq* DEQ_TOPP = (DeqReq*)1ull;   // >d

// 64-bit atomic CAS on values
static __device__ __forceinline__ uint64_t cas64(uint64_t* addr, uint64_t expected, uint64_t desired) {
    return atomicCAS(reinterpret_cast<unsigned long long*>(addr),
                     static_cast<unsigned long long>(expected),
                     static_cast<unsigned long long>(desired));
}

static __device__ __forceinline__ uint64_t faa64(uint64_t* addr, uint64_t inc) {
    return atomicAdd(reinterpret_cast<unsigned long long*>(addr),
                     static_cast<unsigned long long>(inc));
}

// Coherent atomic load for 64-bit globals (device scope)
static __device__ __forceinline__ uint64_t atomic_load_u64(uint64_t* addr) {
    return atomicAdd(reinterpret_cast<unsigned long long*>(addr), 0ull);
}

// pointer CAS helpers (casts via 64-bit integer domain)
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
// Layout: state = (pending<<63) | (low63)
static __device__ __forceinline__ uint64_t pack_state(int pending, uint64_t low63) {
    return (static_cast<uint64_t>(pending) << 63) | (low63 & ((1ull<<63)-1));
}
static __device__ __forceinline__ int state_pending(uint64_t st)       { return static_cast<int>(st >> 63); }
static __device__ __forceinline__ uint64_t state_low(uint64_t st)      { return st & ((1ull<<63)-1); }

/*************** Core structs ****************/
struct EnqReq {
    uint64_t val;     // to be enqueued
    uint64_t state;   // (pending,id/cid)
};

struct DeqReq {
    uint64_t id;      // request id (cell index from last fast-path attempt)
    uint64_t state;   // (pending, idx)
};

struct Cell {
    uint64_t val;     // data or WF_BOTTOM/WF_TOP
    EnqReq*  enq;     // ⊥e/null, request*, or >e
    DeqReq*  deq;     // ⊥d/null, request*, or >d
};

struct Segment {
    int64_t  id;      // segment id (0..)
    Segment* next;    // prelinked at init
    Cell     cells[WF_SEGMENT_SIZE];
};

struct SegmentPool {
    Segment* segments;        // contiguous block of segments
    uint32_t capacity;        // total segments allocated
    uint32_t segment_size;    // = WF_SEGMENT_SIZE
    uint32_t active_segments; // watermark of logically-active segments (grows)
};

struct wf_handle {
    Segment*  tail;      // local tail seg for enqueues
    Segment*  head;      // local head seg for dequeues
    wf_handle* next;     // ring linkage

    struct { EnqReq req; wf_handle* peer; uint64_t id_last; } enq;
    struct { DeqReq req; wf_handle* peer; } deq;

    Segment* hzdp;       // hazard pointer (assist/deq slow path)
};

struct wf_queue {
    Segment* Q;     // head segment pointer (segment[0])
    uint64_t T;     // global tail cell index (FAA by enq)
    uint64_t H;     // global head cell index (FAA by deq)
    int64_t  I;     // id of oldest segment (for reclamation bookkeeping)
    SegmentPool pool;
#ifdef WF_BOUNDS_CHECK
    unsigned int bounds_oob; // device sets to 1 if an index exceeds pool capacity
#endif
};

/*************** Utility ****************/
static __device__ __forceinline__ void advance_end_for_linearizability(uint64_t* E, uint64_t cid) {
    // ensure *E >= cid (used for T and H)
    while (true) {
        uint64_t e = *E; // relaxed read
        if (e >= cid) break;
        uint64_t old = cas64(E, e, cid);
        if (old == e) break;
    }
}

static __device__ __forceinline__ void init_cell(Cell& c) {
    c.val = WF_BOTTOM; c.enq = ENQ_NULL; c.deq = DEQ_NULL;
}

/*************** Segment access ****************/
static __device__ __forceinline__ Cell* find_cell(wf_queue* q, Segment** sp, int64_t cell_id) {
    if (cell_id < 0) {
        *sp = nullptr;
        return nullptr;
    }
    
    // map cell_id to segment index & cell offset; grow active watermark
    uint32_t seg_idx = static_cast<uint32_t>(cell_id / static_cast<int64_t>(WF_SEGMENT_SIZE));
    uint32_t cap = q->pool.capacity;

    if (seg_idx >= cap) {
        // exceeded pool capacity; return nullptr so caller can handle
        // atomicExch(reinterpret_cast<unsigned int*>(&q->bounds_oob), 1u);
        *sp = nullptr;
        return nullptr;
    }
#ifdef WF_BOUNDS_CHECK
    if (__builtin_expect(seg_idx >= cap, 0)) {
        atomicExch(reinterpret_cast<unsigned int*>(&q->bounds_oob), 1u);
        seg_idx = cap - 1; // clamp to last segment to avoid OOB fault
    }
#endif
    // grow watermark if within capacity
    if (seg_idx < cap) {
        unsigned int* act = reinterpret_cast<unsigned int*>(&q->pool.active_segments);
        atomicMax(act, seg_idx + 1);
    }
    Segment* s = &q->pool.segments[seg_idx];
    *sp = s;
    return &s->cells[static_cast<uint32_t>(cell_id) & (WF_SEGMENT_SIZE - 1)];
}

/*************** Enqueue fast/slow + helpers ****************/
static __device__ __forceinline__ bool try_to_claim_req(uint64_t* state_addr, uint64_t id, uint64_t cell_id) {
    uint64_t expected = pack_state(1, id);
    uint64_t desired  = pack_state(0, cell_id);
    return cas64(state_addr, expected, desired) == expected;
}

static __device__ __forceinline__ void enq_commit(wf_queue* q, Cell* c, uint64_t v, uint64_t cid) {
    // linearize before publishing value
    advance_end_for_linearizability(&q->T, cid + 1);
    // publish value (single-writer for the claimed cell), then fence to make it visible
    c->val = v;
    __threadfence();
}

static __device__ __forceinline__ bool enq_fast(wf_queue* q, wf_handle* h, uint64_t v, uint64_t* cid_out) {
    uint64_t i = faa64(&q->T, 1ull);
    Cell* c = find_cell(q, &h->tail, static_cast<int64_t>(i));
    if (!c) {  
        *cid_out = i;
        return false;
    }
    // try to deposit directly if empty
    if (cas64(&c->val, WF_BOTTOM, v) == WF_BOTTOM) return true;
    *cid_out = i; return false;
}

// Forward decl for help_enq used by dequeue helpers
static __device__ uint64_t help_enq(wf_queue* q, wf_handle* h, Cell* c, uint64_t i);

static __device__ __forceinline__ void enq_slow(wf_queue* q, wf_handle* h, uint64_t v, uint64_t cell_id) {
    EnqReq* r = &h->enq.req;
    r->val   = v;
    r->state = pack_state(1, cell_id); // pending with original id

    Segment* tmp_tail = h->tail; // local traversal pointer
    while (true) {
        uint64_t i = faa64(&q->T, 1ull);
        Cell* c = find_cell(q, &tmp_tail, static_cast<int64_t>(i));
        if (!c) break;  
        // Dijkstra protocol: reserve enq slot, but only commit if val still empty
        if (cas_enq(&c->enq, ENQ_NULL, r) == ENQ_NULL && c->val == WF_BOTTOM) {
            (void)try_to_claim_req(&r->state, cell_id, i); // lock request to this cell (id may already be moved)
            break;
        }
        if (state_pending(r->state) == 0) break; // got helped
    }
    uint64_t final_id = state_low(r->state);
    Cell* dest = find_cell(q, &h->tail, static_cast<int64_t>(final_id));
    if (dest) enq_commit(q, dest, v, final_id);
    // enq_commit(q, dest, v, final_id);
}

// Enqueue helper invoked by dequeuers as they sweep cells at index i
static __device__ uint64_t help_enq(wf_queue* q, wf_handle* h, Cell* c, uint64_t i) {
    // Try to mark cell unusable for enqueues (⊥ -> >). If already has value, return it.
    uint64_t old = cas64(&c->val, WF_BOTTOM, WF_TOP);
    if (old != WF_BOTTOM) {
        if (old != WF_TOP) return old; // a real value deposited -> return value
        // else, already TOP, continue helping requests
    }

    // No value yet; try to help a pending enqueue request targetting <= i
    if (c->enq == ENQ_NULL) {
        // peer ring walk (at most a couple of iterations typically)
        while (true) {
            wf_handle* p = h->enq.peer;
            EnqReq* r = &p->enq.req;
            uint64_t s = r->state; // (pending,id)
            uint64_t rid = state_low(s);
            int pend = state_pending(s);

            if (h->enq.id_last != 0 && h->enq.id_last != rid) {
                // peer published a new request since last help; break to consider it
                h->enq.id_last = 0; // establish break condition next pass
            }

            if (pend && rid <= i) {
                // try to reserve this cell for peer's request
                if (cas_enq(&c->enq, ENQ_NULL, r) != ENQ_NULL) {
                    h->enq.id_last = rid; // remember we saw this id
                }
            } else {
                // can't help this peer for this cell; advance
                h->enq.peer = p->next;
            }

            // If no request recorded yet in cell, prevent other helpers from spinning forever
            if (c->enq == ENQ_NULL) {
                (void)cas_enq(&c->enq, ENQ_NULL, ENQ_TOPP);
            }
            break; // one peer consideration per call keeps forward progress
        }
    }

    // At this point enq is either a request* or >e (ENQ_TOPP)
    if (c->enq == ENQ_TOPP) {
        // No enqueue will use this cell; EMPTY only if not enough enqueues linearized before i
        uint64_t t = atomic_load_u64(&q->T); // relaxed read acceptable; linearized via advance_end when committing
        return (t <= i ? WF_EMPTY : WF_RETRY);
    }

    // enq holds a request pointer
    EnqReq* r = c->enq;
    uint64_t s = r->state; // read state before value per paper (reverse order)
    uint64_t rid = state_low(s);
    uint64_t v   = r->val;

    if (rid > i) {
        // request unsuitable for this cell; may still be EMPTY if not enough enqueues
        if (c->val == WF_TOP) {
            uint64_t t = atomic_load_u64(&q->T);
            if (t <= i) return WF_EMPTY;
        }
        return WF_RETRY;
    }

    // Try to claim the request for this cell, or commit if already claimed and cell marked TOP
    if (try_to_claim_req(&r->state, rid, i) || (r->state == pack_state(0, i) && c->val == WF_TOP)) {
        enq_commit(q, c, v, i);
    }
    return c->val; // either value or TOP
}

/*************** Dequeue fast/slow + helpers ****************/
static __device__ __forceinline__ uint64_t deq_fast(wf_queue* q, wf_handle* h, uint64_t* id_out) {
    uint64_t i = faa64(&q->H, 1ull);
    Cell* c = find_cell(q, &h->head, static_cast<int64_t>(i));
    if (!c) {  
        *id_out = i;
        return WF_EMPTY;  // Or WF_RETRY depending on semantics
    }
    uint64_t v = help_enq(q, h, c, i);
    if (v == WF_EMPTY) return WF_EMPTY; // nothing to dequeue
    if (v != WF_RETRY) {
        if (cas_deq(&c->deq, DEQ_NULL, DEQ_TOPP) == DEQ_NULL) return v;
    }
    *id_out = i; return WF_RETRY;
}

static __device__ __forceinline__ void help_deq(wf_queue* q, wf_handle* h, wf_handle* helpee) {
    DeqReq* r = &helpee->deq.req;
    uint64_t s = r->state; // (pending, idx)
    uint64_t id = r->id;
    if (!state_pending(s) || state_low(s) < id) return; // nothing to help

    // local segment pointers; hazard pointer for announced traversal base
    Segment* ha = helpee->head; // announced cells start
    h->hzdp = ha; __threadfence();

    uint64_t prior = id;
    uint64_t i = id;
    uint64_t cand = 0;

    while (true) {
        // search for a candidate if none yet and no candidate announced yet
        s = r->state;
        for (Segment* hc = ha; !cand && (state_pending(s) && state_low(s) == prior); ) {
            Cell* c = find_cell(q, &hc, static_cast<int64_t>(++i));
            if (!c) break;
            uint64_t v = help_enq(q, h, c, i);
            if (v == WF_EMPTY || (v != WF_RETRY && c->deq == DEQ_NULL)) {
                cand = i;
                break;
            } else {
                s = r->state; // re-read request state
            }
        }
        if (cand) {
            // attempt to announce our candidate
            uint64_t exp = pack_state(1, prior);
            uint64_t des = pack_state(1, cand);
            cas64(&r->state, exp, des);
            cand = 0; // either way, reset and use announced s.idx next
        }

        s = r->state; // announced candidate is s.idx
        if (!state_pending(s) || r->id != id) return; // complete or new request

        // operate on announced candidate
        uint64_t aidx = state_low(s);
        Cell* c = find_cell(q, &ha, static_cast<int64_t>(aidx));
        if (!c) {
            // h->hzdp = nullptr; __threadfence();
            // return; // out of bounds
            return; // out of bounds
        }
        if (c->val == WF_TOP || cas_deq(&c->deq, DEQ_NULL, &helpee->deq.req) == DEQ_NULL || c->deq == &helpee->deq.req) {
            // close request
            cas64(&r->state, s, pack_state(0, aidx));
            return;
        }
        // prepare next iteration
        prior = aidx;
        if (aidx >= i) { i = aidx; }
    }
}

static __device__ __forceinline__ uint64_t deq_slow(wf_queue* q, wf_handle* h, uint64_t cid) {
    DeqReq* r = &h->deq.req;
    r->id = cid;
    r->state = pack_state(1, cid); // pending at starting idx
    help_deq(q, h, h);
    uint64_t idx = state_low(r->state);
    Cell* c = find_cell(q, &h->head, static_cast<int64_t>(idx));
    if (!c) {
        advance_end_for_linearizability(&q->H, idx + 1);
        return WF_EMPTY;
    }
    uint64_t v = c->val;
    advance_end_for_linearizability(&q->H, idx + 1);
    return (v == WF_TOP ? WF_EMPTY : v);
}

/*************** Public device API ****************/
__device__ __forceinline__ void wf_enqueue(wf_queue* q, wf_handle* h, uint64_t value) {
    for (int p = WF_PATIENCE; p >= 0; --p) {
        uint64_t cid = 0;
        if (enq_fast(q, h, value, &cid)) return;
        if (p == 0) { enq_slow(q, h, value, cid); return; }
    }
}

__device__ __forceinline__ uint64_t wf_dequeue(wf_queue* q, wf_handle* h) {
    uint64_t v = WF_RETRY; uint64_t cid = 0;
    for (int p = WF_PATIENCE; p >= 0; --p) {
        v = deq_fast(q, h, &cid);
        if (v != WF_RETRY) break;
    }
    if (v == WF_RETRY) v = deq_slow(q, h, cid);

    if (v != WF_EMPTY) {
        // got a value; help peer then advance ring
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

    // If queue appears empty, do not do slow path that scans & advances H a ton
    uint64_t t = atomic_load_u64(&q->T);
    uint64_t hcur = atomic_load_u64(&q->H);
    if (t <= hcur + 1) return WF_EMPTY;
  }
  // optional: still allow a bounded slow scan (cap the distance)
  return WF_EMPTY;
}


/*************** Init kernel (not used in USE_OPT path, but provided) ****************/
__global__ void wf_init_kernel(wf_queue* q, wf_handle* handles, int num_threads) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_threads) return;
    wf_handle* h = &handles[tid];
    h->tail = q->Q;
    h->head = q->Q;
    h->next = &handles[(tid + 1) % num_threads];
    h->enq.peer = h->next;
    h->deq.peer = h->next;
    h->enq.req.val = 0; h->enq.req.state = pack_state(0, 0);
    h->deq.req.id  = 0; h->deq.req.state = pack_state(0, 0);
    h->enq.id_last = 0;
    h->hzdp = nullptr;
}

/*************** Host-side initialization ****************/
static inline void wf_queue_host_init_ex(wf_queue** d_q, wf_handle** d_handles, int num_threads, int ops_per_thread) {
    // Conservative bound: allow 3*ops per thread (dequeue search and lookahead) + safety
    const int SAFETY_CELLS = 128;
    uint64_t cells_per_thread = (uint64_t)(3 * ops_per_thread + SAFETY_CELLS);
    const uint64_t total_cells = (uint64_t)num_threads * cells_per_thread;
    uint32_t seg_needed = (uint32_t)((total_cells + WF_SEGMENT_SIZE - 1) / WF_SEGMENT_SIZE + WF_SEGMENT_SAFETY);
    if (seg_needed < 2) seg_needed = 2;

    // Allocate queue and handles on device
    hipMalloc((void**)d_q, sizeof(wf_queue));
    hipMalloc((void**)d_handles, sizeof(wf_handle) * (size_t)num_threads);

    // Allocate segments pool (managed for easy prefetch) and prelink
    wf_queue hq_host{};
    Segment* d_segments = nullptr;
    size_t seg_bytes = sizeof(Segment) * (size_t)seg_needed;
    hipMallocManaged((void**)&d_segments, seg_bytes);

    for (uint32_t i = 0; i < seg_needed; ++i) {
        Segment* s = &d_segments[i];
        s->id = (int64_t)i;
        s->next = (i + 1 < seg_needed) ? &d_segments[i + 1] : nullptr;
        for (uint32_t j = 0; j < WF_SEGMENT_SIZE; ++j) {
            s->cells[j].val = 0ull; s->cells[j].enq = (EnqReq*)0ull; s->cells[j].deq = (DeqReq*)0ull;
        }
    }

    hq_host.Q = &d_segments[0];
    hq_host.T = 0ull; hq_host.H = 0ull; hq_host.I = 0;
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

    int dev = 0; hipGetDevice(&dev);
    hipMemPrefetchAsync(d_segments, seg_bytes, dev, 0);
}

__global__ void wf_clear_pool_kernel(wf_queue* q) {
  Segment* base = q->pool.segments;
  uint32_t segs = q->pool.active_segments;   // only clear what was touched
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

  // Clear only touched segments (active_segments was grown by find_cell)
  const int block = 256;
  uint64_t total_cells = (uint64_t)hq.pool.active_segments * (uint64_t)WF_SEGMENT_SIZE;
  int grid = (int)((total_cells + block - 1) / block);
  if (grid < 1) grid = 1;
  if (grid > 65535) grid = 65535;

  wf_clear_pool_kernel<<<grid, block>>>(d_q);
  hipDeviceSynchronize();

  // Rewind indices + watermark
  hq.T = 0;
  hq.H = 0;
  hq.I = 0;
  hq.pool.active_segments = 1;
#ifdef WF_BOUNDS_CHECK
  hq.bounds_oob = 0;
#endif
  hipMemcpy(d_q, &hq, sizeof(wf_queue), hipMemcpyHostToDevice);

  // Re-init handles ring pointers etc.
  int grid2 = (num_threads + block - 1) / block;
  wf_init_kernel<<<grid2, block>>>(d_q, d_handles, num_threads);
  hipDeviceSynchronize();
}


// static inline void wf_queue_reset_for_bfs(wf_queue* d_q, wf_handle* d_handles, int num_threads) {
//     wf_queue hq;
//     hipMemcpy(&hq, d_q, sizeof(wf_queue), hipMemcpyDeviceToHost);
    
//     // Reset indices to 0 (reuse cells from beginning)
//     hq.T = 0;
//     hq.H = 0;
//     hq.I = 0;
//     hq.pool.active_segments = 1;
    
//     hipMemcpy(d_q, &hq, sizeof(wf_queue), hipMemcpyHostToDevice);
    
//     // clear cells
    
//     // Re-init handles
//     int block = 256;
//     int grid = (num_threads + block - 1) / block;
//     wf_init_kernel<<<grid, block>>>(d_q, d_handles, num_threads);
//     hipDeviceSynchronize();
// }

// Add new initialization function that takes graph size into account
static inline void wf_queue_host_init_for_bfs(wf_queue** d_q, wf_handle** d_handles, 
                                               int num_threads, long long graph_edges, int graph_vertices, int chunk_size) {
    // For BFS, we need enough cells for all vertices plus safety margin
    // Each vertex can be enqueued at most once
    printf("DEBUG: graph_edges=%lld graph_vertices=%d chunk_size=%d\n", 
           graph_edges, graph_vertices, chunk_size);
    uint64_t work_items = (graph_edges + chunk_size - 1) / chunk_size; // approximate active front size
    uint64_t min_cells_needed = (uint64_t)graph_vertices + work_items;  // safety margin
    min_cells_needed = (uint64_t)(min_cells_needed * 1.20) + 10000;
    // min_cells_needed = min_cells_needed * 4ull + 100000ull;
    
    uint32_t seg_needed = (uint32_t)((min_cells_needed + WF_SEGMENT_SIZE - 1) / WF_SEGMENT_SIZE);
    seg_needed += WF_SEGMENT_SAFETY;
    if (seg_needed < 2) seg_needed = 2;

    // Allocate queue and handles on device
    hipMalloc((void**)d_q, sizeof(wf_queue));
    hipMalloc((void**)d_handles, sizeof(wf_handle) * (size_t)num_threads);

    // Allocate segments pool
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
    hq_host.T = 0ull; hq_host.H = 0ull; hq_host.I = 0;
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

    int dev = 0; hipGetDevice(&dev);
    hipMemPrefetchAsync(d_segments, seg_bytes, dev, 0);
    
    printf("DEBUG: Allocated %u segments = %llu cells for %d vertices\n",
           seg_needed, (unsigned long long)seg_needed * WF_SEGMENT_SIZE, graph_vertices);
}

static inline void wf_queue_host_init(wf_queue** d_q, wf_handle** d_handles, int num_threads) {
    // Backward-compatible wrapper.
    wf_queue_host_init_ex(d_q, d_handles, num_threads, WF_PREALLOC_OPS_PER_THREAD);
};
//     Segment* d_segments = nullptr;
//     size_t seg_bytes = sizeof(Segment) * (size_t)seg_needed;
//     hipMallocManaged((void**)&d_segments, seg_bytes);

//     // Initialize segments on host (managed memory visible on host)
//     for (uint32_t i = 0; i < seg_needed; ++i) {
//         Segment* s = &d_segments[i];
//         s->id = (int64_t)i;
//         s->next = (i + 1 < seg_needed) ? &d_segments[i + 1] : nullptr;
//         for (uint32_t j = 0; j < WF_SEGMENT_SIZE; ++j) {
//             s->cells[j].val = 0ull; s->cells[j].enq = (EnqReq*)0ull; s->cells[j].deq = (DeqReq*)0ull;
//         }
//     }

//     // Fill queue header
//     hq_host.Q = &d_segments[0];
//     hq_host.T = 0ull; hq_host.H = 0ull; hq_host.I = 0;
//     hq_host.pool.segments = d_segments;
//     hq_host.pool.capacity = seg_needed;
//     hq_host.pool.segment_size = WF_SEGMENT_SIZE;
//     hq_host.pool.active_segments = 1u;

//     // Copy queue header to device
//     hipMemcpy(*d_q, &hq_host, sizeof(wf_queue), hipMemcpyHostToDevice);

//     // Initialize handles array via a tiny kernel for ring linkage (so pointers are device-correct)
//     int block = 256;
//     int grid = (num_threads + block - 1) / block;
//     wf_init_kernel<<<grid, block>>>(*d_q, *d_handles, num_threads);
//     hipDeviceSynchronize();

//     // Prefetch managed segments to current device to avoid first-touch stalls
//     int dev = 0; hipGetDevice(&dev);
//     hipMemPrefetchAsync(d_segments, seg_bytes, dev, 0);
// }
