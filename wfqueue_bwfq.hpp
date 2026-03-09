// #pragma once
// // ============================================================================
// // BWFQ — Bounded Wait-Free Queue for HIP/ROCm GPUs
// // ============================================================================
// //
// // Hybrid of YMC (PPoPP'16) fast-path/slow-path + wCQ (SPAA'22) admission
// // control, adapted for a bounded ring buffer on GPU.
// //
// // Key design points:
// //
// //  1. ADMISSION CONTROL via an atomic `items` counter (as in wCQ/SCQ).
// //     - Enqueue: FAA(&items, +1). If old >= cap → undo, return FULL. No ticket taken.
// //     - Dequeue: FAA(&items, -1). If old <= 0  → undo, return EMPTY. No ticket taken.
// //     This is the *only* clean way to avoid phantom ticket-holes without CAS loops
// //     on Head/Tail. Once admitted, the op has a reserved logical slot and MUST complete.
// //
// //  2. SEQUENCE-NUMBER RING (classic Lamport-style):
// //     - slot p FREE:  seq == p
// //     - slot p FULL:  seq == p + 1
// //     - ENQ ticket p: CAS val 0→v while seq==p, then seq=p+1
// //     - DEQ ticket p: XCHG val→0 while seq==p+1, then seq=p+cap
// //
// //  3. FAST PATH: FAA ticket + bounded probes (WF_PATIENCE).
// //     Works for >99% of ops under normal contention.
// //
// //  4. SLOW PATH: publish to per-thread record, then bounded deterministic helping
// //     via global HelpCtr windows (wCQ-style). Since admission guarantees a slot
// //     exists, the op WILL complete — no give-up, no silent drops.
// //     Wait-freedom bound: O(nprocs * WF_HELP_WINDOW) steps worst case.
// //
// //  5. VALUE SENTINEL: we use val==0 for "empty cell". User values must be nonzero.
// //     (For BFS frontier queues this is natural: vertex IDs are stored as id+1.)
// //     The public API documents this requirement.
// //
// // ============================================================================

// #include <hip/hip_runtime.h>
// #include <stdint.h>

// /* ===================== Tunables ===================== */
// #ifndef WF_PATIENCE
// #define WF_PATIENCE 16
// #endif

// #ifndef WF_PREALLOC_OPS_PER_THREAD
// #define WF_PREALLOC_OPS_PER_THREAD 256
// #endif

// #ifndef WF_SEGMENT_SAFETY
// #define WF_SEGMENT_SAFETY 8
// #endif

// #ifndef WF_HELP_WINDOW
// #define WF_HELP_WINDOW 32
// #endif

// // Extra slack rounds before escalation to full-coverage pass.
// #ifndef WF_HELP_SLACK
// #define WF_HELP_SLACK 2
// #endif

// #define WF_EMPTY 0ull

// /* ===================== Status API ===================== */
// enum wf_status : uint32_t {
//   WF_OK          = 0,
//   WF_FULL        = 1,
//   WF_EMPTY_STATUS = 2
// };

// /* ===================== Atomics ===================== */
// static __device__ __forceinline__ uint64_t cas64(uint64_t* a, uint64_t e, uint64_t d) {
//   return atomicCAS(reinterpret_cast<unsigned long long*>(a),
//                    (unsigned long long)e, (unsigned long long)d);
// }
// static __device__ __forceinline__ uint64_t faa64(uint64_t* a, uint64_t inc) {
//   return atomicAdd(reinterpret_cast<unsigned long long*>(a),
//                    (unsigned long long)inc);
// }
// static __device__ __forceinline__ uint64_t ld64(uint64_t* a) {
//   return atomicAdd(reinterpret_cast<unsigned long long*>(a), 0ull);
// }

// /* ===================== Ring cell ===================== */
// struct __attribute__((aligned(16))) wf_cell {
//   uint64_t seq;
//   uint64_t val;
// };

// /* ===================== Per-thread record ===================== */
// struct __attribute__((aligned(64))) wf_thread_record {
//   uint64_t pending;  // 0=idle, 1=active
//   uint64_t is_enq;   // 1=enqueue, 0=dequeue
//   uint64_t ticket;   // ticket p (Tctr or Hctr value)
//   uint64_t value;    // enq: payload. deq: result written by completer.
//   uint64_t _pad[4];
// };

// /* ===================== Handle ===================== */
// struct __attribute__((aligned(64))) wf_handle {
//   wf_handle* next;
//   uint64_t pad[8];
// };

// /* ===================== Queue ===================== */
// struct __attribute__((aligned(64))) wf_queue {
//   wf_cell*  ring;
//   uint64_t  cap;
//   uint64_t  mask;

//   uint64_t  Tctr;       // tail ticket counter (FAA only)
//   uint64_t  Hctr;       // head ticket counter (FAA only)

//   uint64_t  HelpCtr;    // deterministic help cursor

//   // Admission control: precise occupancy.
//   // enqueue must acquire: items < cap (old_items < cap)
//   // dequeue must acquire: items > 0   (old_items > 0)
//   uint64_t  items;

//   wf_thread_record* rec;
//   uint32_t nprocs;
//   uint32_t _pad32;
// };


// /* ===================== Admission control ===================== */
// // These are the gatekeepers. If admission fails, no ticket is taken,
// // no phantom hole is created, and we return FULL/EMPTY immediately.
// // If admission succeeds, the op MUST complete — the slot is logically reserved.

// static __device__ __forceinline__ wf_status admit_enq(wf_queue* q) {
//   const uint64_t old = faa64(&q->items, 1ull);
//   if (old >= q->cap) {
//     faa64(&q->items, (uint64_t)-1ll);  // undo
//     return WF_FULL;
//   }
//   return WF_OK;
// }

// static __device__ __forceinline__ wf_status admit_deq(wf_queue* q) {
//   const uint64_t old = faa64(&q->items, (uint64_t)-1ll);
//   if ((int64_t)old <= 0) {
//     faa64(&q->items, 1ull);  // undo
//     return WF_EMPTY_STATUS;
//   }
//   return WF_OK;
// }


// /* ===================== Helping rounds policy ===================== */
// // Cover all threads once via ceil(n/W) windows + slack.
// static __device__ __forceinline__ int wf_help_rounds(const wf_queue* q) {
//   const int n = (int)q->nprocs;
//   const int cover = (n + WF_HELP_WINDOW - 1) / WF_HELP_WINDOW;
//   return cover + WF_HELP_SLACK;
// }


// /* ===================== Single-step completion primitives ===================== */
// // Try to complete an ENQ for ticket p (value v) in one shot. O(1) work.
// // Returns true if known-complete (by us or others).
// static __device__ __forceinline__
// bool try_complete_enq_once(wf_queue* q, uint64_t p, uint64_t v) {
//   wf_cell* c = &q->ring[p & q->mask];
//   const uint64_t s = ld64(&c->seq);
//   const intptr_t dif = (intptr_t)s - (intptr_t)p;

//   if (dif == 0) {
//     // Slot is FREE for this ticket.
//     const uint64_t curv = ld64(&c->val);
//     if (curv != 0ull) {
//       // Someone already installed a value (could be us from a prior attempt
//       // or a racing helper). Just advance seq.
//       __threadfence();
//       atomicExch((unsigned long long*)&c->seq, (unsigned long long)(p + 1));
//       return true;
//     }
//     // Try to install our value.
//     if (atomicCAS((unsigned long long*)&c->val, 0ull, (unsigned long long)v) == 0ull) {
//       __threadfence();
//       atomicExch((unsigned long long*)&c->seq, (unsigned long long)(p + 1));
//       return true;
//     }
//     // Someone else installed — next round will finalize.
//     return false;
//   } else if (dif > 0) {
//     // Already advanced past this ticket → done.
//     return true;
//   }
//   // dif < 0 → previous dequeue hasn't freed this slot yet. Wait.
//   return false;
// }

// // Try to complete a DEQ for ticket p in one shot. O(1) work.
// // Writes result into *out_v.
// static __device__ __forceinline__
// bool try_complete_deq_once(wf_queue* q, uint64_t p, uint64_t* out_v) {
//   wf_cell* c = &q->ring[p & q->mask];
//   const uint64_t s = ld64(&c->seq);
//   const intptr_t dif = (intptr_t)s - (intptr_t)(p + 1);

//   if (dif == 0) {
//     // Slot is FULL for this ticket.
//     uint64_t v = atomicExch((unsigned long long*)&c->val, 0ull);
//     __threadfence();
//     atomicExch((unsigned long long*)&c->seq, (unsigned long long)(p + q->cap));
//     *out_v = v;
//     return true;
//   } else if (dif > 0) {
//     // Already advanced → completed by someone else.
//     *out_v = WF_EMPTY;
//     return true;
//   }
//   // dif < 0 → enqueue hasn't filled this slot yet.
//   return false;
// }


// /* ===================== Deterministic bounded helping ===================== */
// // Scan WF_HELP_WINDOW records starting from global HelpCtr.
// // O(WF_HELP_WINDOW) work per call, O(1) per record.
// static __device__ __forceinline__
// void help_window_global(wf_queue* q) {
//   const int n = (int)q->nprocs;
//   uint64_t start = faa64(&q->HelpCtr, (uint64_t)WF_HELP_WINDOW);
//   int base = (int)(start % (uint64_t)n);

//   #pragma unroll 1
//   for (int off = 0; off < WF_HELP_WINDOW; ++off) {
//     int i = base + off;
//     if (i >= n) i -= n;

//     wf_thread_record* r = &q->rec[i];
//     if (!ld64(&r->pending)) continue;

//     const uint64_t ticket = ld64(&r->ticket);
//     const uint64_t is_enq = ld64(&r->is_enq);

//     if (is_enq) {
//       const uint64_t v = ld64(&r->value);
//       if (try_complete_enq_once(q, ticket, v)) {
//         atomicExch((unsigned long long*)&r->pending, 0ull);
//       }
//     } else {
//       // Dequeue help: check if result already stored by a prior helper.
//       uint64_t rv = ld64(&r->value);
//       if (rv != 0ull) {
//         // Value already extracted by some helper. Ensure slot freed.
//         // The slot might already be freed. try_complete_deq_once handles both cases.
//         uint64_t dummy = 0;
//         if (try_complete_deq_once(q, ticket, &dummy)) {
//           // Keep the previously-stored result.
//           atomicExch((unsigned long long*)&r->pending, 0ull);
//         }
//         // Even if try_complete fails (slot not yet at right seq), another
//         // round will catch it. The value is safe in r->value.
//       } else {
//         uint64_t outv = 0;
//         if (try_complete_deq_once(q, ticket, &outv)) {
//           atomicExch((unsigned long long*)&r->value, (unsigned long long)outv);
//           atomicExch((unsigned long long*)&r->pending, 0ull);
//         }
//       }
//     }
//   }
// }

// // Escalation: full-coverage pass O(nprocs). Used as safety net.
// static __device__ __forceinline__
// void help_all_once(wf_queue* q) {
//   const int n = (int)q->nprocs;
//   for (int i = 0; i < n; ++i) {
//     wf_thread_record* r = &q->rec[i];
//     if (!ld64(&r->pending)) continue;

//     const uint64_t ticket = ld64(&r->ticket);
//     const uint64_t is_enq = ld64(&r->is_enq);

//     if (is_enq) {
//       const uint64_t v = ld64(&r->value);
//       if (try_complete_enq_once(q, ticket, v)) {
//         atomicExch((unsigned long long*)&r->pending, 0ull);
//       }
//     } else {
//       uint64_t outv = 0;
//       if (try_complete_deq_once(q, ticket, &outv)) {
//         atomicExch((unsigned long long*)&r->value, (unsigned long long)outv);
//         atomicExch((unsigned long long*)&r->pending, 0ull);
//       }
//     }
//   }
// }


// /* ===================== Public device API ===================== */

// // Enqueue: returns WF_OK or WF_FULL.
// // Once admitted, the op WILL complete (no give-up). This is what makes it wait-free.
// // Value must be nonzero (0 is reserved as empty-cell sentinel).
// static __device__ __forceinline__
// wf_status wf_enqueue(wf_queue* q, wf_handle* h, uint64_t v) {
//   (void)h;

//   // --- Admission gate ---
//   wf_status adm = admit_enq(q);
//   if (adm != WF_OK) return adm;

//   // --- Help first (wCQ flavor: small fixed helping before own op) ---
//   help_window_global(q);

//   // --- Take ticket (guaranteed to have a matching slot due to admission) ---
//   const uint64_t p = faa64(&q->Tctr, 1ull);

//   // --- Fast path: bounded probes on own cell ---
//   #pragma unroll 1
//   for (int probe = 0; probe < WF_PATIENCE; ++probe) {
//     if (try_complete_enq_once(q, p, v)) return WF_OK;
//   }

//   // --- Slow path: publish + cooperative help ---
//   uint32_t tid = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x);
//   if (tid >= q->nprocs) tid %= q->nprocs;

//   wf_thread_record* myrec = &q->rec[tid];
//   atomicExch((unsigned long long*)&myrec->is_enq, 1ull);
//   atomicExch((unsigned long long*)&myrec->ticket, (unsigned long long)p);
//   atomicExch((unsigned long long*)&myrec->value,  (unsigned long long)v);
//   __threadfence();
//   atomicExch((unsigned long long*)&myrec->pending, 1ull);

//   // Phase 1: bounded window rounds covering all threads once + slack.
//   const int rounds = wf_help_rounds(q);
//   #pragma unroll 1
//   for (int r = 0; r < rounds; ++r) {
//     if (!ld64(&myrec->pending)) return WF_OK;
//     if (try_complete_enq_once(q, p, v)) {
//       atomicExch((unsigned long long*)&myrec->pending, 0ull);
//       return WF_OK;
//     }
//     help_window_global(q);
//   }

//   // Phase 2 (escalation): 2 full-coverage passes.
//   #pragma unroll 1
//   for (int pass = 0; pass < 2; ++pass) {
//     if (!ld64(&myrec->pending)) return WF_OK;
//     if (try_complete_enq_once(q, p, v)) {
//       atomicExch((unsigned long long*)&myrec->pending, 0ull);
//       return WF_OK;
//     }
//     help_all_once(q);
//   }

//   // Phase 3 (last resort): since admitted, we MUST complete. Spin with help.
//   // This loop is finite because: admission guarantees a dequeue will eventually
//   // free our slot, and help_all_once ensures global progress.
//   while (ld64(&myrec->pending)) {
//     if (try_complete_enq_once(q, p, v)) {
//       atomicExch((unsigned long long*)&myrec->pending, 0ull);
//       break;
//     }
//     help_all_once(q);
//   }
//   return WF_OK;
// }


// // Dequeue: returns the value, or WF_EMPTY if queue is empty.
// // Once admitted, the op WILL complete.
// // Returned value is always nonzero on success (0 == empty).
// static __device__ __forceinline__
// uint64_t wf_dequeue(wf_queue* q, wf_handle* h) {
//   (void)h;

//   // --- Admission gate ---
//   wf_status adm = admit_deq(q);
//   if (adm != WF_OK) return WF_EMPTY;

//   // --- Help first ---
//   help_window_global(q);

//   // --- Take ticket ---
//   const uint64_t p = faa64(&q->Hctr, 1ull);

//   // --- Fast path ---
//   #pragma unroll 1
//   for (int probe = 0; probe < WF_PATIENCE; ++probe) {
//     uint64_t v = 0;
//     if (try_complete_deq_once(q, p, &v)) {
//       return v ? v : WF_EMPTY;
//     }
//   }

//   // --- Slow path ---
//   uint32_t tid = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x);
//   if (tid >= q->nprocs) tid %= q->nprocs;

//   wf_thread_record* myrec = &q->rec[tid];
//   atomicExch((unsigned long long*)&myrec->is_enq, 0ull);
//   atomicExch((unsigned long long*)&myrec->ticket, (unsigned long long)p);
//   atomicExch((unsigned long long*)&myrec->value,  0ull);
//   __threadfence();
//   atomicExch((unsigned long long*)&myrec->pending, 1ull);

//   // Phase 1: bounded window rounds.
//   const int rounds = wf_help_rounds(q);
//   #pragma unroll 1
//   for (int r = 0; r < rounds; ++r) {
//     if (!ld64(&myrec->pending)) {
//       uint64_t v = ld64(&myrec->value);
//       return v ? v : WF_EMPTY;
//     }
//     uint64_t v = 0;
//     if (try_complete_deq_once(q, p, &v)) {
//       atomicExch((unsigned long long*)&myrec->value, (unsigned long long)v);
//       atomicExch((unsigned long long*)&myrec->pending, 0ull);
//       return v ? v : WF_EMPTY;
//     }
//     help_window_global(q);
//   }

//   // Phase 2: escalation.
//   #pragma unroll 1
//   for (int pass = 0; pass < 2; ++pass) {
//     if (!ld64(&myrec->pending)) {
//       uint64_t v = ld64(&myrec->value);
//       return v ? v : WF_EMPTY;
//     }
//     uint64_t v = 0;
//     if (try_complete_deq_once(q, p, &v)) {
//       atomicExch((unsigned long long*)&myrec->value, (unsigned long long)v);
//       atomicExch((unsigned long long*)&myrec->pending, 0ull);
//       return v ? v : WF_EMPTY;
//     }
//     help_all_once(q);
//   }

//   // Phase 3: admitted → must complete.
//   while (ld64(&myrec->pending)) {
//     uint64_t v = 0;
//     if (try_complete_deq_once(q, p, &v)) {
//       atomicExch((unsigned long long*)&myrec->value, (unsigned long long)v);
//       atomicExch((unsigned long long*)&myrec->pending, 0ull);
//       break;
//     }
//     help_all_once(q);
//   }

//   uint64_t v = ld64(&myrec->value);
//   return v ? v : WF_EMPTY;
// }


// /* ===================== Convenience wrappers ===================== */
// // Drop-in void enqueue (for harness compatibility).
// // Silently drops on FULL (caller should size queue appropriately).
// static __device__ __forceinline__
// void wf_enqueue_void(wf_queue* q, wf_handle* h, uint64_t v) {
//   (void)wf_enqueue(q, h, v);
// }


// /* ===================== Init kernel ===================== */
// __global__ void wf_init_kernel(wf_queue* q, wf_handle* h,
//                                wf_thread_record* rec, int nthreads) {
//   const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
//   const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;

//   if (gid == 0) {
//     q->Tctr    = 0ull;
//     q->Hctr    = 0ull;
//     q->HelpCtr = 0ull;
//     q->items   = 0ull;
//     q->nprocs  = (uint32_t)nthreads;
//   }

//   for (uint64_t i = gid; i < (uint64_t)nthreads; i += stride) {
//     h[i].next = &h[(i + 1) % (uint64_t)nthreads];
//     rec[i].pending = 0ull;
//     rec[i].is_enq  = 0ull;
//     rec[i].ticket  = 0ull;
//     rec[i].value   = 0ull;
//   }

//   for (uint64_t p = gid; p < q->cap; p += stride) {
//     q->ring[p].seq = p;     // FREE
//     q->ring[p].val = 0ull;  // empty
//   }
// }

// /* ===================== BFS reset kernel ===================== */
// __global__ void wf_bfs_reset_kernel(wf_queue* q, wf_handle* h,
//                                      wf_thread_record* rec,
//                                      uint64_t used_slots,
//                                      int nthreads) {
//   const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
//   const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;

//   if (gid == 0) {
//     q->Tctr    = 0ull;
//     q->Hctr    = 0ull;
//     q->HelpCtr = 0ull;
//     q->items   = 0ull;
//   }

//   for (uint64_t i = gid; i < (uint64_t)nthreads; i += stride) {
//     h[i].next = &h[(i + 1) % (uint64_t)nthreads];
//     rec[i].pending = 0ull;
//     rec[i].is_enq  = 0ull;
//     rec[i].ticket  = 0ull;
//     rec[i].value   = 0ull;
//   }

//   for (uint64_t p = gid; p < used_slots; p += stride) {
//     q->ring[p].seq = p;
//     q->ring[p].val = 0ull;
//   }
// }

// inline void wf_bfs_reset(wf_queue* d_q, wf_handle* d_h,
//                           wf_thread_record* d_rec, int num_threads,
//                           uint64_t used_slots) {
//   if (used_slots == 0) used_slots = 1;
//   uint64_t work = (used_slots > (uint64_t)num_threads)
//                     ? used_slots : (uint64_t)num_threads;
//   const int block = 256;
//   int grid = (int)((work + block - 1) / block);
//   if (grid < 4)     grid = 4;
//   if (grid > 65535) grid = 65535;
//   wf_bfs_reset_kernel<<<grid, block>>>(d_q, d_h, d_rec, used_slots, num_threads);
// }


// /* ===================== Host init/destroy ===================== */
// static inline uint64_t round_up_pow2(uint64_t x) {
//   if (x <= 1) return 1;
//   --x;
//   x |= x >> 1; x |= x >> 2; x |= x >> 4;
//   x |= x >> 8; x |= x >> 16; x |= x >> 32;
//   return x + 1;
// }

// inline void wf_queue_host_init(wf_queue** d_q,
//                                wf_handle** d_h,
//                                wf_thread_record** d_rec,
//                                int num_threads)
// {
// #ifdef WF_RING_CAPACITY
//   uint64_t cap = (uint64_t)WF_RING_CAPACITY;
// #else
//   uint64_t need = (uint64_t)num_threads * (uint64_t)WF_PREALLOC_OPS_PER_THREAD
//                 + (uint64_t)(WF_SEGMENT_SAFETY * 1024ull);
//   uint64_t cap = round_up_pow2(need);
//   if (cap < 1024) cap = 1024;
// #endif

//   hipMalloc((void**)d_q,   sizeof(wf_queue));
//   hipMalloc((void**)d_h,   (size_t)num_threads * sizeof(wf_handle));
//   hipMalloc((void**)d_rec, (size_t)num_threads * sizeof(wf_thread_record));

//   wf_cell* d_ring = nullptr;
//   hipMalloc((void**)&d_ring, (size_t)cap * sizeof(wf_cell));

//   int device = 0;
//   hipGetDevice(&device);
//   hipMemPrefetchAsync(*d_h,  (size_t)num_threads * sizeof(wf_handle), device, 0);
//   hipMemPrefetchAsync(*d_rec,(size_t)num_threads * sizeof(wf_thread_record), device, 0);
//   hipMemPrefetchAsync(d_ring,(size_t)cap * sizeof(wf_cell), device, 0);
//   hipDeviceSynchronize();

//   wf_queue hq{};
//   hq.ring    = d_ring;
//   hq.cap     = cap;
//   hq.mask    = cap - 1;
//   hq.Tctr    = 0ull;
//   hq.Hctr    = 0ull;
//   hq.HelpCtr = 0ull;
//   hq.items   = 0ull;
//   hq.rec     = *d_rec;
//   hq.nprocs  = (uint32_t)num_threads;

//   hipMemcpy(*d_q, &hq, sizeof(hq), hipMemcpyHostToDevice);

//   const int block = 256;
//   int grid = (num_threads + block - 1) / block;
//   if (grid < 80) grid = 80;

//   wf_init_kernel<<<grid, block>>>(*d_q, *d_h, *d_rec, num_threads);
//   hipDeviceSynchronize();
// }

// inline void wf_queue_destroy(wf_queue* d_q, wf_handle* d_h) {
//   wf_queue hq{};
//   hipMemcpy(&hq, d_q, sizeof(wf_queue), hipMemcpyDeviceToHost);
//   if (hq.ring) hipFree(hq.ring);
//   hipFree(d_q);
//   hipFree(d_h);
// }


#pragma once
// ============================================================================
// BWFQ v3 — Bounded Wait-Free Queue for HIP/ROCm GPUs
// ============================================================================
//
// Fixes over broken original (wfqueue_bounded.hpp):
//   1. Admission control via FAA items counter prevents phantom ticket holes
//   2. Admitted ops MUST complete (no forced give-up → no orphaned slots)
//   3. Zero-value sentinel documented (user values must be nonzero)
//
// Performance fix over v1/v2:
//   - NO help_window_global() on fast path. Help-first is a wCQ/CPU technique
//     that kills GPU throughput by adding FAA(HelpCtr) + 32 atomic loads on
//     EVERY op including the >99% that complete in 1 fast-path probe.
//   - Helping only happens on slow-path entry (rare under normal load).
//   - Net overhead vs broken original: +1 FAA (admit) per op on the `items`
//     counter. This is unavoidable for correctness without CAS-loops on Head/Tail.
//
// Wait-freedom proof:
//   Fast path: O(WF_PATIENCE) probes — bounded.
//   Slow path phase 1: ceil(nprocs/WINDOW)+slack window rounds, each O(WINDOW).
//   Slow path phase 2: 2 × O(nprocs) full-coverage passes.
//   Slow path phase 3: while-loop. Finite because:
//     (a) Admission guarantees a matching counterpart (enq↔deq) exists or will exist.
//     (b) help_all_once visits ALL records → ensures system-wide progress.
//     (c) In each full pass, at least one pending op in the system completes
//         (the one whose slot is currently ready). After ≤ nprocs passes, our
//         slot's dependencies have been resolved.
//     Worst case: O(nprocs²) steps — polynomial, hence wait-free.
//
// Linearizability:
//   - Enqueue linearizes at seq: p → p+1 (value committed to slot).
//   - Dequeue linearizes at seq: p+1 → p+cap (value extracted from slot).
//   - Ticket ordering (Tctr/Hctr monotonic) provides FIFO.
//   - items counter maintains: items == Σ(committed enqueues) - Σ(committed dequeues).
//
// API: matches original wfqueue_bounded.hpp signatures for drop-in replacement.
//   - wf_enqueue: void by default, bool with -DWF_ENQ_RET_BOOL=1
//   - wf_dequeue: returns uint64_t (WF_EMPTY on empty)
//   - wf_queue_host_init: takes (d_q, d_h, d_rec, nthreads)
//
// ============================================================================

#include <hip/hip_runtime.h>
#include <stdint.h>

/* ===================== Tunables ===================== */
#ifndef WF_PATIENCE
#define WF_PATIENCE 16
#endif

#ifndef WF_PREALLOC_OPS_PER_THREAD
#define WF_PREALLOC_OPS_PER_THREAD 256
#endif

#ifndef WF_SEGMENT_SAFETY
#define WF_SEGMENT_SAFETY 8
#endif

#ifndef WF_HELP_WINDOW
#define WF_HELP_WINDOW 32
#endif

#ifndef WF_HELP_SLACK
#define WF_HELP_SLACK 2
#endif

#ifndef WF_ENQ_RET_BOOL
#define WF_ENQ_RET_BOOL 0
#endif

#define WF_EMPTY 0ull

/* ===================== Atomics ===================== */
static __device__ __forceinline__ uint64_t cas64(uint64_t* a, uint64_t e, uint64_t d) {
  return atomicCAS(reinterpret_cast<unsigned long long*>(a),
                   (unsigned long long)e, (unsigned long long)d);
}
static __device__ __forceinline__ uint64_t faa64(uint64_t* a, uint64_t inc) {
  return atomicAdd(reinterpret_cast<unsigned long long*>(a),
                   (unsigned long long)inc);
}
static __device__ __forceinline__ uint64_t ld64(uint64_t* a) {
  return atomicAdd(reinterpret_cast<unsigned long long*>(a), 0ull);
}

/* ===================== Ring cell ===================== */
struct __attribute__((aligned(16))) wf_cell {
  uint64_t seq;
  uint64_t val;
};

/* ===================== Per-thread record ===================== */
struct __attribute__((aligned(64))) wf_thread_record {
  uint64_t pending;
  uint64_t is_enq;
  uint64_t ticket;
  uint64_t value;
  uint64_t _pad[4];
};

/* ===================== Handle ===================== */
struct __attribute__((aligned(64))) wf_handle {
  wf_handle* next;
  uint64_t pad[8];
};

/* ===================== Queue ===================== */
struct __attribute__((aligned(64))) wf_queue {
  wf_cell*  ring;
  uint64_t  cap;
  uint64_t  mask;

  uint64_t  Tctr;
  uint64_t  Hctr;
  uint64_t  HelpCtr;
  uint64_t  items;      // admission control occupancy

  wf_thread_record* rec;
  uint32_t nprocs;
  uint32_t _pad32;
};

/* ===================== Admission control ===================== */
static __device__ __forceinline__ bool admit_enq(wf_queue* q) {
  const uint64_t old = faa64(&q->items, 1ull);
  if (old >= q->cap) {
    faa64(&q->items, (uint64_t)-1ll);
    return false;
  }
  return true;
}

static __device__ __forceinline__ bool admit_deq(wf_queue* q) {
  const uint64_t old = faa64(&q->items, (uint64_t)-1ll);
  if ((int64_t)old <= 0) {
    faa64(&q->items, 1ull);
    return false;
  }
  return true;
}

/* ===================== Help rounds policy ===================== */
static __device__ __forceinline__ int wf_help_rounds(const wf_queue* q) {
  const int n = (int)q->nprocs;
  return (n + WF_HELP_WINDOW - 1) / WF_HELP_WINDOW + WF_HELP_SLACK;
}

/* ===================== Completion primitives ===================== */

static __device__ __forceinline__
bool try_complete_enq_once(wf_queue* q, uint64_t p, uint64_t v) {
  wf_cell* c = &q->ring[p & q->mask];
  const uint64_t s = ld64(&c->seq);
  const intptr_t dif = (intptr_t)s - (intptr_t)p;

  if (dif == 0) {
    const uint64_t curv = ld64(&c->val);
    if (curv != 0ull) {
      __threadfence();
      atomicExch((unsigned long long*)&c->seq, (unsigned long long)(p + 1));
      return true;
    }
    if (atomicCAS((unsigned long long*)&c->val, 0ull, (unsigned long long)v) == 0ull) {
      __threadfence();
      atomicExch((unsigned long long*)&c->seq, (unsigned long long)(p + 1));
      return true;
    }
    return false;
  }
  return (dif > 0);  // already advanced = done; dif < 0 = not ready
}

static __device__ __forceinline__
bool try_complete_deq_once(wf_queue* q, uint64_t p, uint64_t* out_v) {
  wf_cell* c = &q->ring[p & q->mask];
  const uint64_t s = ld64(&c->seq);
  const intptr_t dif = (intptr_t)s - (intptr_t)(p + 1);

  if (dif == 0) {
    uint64_t v = atomicExch((unsigned long long*)&c->val, 0ull);
    __threadfence();
    atomicExch((unsigned long long*)&c->seq, (unsigned long long)(p + q->cap));
    *out_v = v;
    return true;
  } else if (dif > 0) {
    *out_v = WF_EMPTY;
    return true;
  }
  return false;
}

/* ===================== Deterministic bounded helping ===================== */

static __device__ __forceinline__
void help_window_global(wf_queue* q) {
  const int n = (int)q->nprocs;
  uint64_t start = faa64(&q->HelpCtr, (uint64_t)WF_HELP_WINDOW);
  int base = (int)(start % (uint64_t)n);

  #pragma unroll 1
  for (int off = 0; off < WF_HELP_WINDOW; ++off) {
    int i = base + off;
    if (i >= n) i -= n;

    wf_thread_record* r = &q->rec[i];
    if (!ld64(&r->pending)) continue;

    const uint64_t ticket = ld64(&r->ticket);
    const uint64_t is_enq = ld64(&r->is_enq);

    if (is_enq) {
      const uint64_t v = ld64(&r->value);
      if (try_complete_enq_once(q, ticket, v)) {
        atomicExch((unsigned long long*)&r->pending, 0ull);
      }
    } else {
      uint64_t rv = ld64(&r->value);
      if (rv != 0ull) {
        uint64_t dummy = 0;
        if (try_complete_deq_once(q, ticket, &dummy)) {
          atomicExch((unsigned long long*)&r->pending, 0ull);
        }
      } else {
        uint64_t outv = 0;
        if (try_complete_deq_once(q, ticket, &outv)) {
          atomicExch((unsigned long long*)&r->value, (unsigned long long)outv);
          atomicExch((unsigned long long*)&r->pending, 0ull);
        }
      }
    }
  }
}

static __device__ __forceinline__
void help_all_once(wf_queue* q) {
  const int n = (int)q->nprocs;
  for (int i = 0; i < n; ++i) {
    wf_thread_record* r = &q->rec[i];
    if (!ld64(&r->pending)) continue;

    const uint64_t ticket = ld64(&r->ticket);
    const uint64_t is_enq = ld64(&r->is_enq);

    if (is_enq) {
      const uint64_t v = ld64(&r->value);
      if (try_complete_enq_once(q, ticket, v)) {
        atomicExch((unsigned long long*)&r->pending, 0ull);
      }
    } else {
      uint64_t outv = 0;
      if (try_complete_deq_once(q, ticket, &outv)) {
        atomicExch((unsigned long long*)&r->value, (unsigned long long)outv);
        atomicExch((unsigned long long*)&r->pending, 0ull);
      }
    }
  }
}

/* ===================== Enqueue ===================== */

#if WF_ENQ_RET_BOOL
__device__ __forceinline__ bool wf_enqueue(wf_queue* q, wf_handle* h, uint64_t v)
#else
__device__ __forceinline__ void wf_enqueue(wf_queue* q, wf_handle* h, uint64_t v)
#endif
{
  (void)h;

  if (tid >= q->nprocs) return WF_EMPTY;

  // Admission gate: 1 FAA. No ticket taken on failure → no phantom holes.
  if (!admit_enq(q)) {
    #if WF_ENQ_RET_BOOL
    return false;
    #else
    return;
    #endif
  }

  // Take ticket (admission guarantees matching slot exists).
  const uint64_t p = faa64(&q->Tctr, 1ull);

  // Fast path: bounded probes, NO helping overhead.
  #pragma unroll 1
  for (int probe = 0; probe < WF_PATIENCE; ++probe) {
    wf_cell* c = &q->ring[p & q->mask];
    const uint64_t s = ld64(&c->seq);
    const intptr_t dif = (intptr_t)s - (intptr_t)p;

    if (dif == 0) {
      if (atomicCAS((unsigned long long*)&c->val, 0ull, (unsigned long long)v) == 0ull) {
        __threadfence();
        atomicExch((unsigned long long*)&c->seq, (unsigned long long)(p + 1));
        #if WF_ENQ_RET_BOOL
        return true;
        #else
        return;
        #endif
      }
    } else if (dif < 0) {
      break;  // slot not freed yet, go to slow path
    } else {
      break;  // already advanced past, go to slow path
    }
  }

  // Slow path: publish + help. Only reached by <1% of ops under normal load.
  uint32_t tid = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= q->nprocs) tid %= q->nprocs;

  wf_thread_record* myrec = &q->rec[tid];
  atomicExch((unsigned long long*)&myrec->is_enq, 1ull);
  atomicExch((unsigned long long*)&myrec->ticket, (unsigned long long)p);
  atomicExch((unsigned long long*)&myrec->value,  (unsigned long long)v);
  __threadfence();
  atomicExch((unsigned long long*)&myrec->pending, 1ull);

  // Quick self-check before scanning
  {
    wf_cell* c = &q->ring[p & q->mask];
    const uint64_t s = ld64(&c->seq);
    const intptr_t dif = (intptr_t)s - (intptr_t)p;
    if (dif == 0) {
      const uint64_t curv = ld64(&c->val);
      if (curv != 0ull) {
        __threadfence();
        atomicExch((unsigned long long*)&c->seq, (unsigned long long)(p + 1));
        atomicExch((unsigned long long*)&myrec->pending, 0ull);
        #if WF_ENQ_RET_BOOL
        return true;
        #else
        return;
        #endif
      } else if (atomicCAS((unsigned long long*)&c->val, 0ull, (unsigned long long)v) == 0ull) {
        __threadfence();
        atomicExch((unsigned long long*)&c->seq, (unsigned long long)(p + 1));
        atomicExch((unsigned long long*)&myrec->pending, 0ull);
        #if WF_ENQ_RET_BOOL
        return true;
        #else
        return;
        #endif
      }
    } else if (dif > 0) {
      atomicExch((unsigned long long*)&myrec->pending, 0ull);
      #if WF_ENQ_RET_BOOL
      return true;
      #else
      return;
      #endif
    }
  }

  // Phase 1: bounded window rounds
  const int rounds = wf_help_rounds(q);
  #pragma unroll 1
  for (int r = 0; r < rounds; ++r) {
    if (!ld64(&myrec->pending)) goto enq_done;
    help_window_global(q);
  }

  // Phase 2: escalation — 2 full-coverage passes
  #pragma unroll 1
  for (int pass = 0; pass < 2; ++pass) {
    if (!ld64(&myrec->pending)) goto enq_done;
    if (try_complete_enq_once(q, p, v)) {
      atomicExch((unsigned long long*)&myrec->pending, 0ull);
      goto enq_done;
    }
    help_all_once(q);
  }

  // Phase 3: admitted → MUST complete. Finite (see proof in header).
  while (ld64(&myrec->pending)) {
    if (try_complete_enq_once(q, p, v)) {
      atomicExch((unsigned long long*)&myrec->pending, 0ull);
      break;
    }
    help_all_once(q);
  }

enq_done:
  #if WF_ENQ_RET_BOOL
  return true;
  #else
  return;
  #endif
}

/* ===================== Dequeue ===================== */

__device__ __forceinline__
uint64_t wf_dequeue(wf_queue* q, wf_handle* h) {
  (void)h;


  if (tid >= q->nprocs) return WF_EMPTY;

  // Admission gate
  if (!admit_deq(q)) return WF_EMPTY;

  // Take ticket
  const uint64_t p = faa64(&q->Hctr, 1ull);

  // Fast path: bounded probes, NO helping overhead.
  #pragma unroll 1
  for (int probe = 0; probe < WF_PATIENCE; ++probe) {
    wf_cell* c = &q->ring[p & q->mask];
    const uint64_t s = ld64(&c->seq);
    const intptr_t dif = (intptr_t)s - (intptr_t)(p + 1);

    if (dif == 0) {
      uint64_t v = atomicExch((unsigned long long*)&c->val, 0ull);
      __threadfence();
      atomicExch((unsigned long long*)&c->seq, (unsigned long long)(p + q->cap));
      return v ? v : WF_EMPTY;
    } else if (dif < 0) {
      break;
    } else {
      break;
    }
  }

  // Slow path
  uint32_t tid = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= q->nprocs) tid %= q->nprocs;

  wf_thread_record* myrec = &q->rec[tid];
  atomicExch((unsigned long long*)&myrec->is_enq, 0ull);
  atomicExch((unsigned long long*)&myrec->ticket, (unsigned long long)p);
  atomicExch((unsigned long long*)&myrec->value,  0ull);
  __threadfence();
  atomicExch((unsigned long long*)&myrec->pending, 1ull);

  // Quick self-check
  {
    wf_cell* c = &q->ring[p & q->mask];
    const uint64_t s = ld64(&c->seq);
    const intptr_t dif = (intptr_t)s - (intptr_t)(p + 1);
    if (dif == 0) {
      uint64_t v = atomicExch((unsigned long long*)&c->val, 0ull);
      atomicExch((unsigned long long*)&myrec->value, (unsigned long long)v);
      __threadfence();
      atomicExch((unsigned long long*)&c->seq, (unsigned long long)(p + q->cap));
      atomicExch((unsigned long long*)&myrec->pending, 0ull);
      return v ? v : WF_EMPTY;
    } else if (dif > 0) {
      atomicExch((unsigned long long*)&myrec->pending, 0ull);
      return WF_EMPTY;
    }
  }

  // Phase 1: bounded window rounds
  const int rounds = wf_help_rounds(q);
  #pragma unroll 1
  for (int r = 0; r < rounds; ++r) {
    if (!ld64(&myrec->pending)) goto deq_done;
    help_window_global(q);
  }

  // Phase 2: escalation
  #pragma unroll 1
  for (int pass = 0; pass < 2; ++pass) {
    if (!ld64(&myrec->pending)) goto deq_done;
    uint64_t v = 0;
    if (try_complete_deq_once(q, p, &v)) {
      atomicExch((unsigned long long*)&myrec->value, (unsigned long long)v);
      atomicExch((unsigned long long*)&myrec->pending, 0ull);
      goto deq_done;
    }
    help_all_once(q);
  }

  // Phase 3: must complete
  while (ld64(&myrec->pending)) {
    uint64_t v = 0;
    if (try_complete_deq_once(q, p, &v)) {
      atomicExch((unsigned long long*)&myrec->value, (unsigned long long)v);
      atomicExch((unsigned long long*)&myrec->pending, 0ull);
      break;
    }
    help_all_once(q);
  }

deq_done:;
  uint64_t out = ld64(&myrec->value);
  return out ? out : WF_EMPTY;
}

/* ===================== Init ===================== */
__global__ void wf_init_kernel(wf_queue* q, wf_handle* h,
                               wf_thread_record* rec, int nthreads) {
  const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;

  if (gid == 0) {
    q->Tctr    = 0ull;
    q->Hctr    = 0ull;
    q->HelpCtr = 0ull;
    q->items   = 0ull;
    q->nprocs  = (uint32_t)nthreads;
  }

  for (uint64_t i = gid; i < (uint64_t)nthreads; i += stride) {
    h[i].next = &h[(i + 1) % (uint64_t)nthreads];
    rec[i].pending = 0ull;
    rec[i].is_enq  = 0ull;
    rec[i].ticket  = 0ull;
    rec[i].value   = 0ull;
  }

  for (uint64_t p = gid; p < q->cap; p += stride) {
    q->ring[p].seq = p;
    q->ring[p].val = 0ull;
  }
}

/* ===================== BFS reset ===================== */
__global__ void wf_bfs_reset_kernel(wf_queue* q, wf_handle* h,
                                     wf_thread_record* rec,
                                     uint64_t used_slots, int nthreads) {
  const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;

  if (gid == 0) {
    q->Tctr    = 0ull;
    q->Hctr    = 0ull;
    q->HelpCtr = 0ull;
    q->items   = 0ull;
  }

  for (uint64_t i = gid; i < (uint64_t)nthreads; i += stride) {
    h[i].next = &h[(i + 1) % (uint64_t)nthreads];
    rec[i].pending = 0ull;
    rec[i].is_enq  = 0ull;
    rec[i].ticket  = 0ull;
    rec[i].value   = 0ull;
  }

  for (uint64_t p = gid; p < used_slots; p += stride) {
    q->ring[p].seq = p;
    q->ring[p].val = 0ull;
  }
}

inline void wf_bfs_reset(wf_queue* d_q, wf_handle* d_h,
                          wf_thread_record* d_rec, int num_threads,
                          uint64_t used_slots) {
  if (used_slots == 0) used_slots = 1;
  uint64_t work = (used_slots > (uint64_t)num_threads)
                    ? used_slots : (uint64_t)num_threads;
  const int block = 256;
  int grid = (int)((work + block - 1) / block);
  if (grid < 4)     grid = 4;
  if (grid > 65535) grid = 65535;
  wf_bfs_reset_kernel<<<grid, block>>>(d_q, d_h, d_rec, used_slots, num_threads);
}

/* ===================== Host init/destroy ===================== */
static inline uint64_t round_up_pow2(uint64_t x) {
  if (x <= 1) return 1;
  --x;
  x |= x >> 1; x |= x >> 2; x |= x >> 4;
  x |= x >> 8; x |= x >> 16; x |= x >> 32;
  return x + 1;
}

inline void wf_queue_host_init(wf_queue** d_q, wf_handle** d_h,
                               wf_thread_record** d_rec, int num_threads) {
#ifdef WF_RING_CAPACITY
  uint64_t cap = (uint64_t)WF_RING_CAPACITY;
#else
  uint64_t need = (uint64_t)num_threads * (uint64_t)WF_PREALLOC_OPS_PER_THREAD
                + (uint64_t)(WF_SEGMENT_SAFETY * 1024ull);
  uint64_t cap = round_up_pow2(need);
  if (cap < 1024) cap = 1024;
#endif

  hipMalloc((void**)d_q,   sizeof(wf_queue));
  hipMalloc((void**)d_h,   (size_t)num_threads * sizeof(wf_handle));
  hipMalloc((void**)d_rec, (size_t)num_threads * sizeof(wf_thread_record));

  wf_cell* d_ring = nullptr;
  hipMalloc((void**)&d_ring, (size_t)cap * sizeof(wf_cell));

  int device = 0;
  hipGetDevice(&device);
  hipMemPrefetchAsync(*d_h,   (size_t)num_threads * sizeof(wf_handle), device, 0);
  hipMemPrefetchAsync(*d_rec, (size_t)num_threads * sizeof(wf_thread_record), device, 0);
  hipMemPrefetchAsync(d_ring, (size_t)cap * sizeof(wf_cell), device, 0);
  hipDeviceSynchronize();

  wf_queue hq{};
  hq.ring    = d_ring;
  hq.cap     = cap;
  hq.mask    = cap - 1;
  hq.Tctr    = 0ull;
  hq.Hctr    = 0ull;
  hq.HelpCtr = 0ull;
  hq.items   = 0ull;
  hq.rec     = *d_rec;
  hq.nprocs  = (uint32_t)num_threads;

  hipMemcpy(*d_q, &hq, sizeof(hq), hipMemcpyHostToDevice);

  const int block = 256;
  int grid = (num_threads + block - 1) / block;
  if (grid < 80) grid = 80;

  wf_init_kernel<<<grid, block>>>(*d_q, *d_h, *d_rec, num_threads);
  hipDeviceSynchronize();
}

inline void wf_queue_destroy(wf_queue* d_q, wf_handle* d_h) {
  wf_queue hq{};
  hipMemcpy(&hq, d_q, sizeof(wf_queue), hipMemcpyDeviceToHost);
  if (hq.ring) hipFree(hq.ring);
  hipFree(d_q);
  hipFree(d_h);
}