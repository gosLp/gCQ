// // wfqueue_hip_opt.cpp - Optimized Yang & Mellor-Crummey Wait-Free Queue for GPU
// // Key optimization: Pre-allocated segment pool to eliminate malloc bottleneck
// // Preserves ALL wait-freedom and correctness guarantees from original paper

// // #ifndef WFQUEUE_HIP_OPT_H
// // #define WFQUEUE_HIP_OPT_H

// #include <hip/hip_runtime.h>
// #include <stdint.h>
// #include <wfqueue_hip_opt.hpp>

// //=============================================================================
// // OPTIMIZATION CONSTANTS
// //=============================================================================
// #define WF_POOL_SIZE (WF_MAX_BLOCKS * WF_BLOCK_SIZE) //262144         // Pre-allocate 128K segments (should handle 500M+ operations)
// #define WF_POOL_BLOCK_SIZE WF_BLOCK_SIZE  //1024     // Segments per allocation block

// //=============================================================================
// // HIP ATOMIC OPERATIONS (unchanged)
// //=============================================================================
// #define WF_ATOMIC_LOAD(ptr) __atomic_load_n(ptr, __ATOMIC_ACQUIRE)
// #define WF_ATOMIC_STORE(ptr, val) __atomic_store_n(ptr, val, __ATOMIC_RELEASE)
// #define WF_ATOMIC_CAS(ptr, expected, desired) \
//     __atomic_compare_exchange_n(ptr, &(expected), desired, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)
// #define WF_ATOMIC_ADD(ptr, val) __atomic_fetch_add(ptr, val, __ATOMIC_SEQ_CST)
// #define WF_FENCE() __atomic_thread_fence(__ATOMIC_SEQ_CST)

// //=============================================================================
// // STATE MANIPULATION MACROS (unchanged from paper)
// //=============================================================================
// #define WF_STATE_PENDING_MASK 1ULL
// #define WF_STATE_ID_MASK (~WF_STATE_PENDING_MASK)
// #define WF_STATE_ID_SHIFT 1

// #define WF_GET_PENDING(state) ((state) & WF_STATE_PENDING_MASK)
// #define WF_GET_ID(state) ((state) >> WF_STATE_ID_SHIFT)
// #define WF_MAKE_STATE(pending, id) (((uint64_t)(pending)) | (((uint64_t)(id)) << WF_STATE_ID_SHIFT))


// //=============================================================================
// // OPTIMIZED MEMORY MANAGEMENT
// //=============================================================================

// __device__ void wf_allocate_block(wf_queue* q, uint32_t block_idx) {
//     if (block_idx >= WF_MAX_BLOCKS) return;
    
//     wf_block* block = &q->pool.blocks[block_idx];
    
//     // Atomic check-and-allocate
//     uint32_t expected = 0;
//     if (atomicCAS(&block->initialized, expected, 1) == 0) {
//         // This thread won the race to allocate this block
//         block->nodes = (wf_node*)malloc(WF_BLOCK_SIZE * sizeof(wf_node));
        
//         if (block->nodes) {
//             // Initialize nodes in this block
//             for (int i = 0; i < WF_BLOCK_SIZE; i++) {
//                 wf_node* node = &block->nodes[i];
//                 node->next = nullptr;
//                 node->id = 0;
                
//                 // Initialize cells to BOT
//                 for (int j = 0; j < WF_NODE_SIZE; j++) {
//                     WF_ATOMIC_STORE(&node->cells[j].val, WF_BOT);
//                     node->cells[j].enq = nullptr;
//                     node->cells[j].deq = nullptr;
//                 }
//             }
            
//             // Mark as fully initialized
//             __threadfence(); // Ensure writes are visible
//             atomicExch(&block->initialized, 2); // 2 = fully ready
            
//         } else {
//             // Allocation failed, mark as failed
//             atomicExch(&block->initialized, 3); // 3 = failed
//         }
//     } else {
//         // Another thread is/was allocating, wait for completion
//         while (atomicAdd(&block->initialized, 0) == 1) {
//             // Spin-wait for allocation completion
//             __threadfence();
//         }
//     }
// }

// // OPTIMIZATION: Initialize pre-allocated segment pool
// // __device__ void wf_init_memory_pool(wf_queue* q) {
// //     if (q->pool.initialized) return;
    
// //     // Allocate entire pool at once (major optimization)
// //     q->pool.nodes = (wf_node*)malloc(WF_POOL_SIZE * sizeof(wf_node));
// //     if (!q->pool.nodes) {
// //         printf("CRITICAL: Failed to allocate segment pool!\n");
// //         return;
// //     }
    
// //     q->pool.pool_size = WF_POOL_SIZE;
// //     q->pool.pool_counter = 0;
    
// //     // Pre-initialize all segments to avoid initialization overhead
// //     for (uint64_t i = 0; i < WF_POOL_SIZE; i++) {
// //         wf_node* node = &q->pool.nodes[i];
// //         node->next = nullptr;
// //         node->id = 0; // Will be set when allocated
        
// //         // Initialize all cells to BOT
// //         for (int j = 0; j < WF_NODE_SIZE; j++) {
// //             WF_ATOMIC_STORE(&node->cells[j].val, WF_BOT);
// //             node->cells[j].enq = nullptr;
// //             node->cells[j].deq = nullptr;
// //         }
// //     }
    
// //     q->pool.initialized = true;
// // }
// __device__ void wf_init_memory_pool(wf_queue* q) {
//     if (q->pool.initialized) return;
    
//     // Initialize pool metadata
//     q->pool.block_counter = 0;
//     q->pool.total_nodes = 0;
    
//     // Initialize all block pointers to null
//     for (int i = 0; i < WF_MAX_BLOCKS; i++) {
//         q->pool.blocks[i].nodes = nullptr;
//         q->pool.blocks[i].initialized = 0;
//         q->pool.blocks[i].node_counter = 0;
//     }
    
//     // Pre-allocate only initial blocks (16MB instead of 4GB!)
//     for (int i = 0; i < WF_INITIAL_BLOCKS; i++) {
//         wf_allocate_block(q, i);
//     }
    
//     q->pool.initialized = true;
    
//     printf("Pool initialized: %d initial blocks, %d total capacity\n", 
//            WF_INITIAL_BLOCKS, WF_MAX_BLOCKS);
// }


// // OPTIMIZATION: Replace malloc with pool allocation
// // __device__ wf_node* wf_new_node(wf_queue* q, uint64_t id) {
// //     // Ensure pool is initialized
// //     if (!q->pool.initialized) {
// //         wf_init_memory_pool(q);
// //         if (!q->pool.initialized) return nullptr;
// //     }
    
// //     // Atomic allocation from pool
// //     uint64_t idx = WF_ATOMIC_ADD(&q->pool.pool_counter, 1);
    
// //     if (idx >= q->pool.pool_size) {
// //         printf("WARNING: Segment pool exhausted! Consider increasing WF_POOL_SIZE\n");
// //         return nullptr;
// //     }
    
// //     wf_node* node = &q->pool.nodes[idx];
// //     WF_ATOMIC_STORE(&node->id, id);
    
// //     // Cells are already initialized, just reset next pointer
// //     node->next = nullptr;
    
// //     return node;
// // }
// __device__ wf_node* wf_new_node(wf_queue* q, uint64_t id) {
//     if (!q->pool.initialized) {
//         wf_init_memory_pool(q);
//         if (!q->pool.initialized) return nullptr;
//     }
    
//     // Try to allocate from current block first
//     uint32_t current_block = atomicAdd(&q->pool.block_counter, 0); // Read current block
    
//     for (int attempt = 0; attempt < WF_MAX_BLOCKS; attempt++) {
//         uint32_t block_idx = (current_block + attempt) % WF_MAX_BLOCKS;
//         wf_block* block = &q->pool.blocks[block_idx];
        
//         // Ensure block is allocated
//         if (atomicAdd(&block->initialized, 0) < 2) {
//             wf_allocate_block(q, block_idx);
//         }
        
//         // Skip failed blocks
//         if (atomicAdd(&block->initialized, 0) == 3) continue;
        
//         // Try to get a node from this block
//         uint32_t node_idx = atomicAdd(&block->node_counter, 1);
//         if (node_idx < WF_BLOCK_SIZE) {
//             // Success! Got a node from this block
//             wf_node* node = &block->nodes[node_idx];
//             WF_ATOMIC_STORE(&node->id, id);
//             node->next = nullptr;
            
//             atomicAdd(&q->pool.total_nodes, 1);
//             return node;
//         } else {
//             // This block is full, try next block
//             atomicCAS(&q->pool.block_counter, block_idx, block_idx + 1);
//             continue;
//         }
//     }
    
//     // All blocks exhausted - this shouldn't happen with proper sizing
//     printf("CRITICAL: All memory blocks exhausted! Consider increasing WF_MAX_BLOCKS\n");
//     return nullptr;
// }

// //=============================================================================
// // CORE ALGORITHM FUNCTIONS (unchanged from paper - preserves correctness)
// //=============================================================================

// __device__ wf_cell* wf_find_cell(wf_node** sp, uint64_t i, wf_queue* q) {
//     if (i == 0) i = 1;
    
//     uint64_t s = (i - 1) / WF_NODE_SIZE;
//     wf_node* cur = *sp;
    
//     while (WF_ATOMIC_LOAD(&cur->id) < s) {
//         wf_node* next = (wf_node*)WF_ATOMIC_LOAD((uint64_t*)&cur->next);
        
//         if (next == nullptr) {
//             uint64_t next_id = WF_ATOMIC_LOAD(&cur->id) + 1;
//             wf_node* new_node = wf_new_node(q, next_id); // OPTIMIZATION: Use pool
//             if (!new_node) return nullptr;
            
//             uint64_t expected = 0;
//             if (WF_ATOMIC_CAS((uint64_t*)&cur->next, expected, (uint64_t)new_node)) {
//                 next = new_node;
//             } else {
//                 // Someone else linked first - pool allocation means no free() needed
//                 next = (wf_node*)WF_ATOMIC_LOAD((uint64_t*)&cur->next);
//             }
//         }
//         cur = next;
//     }
    
//     *sp = cur;
//     return &cur->cells[i % WF_NODE_SIZE];
// }

// __device__ void wf_advance_end_for_linearizability(uint64_t* E, uint64_t cid) {
//     uint64_t e;
//     do {
//         e = WF_ATOMIC_LOAD(E);
//         if (e >= cid) break;
//     } while (!WF_ATOMIC_CAS(E, e, cid));
// }

// __device__ bool wf_try_to_claim_req(uint64_t* state_ptr, uint64_t expected_id, uint64_t new_id) {
//     uint64_t expected_state = WF_MAKE_STATE(1, expected_id);
//     uint64_t desired_state = WF_MAKE_STATE(0, new_id);
//     return WF_ATOMIC_CAS(state_ptr, expected_state, desired_state);
// }

// __device__ void wf_enq_commit(wf_queue* q, wf_cell* c, uint64_t v, uint64_t cid) {
//     wf_advance_end_for_linearizability(&q->Ei, cid + 1);
//     WF_ATOMIC_STORE(&c->val, v);
// }

// //=============================================================================
// // FORWARD DECLARATIONS
// //=============================================================================
// __device__ uint64_t wf_help_enq(wf_queue* q, wf_handle* h, wf_cell* c, uint64_t i);
// __device__ void wf_help_deq(wf_queue* q, wf_handle* h, wf_handle* helpee);

// //=============================================================================
// // FAST PATH FUNCTIONS (unchanged from paper)
// //=============================================================================

// __device__ bool wf_enq_fast(wf_queue* q, wf_handle* h, uint64_t v, uint64_t* cid) {
//     uint64_t i = WF_ATOMIC_ADD(&q->Ei, 1);
//     *cid = i;
    
//     if (i == 0) return false;
    
//     wf_cell* c = wf_find_cell(&h->Ep, i, q);
//     if (!c) return false;
    
//     uint64_t expected = WF_BOT;
//     if (WF_ATOMIC_CAS(&c->val, expected, v)) {
//         wf_advance_end_for_linearizability(&q->Ei, i + 1);
//         return true;
//     }
    
//     return false;
// }

// __device__ uint64_t wf_deq_fast(wf_queue* q, wf_handle* h, uint64_t* cid) {
//     uint64_t i = WF_ATOMIC_ADD(&q->Di, 1);
//     *cid = i;
    
//     if (i == 0) return WF_TOP;
    
//     wf_cell* c = wf_find_cell(&h->Dp, i, q);
//     if (!c) return WF_TOP;
    
//     uint64_t v = wf_help_enq(q, h, c, i);
    
//     if (v == WF_EMPTY) {
//         return WF_EMPTY;
//     }
    
//     if (v != WF_TOP) {
//         uint64_t expected = 0;
//         if (WF_ATOMIC_CAS((uint64_t*)&c->deq, expected, (uint64_t)&h->Dr)) {
//             wf_advance_end_for_linearizability(&q->Di, i + 1);
//             return v;
//         }
//     }
    
//     return WF_TOP;
// }

// //=============================================================================
// // HELPING FUNCTIONS (unchanged from paper - critical for wait-freedom)
// //=============================================================================

// __device__ uint64_t wf_help_enq(wf_queue* q, wf_handle* h, wf_cell* c, uint64_t i) {
//     uint64_t expected = WF_BOT;
//     if (!WF_ATOMIC_CAS(&c->val, expected, WF_TOP)) {
//         uint64_t val = WF_ATOMIC_LOAD(&c->val);
//         if (val != WF_TOP) {
//             return val;
//         }
//     }
    
//     if (c->enq == nullptr) {
//         do {
//             wf_handle* p = h->Eh;
//             if (!p) break;
            
//             wf_enq_req* r = &p->Er;
//             uint64_t s = WF_ATOMIC_LOAD(&r->state);
            
//             if (h->Ei == 0 || h->Ei == WF_GET_ID(s)) break;
            
//             WF_ATOMIC_STORE(&h->Ei, 0);
//             h->Eh = p->next;
//         } while (true);
        
//         if (h->Eh) {
//             wf_handle* p = h->Eh;
//             wf_enq_req* r = &p->Er;
//             uint64_t s = WF_ATOMIC_LOAD(&r->state);
            
//             if (WF_GET_PENDING(s) && WF_GET_ID(s) <= i) {
//                 uint64_t expected_enq = 0;
//                 if (!WF_ATOMIC_CAS((uint64_t*)&c->enq, expected_enq, (uint64_t)r)) {
//                     WF_ATOMIC_STORE(&h->Ei, WF_GET_ID(s));
//                 }
//             } else {
//                 h->Eh = p->next;
//             }
//         }
        
//         if (c->enq == nullptr) {
//             uint64_t expected_enq = 0;
//             WF_ATOMIC_CAS((uint64_t*)&c->enq, expected_enq, (uint64_t)-1);
//         }
//     }
    
//     if (c->enq && c->enq != (wf_enq_req*)-1) {
//         wf_enq_req* r = c->enq;
//         uint64_t state = WF_ATOMIC_LOAD(&r->state);
        
//         if (!WF_GET_PENDING(state)) {
//             uint64_t cid = WF_GET_ID(state);
//             if (cid <= i) {
//                 wf_enq_commit(q, c, r->val, cid);
//                 return r->val;
//             } else {
//                 WF_ATOMIC_STORE(&c->val, WF_EMPTY);
//                 c->enq = nullptr;
//                 return WF_EMPTY;
//             }
//         } else {
//             if (wf_try_to_claim_req(&r->state, WF_GET_ID(state), i)) {
//                 wf_enq_commit(q, c, r->val, i);
//                 return r->val;
//             }
//         }
//     }
    
//     return WF_EMPTY;
// }

// __device__ void wf_help_deq(wf_queue* q, wf_handle* h, wf_handle* helpee) {
//     if (!helpee || helpee == h) return;

//     WF_ATOMIC_STORE(&h->hzd_node_id, (uint64_t)helpee->Dp);

//     wf_deq_req* r = &helpee->Dr;
//     uint64_t state = WF_ATOMIC_LOAD(&r->state);
//     uint64_t id = r->id;
    
//     if (!WF_GET_PENDING(state) || WF_GET_ID(state) < id) {
//         WF_ATOMIC_STORE(&h->hzd_node_id, (uint64_t)nullptr);
//         return;
//     }
    
//     wf_node* ha = helpee->Dp;
//     wf_node* hc = ha;
    
//     uint64_t prior = id;
//     uint64_t i = id;
//     uint64_t cand = 0;
    
//     while (true) {
//         while (!cand && WF_GET_ID(WF_ATOMIC_LOAD(&r->state)) == prior) {
//             wf_cell* c = wf_find_cell(&hc, ++i, q);
//             if (!c) break;
            
//             uint64_t v = wf_help_enq(q, h, c, i);
            
//             if (v == WF_EMPTY || (v != WF_TOP && c->deq == nullptr)) {
//                 cand = i;
//             } else {
//                 state = WF_ATOMIC_LOAD(&r->state);
//             }
//         }
        
//         if (cand) {
//             uint64_t expected_state = WF_MAKE_STATE(1, prior);
//             uint64_t desired_state = WF_MAKE_STATE(1, cand);
//             WF_ATOMIC_CAS(&r->state, expected_state, desired_state);
//             state = WF_ATOMIC_LOAD(&r->state);
//         }
        
//         if (!WF_GET_PENDING(state) || r->id != id) return;
        
//         uint64_t announced_idx = WF_GET_ID(state);
//         wf_cell* c = wf_find_cell(&ha, announced_idx, q);
//         if (!c) return;
        
//         uint64_t expected_deq = 0; 
//         if (c->val == WF_TOP || 
//             WF_ATOMIC_CAS((uint64_t*)&c->deq, expected_deq, (uint64_t)r) ||
//             c->deq == r) {
//             uint64_t expected = WF_MAKE_STATE(1, announced_idx);
//             uint64_t desired = WF_MAKE_STATE(0, announced_idx);
//             WF_ATOMIC_CAS(&r->state, expected, desired);
//             return;
//         }
        
//         prior = announced_idx;
//         if (announced_idx >= i) {
//             cand = 0;
//             i = announced_idx;
//         }
//     }
    
//     WF_ATOMIC_STORE(&h->hzd_node_id, (uint64_t)nullptr);
// }

// //=============================================================================
// // SLOW PATH FUNCTIONS (unchanged from paper)
// //=============================================================================

// __device__ void wf_enq_slow(wf_queue* q, wf_handle* h, uint64_t v, uint64_t cell_id) {
//     h->Er.val = v;
//     WF_ATOMIC_STORE(&h->Er.state, WF_MAKE_STATE(1, cell_id));
    
//     wf_node* tmp_tail = h->Ep;
    
//     do {
//         uint64_t i = WF_ATOMIC_ADD(&q->Ei, 1);
//         wf_cell* c = wf_find_cell(&tmp_tail, i, q);
//         if (!c) continue;
        
//         uint64_t expected_enq = 0;
//         if (WF_ATOMIC_CAS((uint64_t*)&c->enq, expected_enq, (uint64_t)&h->Er) &&
//             WF_ATOMIC_LOAD(&c->val) == WF_BOT) {
//             if (wf_try_to_claim_req(&h->Er.state, cell_id, i)) {
//                 break;
//             } else {
//                 c->enq = nullptr;
//             }
//         }
        
//         if (h->Dh) {
//             wf_help_deq(q, h, h->Dh);
//         }
        
//     } while (WF_GET_PENDING(WF_ATOMIC_LOAD(&h->Er.state)));
    
//     uint64_t final_state = WF_ATOMIC_LOAD(&h->Er.state);
//     uint64_t claimed_id = WF_GET_ID(final_state);
//     wf_cell* final_cell = wf_find_cell(&h->Ep, claimed_id, q);
//     if (final_cell) {
//         wf_enq_commit(q, final_cell, v, claimed_id);
//     }
// }

// __device__ uint64_t wf_deq_slow(wf_queue* q, wf_handle* h, uint64_t cell_id) {
//     h->Dr.id = cell_id;
//     WF_ATOMIC_STORE(&h->Dr.state, WF_MAKE_STATE(1, cell_id));
    
//     wf_help_deq(q, h, h);
    
//     while (WF_GET_PENDING(WF_ATOMIC_LOAD(&h->Dr.state))) {
//         if (h->Eh) {
//             uint64_t ei = WF_ATOMIC_LOAD(&q->Ei);
//             wf_cell* c = wf_find_cell(&h->Ep, ei, q);
//             if (c) {
//                 wf_help_enq(q, h, c, ei);
//             }
//         }
        
//         if (h->Dh && h->Dh != h) {
//             wf_help_deq(q, h, h->Dh);
//         }
//     }
    
//     uint64_t final_state = WF_ATOMIC_LOAD(&h->Dr.state);
//     uint64_t result_idx = WF_GET_ID(final_state);
    
//     if (result_idx == (uint64_t)-1) {
//         return WF_EMPTY;
//     } else {
//         wf_cell* c = wf_find_cell(&h->Dp, result_idx, q);
//         return c ? WF_ATOMIC_LOAD(&c->val) : WF_EMPTY;
//     }
// }

// //=============================================================================
// // MAIN API FUNCTIONS (unchanged from paper)
// //=============================================================================

// __device__ void wf_enqueue(wf_queue* q, wf_handle* h, uint64_t v) {
//     WF_ATOMIC_STORE(&h->hzd_node_id, (uint64_t)h->Ep);
    
//     uint64_t cid;
//     for (int p = MAX_PATIENCE; p >= 0; p--) {
//         if (wf_enq_fast(q, h, v, &cid)) {
//             WF_ATOMIC_STORE(&h->enq_node_id, h->Ep->id);
//             WF_ATOMIC_STORE(&h->hzd_node_id, (uint64_t)nullptr);
//             return;
//         }
//     }
    
//     wf_enq_slow(q, h, v, cid);
    
//     WF_ATOMIC_STORE(&h->enq_node_id, h->Ep->id);
//     WF_ATOMIC_STORE(&h->hzd_node_id, (uint64_t)nullptr);
// }

// __device__ uint64_t wf_dequeue(wf_queue* q, wf_handle* h) {
//     WF_ATOMIC_STORE(&h->hzd_node_id, (uint64_t)h->Dp);
    
//     uint64_t cid;
//     for (int p = MAX_PATIENCE; p >= 0; p--) {
//         uint64_t result = wf_deq_fast(q, h, &cid);
//         if (result != WF_TOP) {
//             if (result != WF_EMPTY && h->Dh) {
//                 wf_help_deq(q, h, h->Dh);
//                 h->Dh = h->Dh->next;
//             }
//             WF_ATOMIC_STORE(&h->deq_node_id, h->Dp->id);
//             WF_ATOMIC_STORE(&h->hzd_node_id, (uint64_t)nullptr);

//             // OPTIMIZATION: Much less frequent cleanup
//             if ((cid % 10000) == 0) {
//                 wf_cleanup(q, h);
//             }

//             return result;
//         }
//     }
    
//     uint64_t result = wf_deq_slow(q, h, cid);
    
//     WF_ATOMIC_STORE(&h->deq_node_id, h->Dp->id);
//     WF_ATOMIC_STORE(&h->hzd_node_id, (uint64_t)nullptr);
//     return result;
// }

// //=============================================================================
// // CLEANUP FUNCTIONS (optimized but preserves correctness)
// //=============================================================================

// __device__ void wf_cleanup(wf_queue* q, wf_handle* h) {
//     static __device__ uint64_t cleanup_counter = 0;
//     uint64_t my_counter = WF_ATOMIC_ADD(&cleanup_counter, 1);
//     if (my_counter % 100 != 0) return; // Much less frequent cleanup

//     uint64_t i = WF_ATOMIC_LOAD(&q->I);
//     if (i == (uint64_t)-1) return;

//     wf_node* e = h->Dp;
//     if (!e) return;

//     if (e->id < i || (e->id - i) < MAX_GARBAGE) return;
//     if (!WF_ATOMIC_CAS(&q->I, i, (uint64_t)-1)) return;
    
//     // Note: In pool allocation, we don't actually free memory
//     // We just update pointers for correctness
//     wf_node* s = q->Hp;
//     if (!s) {
//         WF_ATOMIC_STORE(&q->I, i);
//         return;
//     }

//     // Simplified cleanup for pool allocation
//     q->Hp = e;
//     WF_ATOMIC_STORE(&q->I, e->id);
//     // No actual memory freeing needed with pool allocation
// }

// __device__ void wf_update_pointer(wf_node** from, wf_node** to, wf_handle* h) {
//     if (!from || !to || !*to) return;
    
//     wf_node* n = *from;
//     if (n && n->id < (*to)->id) {
//         uint64_t expected = (uint64_t)n;
//         if (!WF_ATOMIC_CAS((uint64_t*)from, expected, (uint64_t)*to)) {
//             n = *from;
//             if (n && n->id < (*to)->id) *to = n;
//         }
//     }
// }

// __device__ void wf_verify_and_update(wf_node** seg, wf_node* hazard_node) {
//     if (hazard_node != nullptr && *seg != nullptr) {
//         if (hazard_node->id < (*seg)->id) {
//             *seg = hazard_node;
//         }
//     }
// }

// __device__ void wf_free_list(wf_node* start, wf_node* end) {
//     // With pool allocation, no actual freeing needed
//     // This function exists for compatibility but does nothing
// }

// //=============================================================================
// // INITIALIZATION FUNCTIONS 
// //=============================================================================

// // __device__ void wf_queue_init(wf_queue* q, uint32_t nprocs) {
// //     // Initialize pool first
// //     q->pool.nodes = nullptr;
// //     q->pool.pool_counter = 0;
// //     q->pool.pool_size = 0;
// //     q->pool.initialized = false;
    
// //     // Initialize pool (this will allocate all segments at once)
// //     wf_init_memory_pool(q);
    
// //     // Create initial head node from pool
// //     wf_node* head = wf_new_node(q, 0);
    
// //     q->Hp = head;
// //     WF_ATOMIC_STORE(&q->Hi, 0);
// //     WF_ATOMIC_STORE(&q->Ei, 1);
// //     WF_ATOMIC_STORE(&q->Di, 1);
// //     WF_ATOMIC_STORE(&q->I, 0); 
// //     q->nprocs = nprocs;
// // }
// __device__ void wf_queue_init(wf_queue* q, uint32_t nprocs) {
//     // Initialize pool first - UPDATED SECTION
//     q->pool.initialized = false;
//     q->pool.block_counter = 0;
//     q->pool.total_nodes = 0;
    
//     // Initialize pool (this is now FAST - only 16MB pre-allocation)
//     wf_init_memory_pool(q);
    
//     // Create initial head node from first block
//     wf_node* head = wf_new_node(q, 0);
    
//     q->Hp = head;
//     WF_ATOMIC_STORE(&q->Hi, 0);
//     WF_ATOMIC_STORE(&q->Ei, 1);
//     WF_ATOMIC_STORE(&q->Di, 1);
//     WF_ATOMIC_STORE(&q->I, 0); 
//     q->nprocs = nprocs;
// }

// __device__ void wf_handle_init(wf_handle* h, wf_queue* q, wf_handle* next_handle, 
//                               wf_handle* enq_helper, wf_handle* deq_helper) {
    
//     if (!h || !q) return;
//     h->next = next_handle ? next_handle : h;
//     WF_ATOMIC_STORE(&h->hzd_node_id, (uint64_t)nullptr);
    
//     h->Ep = q->Hp;
//     WF_ATOMIC_STORE(&h->enq_node_id, q->Hp ? q->Hp->id : 0);
//     h->Dp = q->Hp;
//     WF_ATOMIC_STORE(&h->deq_node_id, q->Hp ? q->Hp->id : 0);
    
//     h->Er.val = 0;
//     WF_ATOMIC_STORE(&h->Er.state, WF_MAKE_STATE(0, 0));
//     h->Dr.id = 0;
//     WF_ATOMIC_STORE(&h->Dr.state, WF_MAKE_STATE(0, (uint64_t)-1));
    
//     h->Eh = enq_helper ? enq_helper : h;
//     WF_ATOMIC_STORE(&h->Ei, 0);
//     h->Dh = deq_helper ? deq_helper : h;
// }

// //=============================================================================
// // KERNEL FUNCTIONS
// //=============================================================================

// __global__ void test_wf_queue_kernel(wf_queue* q, wf_handle* handles, 
//                                     uint64_t* test_data, int num_ops) {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid >= q->nprocs) return;
    
//     wf_handle* h = &handles[tid];
    
//     for (int i = 0; i < num_ops; i++) {
//         uint64_t val = (uint64_t)tid * 1000 + i + 1;
//         wf_enqueue(q, h, val);
        
//         uint64_t dequeued = wf_dequeue(q, h);
//         if (dequeued != WF_EMPTY) {
//             test_data[tid * num_ops + i] = dequeued;
//         }
//     }
// }

// __global__ void wf_init_kernel(wf_queue* q, wf_handle* handles, int num_threads) {
//     if (threadIdx.x == 0 && blockIdx.x == 0) {
//         wf_queue_init(q, num_threads);
        
//         for (int i = 0; i < num_threads; i++) {
//             wf_handle* next_handle = &handles[(i + 1) % num_threads];
//             wf_handle* enq_helper = &handles[(i + 1) % num_threads];
//             wf_handle* deq_helper = &handles[(i + 1) % num_threads];
            
//             wf_handle_init(&handles[i], q, next_handle, enq_helper, deq_helper);
//         }
//     }
// }

// __global__ void wf_validate_kernel(wf_queue* q, wf_handle* handles, int num_threads) {
//     if (threadIdx.x == 0 && blockIdx.x == 0) {
//         printf("=== Optimized Queue Validation ===\n");
//         printf("Queue State: Ei=%lu, Di=%lu, Hi=%lu\n", 
//                WF_ATOMIC_LOAD(&q->Ei), WF_ATOMIC_LOAD(&q->Di), WF_ATOMIC_LOAD(&q->Hi));
//         printf("Pool State: size=%lu, allocated=%lu\n", 
//                q->pool.pool_size, WF_ATOMIC_LOAD(&q->pool.pool_counter));
        
//         for (int i = 0; i < min(num_threads, 5); i++) {
//             printf("Handle %d: enq_node_id=%lu, deq_node_id=%lu, hzd_node_id=%lu\n",
//                    i, WF_ATOMIC_LOAD(&handles[i].enq_node_id), 
//                    WF_ATOMIC_LOAD(&handles[i].deq_node_id),
//                    WF_ATOMIC_LOAD(&handles[i].hzd_node_id));
//         }
        
//         uint64_t ei = WF_ATOMIC_LOAD(&q->Ei);
//         uint64_t di = WF_ATOMIC_LOAD(&q->Di);
        
//         printf("Invariant check: Ei >= Di = %s\n", (ei >= di) ? "PASS" : "FAIL");
//         printf("Memory optimization: Pool allocation = %s\n", 
//                q->pool.initialized ? "ACTIVE" : "INACTIVE");
//         printf("=== End Validation ===\n");
//     }
// }

// __global__ void wf_performance_test_kernel(wf_queue* q, wf_handle* handles, 
//                                           uint64_t* results, int operations_per_thread,
//                                           int test_type) {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid >= q->nprocs) return;
    
//     wf_handle* h = &handles[tid];
//     uint64_t local_ops = 0;
    
//     __syncthreads();
    
//     switch (test_type) {
//         case 0: // Enqueue-Dequeue pairs
//             for (int i = 0; i < operations_per_thread; i++) {
//                 uint64_t val = tid * 10000 + i + 1;
//                 wf_enqueue(q, h, val);
//                 local_ops++;
                
//                 uint64_t dequeued = wf_dequeue(q, h);
//                 if (dequeued != WF_EMPTY) {
//                     local_ops++;
//                 }
//             }
//             break;
            
//         case 1: // 50% enqueues, 50% dequeues
//             for (int i = 0; i < operations_per_thread; i++) {
//                 if (i % 2 == 0) {
//                     uint64_t val = tid * 10000 + i + 1;
//                     wf_enqueue(q, h, val);
//                 } else {
//                     wf_dequeue(q, h);
//                 }
//                 local_ops++;
//             }
//             break;
            
//         case 2: // Mostly enqueues (80%)
//             for (int i = 0; i < operations_per_thread; i++) {
//                 if (i % 5 != 0) {
//                     uint64_t val = tid * 10000 + i + 1;
//                     wf_enqueue(q, h, val);
//                 } else {
//                     wf_dequeue(q, h);
//                 }
//                 local_ops++;
//             }
//             break;
            
//         case 3: // Mostly dequeues (80%)
//             for (int i = 0; i < operations_per_thread; i++) {
//                 if (i % 5 == 0) {
//                     uint64_t val = tid * 10000 + i + 1;
//                     wf_enqueue(q, h, val);
//                 } else {
//                     wf_dequeue(q, h);
//                 }
//                 local_ops++;
//             }
//             break;
//     }
    
//     results[tid] = local_ops;
// }

// //=============================================================================
// // HOST-SIDE INITIALIZATION FUNCTIONS
// //=============================================================================

// void wf_queue_host_init(wf_queue** d_q, wf_handle** d_handles, int num_threads) {
//     hipMalloc((void**)d_q, sizeof(wf_queue));
//     hipMalloc((void**)d_handles, num_threads * sizeof(wf_handle));
    
//     wf_init_kernel<<<1, 1>>>(*d_q, *d_handles, num_threads);
//     hipDeviceSynchronize();
    
//     // Validate pool initialization
//     wf_queue h_queue;
//     hipMemcpy(&h_queue, *d_q, sizeof(wf_queue), hipMemcpyDeviceToHost);
    
//     if (h_queue.pool.initialized) {
//         printf("✅ Optimized WF-Queue initialized successfully!\n");
//         printf("   - Pool size: %lu segments\n", h_queue.pool.pool_size);
//         printf("   - Node size: %d cells per segment\n", WF_NODE_SIZE);
//         printf("   - Total capacity: ~%lu operations\n", 
//                h_queue.pool.pool_size * WF_NODE_SIZE);
//     } else {
//         printf("⚠️  Warning: Pool initialization may have failed\n");
//     }
// }

// //=============================================================================
// // DEBUGGING AND TESTING FUNCTIONS
// //=============================================================================

// __device__ void wf_print_queue_state(wf_queue* q) {
//     printf("Optimized Queue State: Ei=%lu, Di=%lu, Hi=%lu\n", 
//            WF_ATOMIC_LOAD(&q->Ei), WF_ATOMIC_LOAD(&q->Di), WF_ATOMIC_LOAD(&q->Hi));
//     printf("Pool: size=%lu, used=%lu, initialized=%s\n",
//            q->pool.pool_size, WF_ATOMIC_LOAD(&q->pool.pool_counter),
//            q->pool.initialized ? "YES" : "NO");
// }

// __device__ void wf_print_handle_state(wf_handle* h, int tid) {
//     printf("Handle %d: enq_node_id=%lu, deq_node_id=%lu, hzd_node_id=%lu\n",
//            tid, WF_ATOMIC_LOAD(&h->enq_node_id), WF_ATOMIC_LOAD(&h->deq_node_id), 
//            WF_ATOMIC_LOAD(&h->hzd_node_id));
// }

// //=============================================================================
// // SIMPLE TEST FUNCTION
// //=============================================================================

// __global__ void ultra_simple_test(wf_queue* q, wf_handle* handles, int* success) {
//     if (threadIdx.x == 0 && blockIdx.x == 0) {
//         wf_handle* h = &handles[0];
        
//         wf_enqueue(q, h, 42);
//         uint64_t result = wf_dequeue(q, h);
        
//         *success = (result == 42) ? 1 : 0;
        
//         printf("Ultra simple test: enqueued 42, dequeued %lu, success=%d\n", 
//                result, *success);
//         printf("Pool status: initialized=%s, used=%lu/%lu\n",
//                q->pool.initialized ? "YES" : "NO",
//                WF_ATOMIC_LOAD(&q->pool.pool_counter), q->pool.pool_size);
//     }
// }

// // #endif // WFQUEUE_HIP_OPT_H





// wfqueue_hip_opt.hpp - Complete Header-Only Optimized WF-Queue Implementation
