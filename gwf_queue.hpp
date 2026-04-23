#pragma once

#ifndef __GWF_QUEUE_HPP__
#define __GWF_QUEUE_HPP__ 1

/*
First GPU wait-free queue implementation, designed to use 64-bit CAS only. 
This version uses the following as the bits for the slot:

THis is the {Note, {Cycle, Safe, Enq reserved, Index} } format. from 128 bit to 64 bit.

Bits:    [63:56]   [56:48]      [47]    [46]    [45:32]       [31:0]
Field:   Note(8)   Cycle(8)    Safe(1)  Enq(1)  reserved(14)  Index(32)


128-bit {Counter, phase2.ptr} Head/Tail pair in 64-bit format.

Bits:   [63:16]     [15:0]
Field:  Counter(48) ThrIdx(16)
*/


#include <hip/hip_runtime.h>
#include <stdint.h>
#include <cstdint>


/*============================================================================
* QUEUE Counter template
*============================================================================*/
template <int THRIDX_BITS = 14>
struct wf_config {
     static constexpr int COUNTER_BITS = 64 - THRIDX_BITS;


    //  Mac number of threads that can use this queue, last value in the ThrIdx range is reserved for NULL_TIDX
    static constexpr uint32_t MAX_THREADS = (1u << THRIDX_BITS) - 1;

    // Sentianel value meaning "No thread", in the ThrIdx field
    static constexpr uint64_t NULL_TID = (1ULL << THRIDX_BITS) - 1;

    //  Mask to extract ThrIdx from the global word.
    static constexpr uint64_t THRIDX_MASK = NULL_TID;

    //  amount to add for fast-path FAA: increment the counter field by 1.
    /*
    Since Counter occupies bits [63..THRIDX_BITS], adding this value 
    bumps the counter by 1, while leaving ThrIdx untouched
    Eg: for THRIDX_BITS = 16:
        COUNTER_INC = 1 << 16 = 0x4000     
        atomicAdd(&Tail, 0x4000) increments the Counter by 1, without changing the ThrIdx field
    */
    static constexpr uint64_t COUNTER_INC = 1ULL << THRIDX_BITS;

    static_assert(THRIDX_BITS >= 5 && THRIDX_BITS <= 20,
        "THRIDX_BITS should be in [5, 20]. Below 5 means <32 threads. "
        "Above 20 leaves <44 counter bits which risks ABA.");
    
    static_assert(COUNTER_BITS >= 44,
        "Need at least 44 counter bits. At 1 Gop/s, 44 bits lasts ~4.9 hours. "
        "Reduce THRIDX_BITS for longer-running kernels.");

    /* 
    Pack a counter value and thread index into a single 64-bit word.
    Layout:
    [63..THRIDX_BITS] Counter
    [THRIDX_BITS-1..0] Thread Index (or NULL_TID for no thread)
    
    */
    __device__ __host__ static __forceinline__ 
    uint64_t pack_global(uint64_t counter, uint32_t threadIdx) {
        return (counter << THRIDX_BITS) | ((uint64_t)threadIdx & THRIDX_MASK);
    }

    // Unpack the counter from a packed global word 64-bit word.
    __device__ __host__ static __forceinline__
    uint64_t unpack_counter(uint64_t global_word) {
        return global_word >> THRIDX_BITS;
    }

    // Unpack the thread index from a packed global word 64-bit word.
    __device__ __host__ static __forceinline__
    uint32_t unpack_thridx(uint64_t global_word) {
        return (uint32_t)(global_word & THRIDX_MASK);
    }

    //Check if the global word has an active phase2 request
    __device__ __host__ static __forceinline__
    bool has_active_phase2(uint64_t global_word) {
        return unpack_thridx(global_word) != (uint32_t)NULL_TID;
    }
};

/* 
* Convient preset configuration aliases
*/
using gwf_1k = wf_config<10>; // 1023 threads max, 54-bit counter
using gwf_2k = wf_config<11>; // 2047 threads max, 53-bit counter
using gwf_4k = wf_config<12>; // 4095 threads max, 52-bit counter
using gwf_8k = wf_config<13>; // 8191 threads max, 51-bit counter
using gwf_16k = wf_config<14>; // 16383 threads max, 50-bit counter
using gwf_32k = wf_config<15>; // 32767 threads max, 49-bit counter
using gwf_64k = wf_config<16>; // 65535 threads max, 48-bit counter


/*============================================================================
 * CYCLE COMPARISON — Modular arithmetic on reduced-width cycle tags
 *============================================================================*/
 
#ifndef WF_CYCLE_BITS
#define WF_CYCLE_BITS 8
#endif

static constexpr uint32_t CYCLE_RANGE = 1u << WF_CYCLE_BITS; // number of distinct cycle values (e.g., 256 for 8 bits)
static constexpr uint32_t CYCLE_MASK = CYCLE_RANGE - 1;
static constexpr uint32_t HALF_CYCLE = CYCLE_RANGE >> 1; // half the cycle range, used for determining "newer" vs "older" cycles

__device__ __forceinline__
uint32_t cycle_of(uint64_t ticket, uint32_t num_slots) {
    return (uint32_t)((ticket / num_slots) & CYCLE_MASK);
}

__device__ __forceinline__
uint32_t slot_of(uint64_t ticket, uint32_t num_slots) {
    return (uint32_t)(ticket & (num_slots - 1));
}

/** Three-way modular cycle comparison.
 *  Returns: -1 if a BEHIND b, 0 if EQUAL, +1 if a AHEAD of b. */
__device__ __forceinline__
int32_t cycle_cmp(uint32_t a, uint32_t b) {
    uint32_t delta = (a - b) & CYCLE_MASK;
    if (delta == 0)          return 0;
    if (delta < HALF_CYCLE)  return 1;
    return -1;
}


/*============================================================================
 * ENTRY — 64-bit packed ring buffer slot
 *
 *   Bit 63                                                Bit 0
 *   ┌────────┬────────┬──┬──┬──────────────┬──────────────────┐
 *   │ Note   │ Cycle  │Sf│Eq│   (zeroed)   │     Index        │
 *   │ 8 bits │ 8 bits │1b│1b│   14 bits    │    32 bits       │
 *   └────────┴────────┴──┴──┴──────────────┴──────────────────┘
*============================================================================*/

namespace wf_entry {

    static constexpr int NOTE_SHIFT = 56;
    static constexpr int CYCLE_SHIFT = 48;
    static constexpr int ISSAFE_SHIFT = 47;
    static constexpr int ENQ_SHIFT = 46;

    static constexpr uint64_t NOTE_FIELD = (uint64_t)CYCLE_MASK << NOTE_SHIFT;
    static constexpr uint64_t CYCLE_FIELD = (uint64_t)CYCLE_MASK << CYCLE_SHIFT;
    static constexpr uint64_t ISSAFE_BIT = 1ULL << ISSAFE_SHIFT;
    static constexpr uint64_t ENQ_BIT = 1ULL << ENQ_SHIFT;
    static constexpr uint64_t INDEX_MASK = 0xFFFFFFFFULL; // lower 32 bits

    static constexpr uint32_t BOT = 0xFFFFFFFEu;  /* ⊥ : empty */
    static constexpr uint32_t BOT_C = 0xFFFFFFFFu;  /* ⊥c : consumed */
    
    /* OR Mask for consume: sets Enq=1 and Index: ⊥c, preserves all else */
    static constexpr uint64_t CONSUME_MASK = ENQ_BIT | (uint64_t)BOT_C;
    

    __device__ __forceinline__ uint32_t note(uint64_t e) { return (uint32_t)((e >> NOTE_SHIFT) & CYCLE_MASK); }
    __device__ __forceinline__ uint32_t cycle(uint64_t e) { return (uint32_t)((e >> CYCLE_SHIFT) & CYCLE_MASK); }
    __device__ __forceinline__ bool issafe(uint64_t e) { return (e >> ISSAFE_SHIFT) & 1; }
    __device__ __forceinline__ bool enq(uint64_t e) { return (e >> ENQ_SHIFT) & 1; }
    __device__ __forceinline__ uint32_t index(uint64_t e) { return (uint32_t)(e & INDEX_MASK);}

    __device__ __forceinline__
    uint64_t pack(uint32_t note_val, uint32_t cycle_val,
                  bool safe, bool enq_val, uint32_t idx) {

        return ((uint64_t)(note_val & CYCLE_MASK) << NOTE_SHIFT)
                | ((uint64_t)(cycle_val & CYCLE_MASK) << CYCLE_SHIFT)
                | ((uint64_t)safe << ISSAFE_SHIFT)
                | ((uint64_t)enq_val << ENQ_SHIFT)
                | ((uint64_t)idx);
    }

    /* Initial entry: Note= -1 (0xFF), Cycle=0, IsSafe=1, Enq=1, Index=BOT */
    static constexpr uint64_t INIT = ((uint64_t)0xFF << NOTE_SHIFT) | ISSAFE_BIT | ENQ_BIT | BOT;

} /* namespace wf_entry */


/*============================================================================
 * LOCAL HEAD/TAIL — Per-thread snapshot with INC/FIN flags
 *
 *   [63..2]  CounterValue   62 bits
 *   [1]      INC            1 bit
 *   [0]      FIN            1 bit
*============================================================================*/

// Why is counter 62 bits, is this local head/tail used for phase2 ? why is it 64_t instead of 32-bit ?

namespace wf_local_ptr {
    static constexpr uint64_t FIN_BIT = 1ULL << 0;
    static constexpr uint64_t INC_BIT = 1ULL << 1;
    static constexpr uint64_t FLAGS = FIN_BIT | INC_BIT;
    static constexpr uint64_t CNT_MASK = ~FLAGS; // upper 62 bits 

    __device__ __forceinline__ uint64_t counter(uint64_t l) { return l & CNT_MASK; }
    __device__ __forceinline__ bool has_fin(uint64_t l)     { return l & FIN_BIT; }
    __device__ __forceinline__ bool has_inc(uint64_t l)     { return l & INC_BIT; }

    __device__ __forceinline__ uint64_t with_fin(uint64_t v) { return (v & CNT_MASK) | FIN_BIT; }
    __device__ __forceinline__ uint64_t with_inc(uint64_t v) { return (v & CNT_MASK) | INC_BIT; }
    __device__ __forceinline__ uint64_t clean(uint64_t v)    { return v & CNT_MASK; }

} /* namespace wf_local_ptr */

/* Cache REMAP */
#ifndef GPU_CACHELINE_BYTES
#define GPU_CACHELINE_BYTES 64
#endif
// static constexpr uint32_t GPU_CACHELINE_BYTES = 128;
static constexpr uint32_t ENTRY_BYTES = sizeof(uint64_t);
static constexpr uint32_t ENTRIES_PER_CACHELINE = (uint32_t)(GPU_CACHELINE_BYTES) / ENTRY_BYTES;

__device__ __forceinline__
uint32_t cache_remap(uint32_t pos, uint32_t num_slots) {
    return ((pos & (ENTRIES_PER_CACHELINE - 1)) * (num_slots / ENTRIES_PER_CACHELINE)) 
        + (pos / ENTRIES_PER_CACHELINE);
}


/* Phase 2 Record */

// struct phase2rec_t {
//     uint32_t            seq1;  /* Incremented before fields written. Init: 1 */
//     unsigned long long* local; /* Ptr to localTail or localHead to CAS */
//     uint64_t            cnt;   /* Counter value before global increment */
//     uint32_t            seq2;  /* Set to seq1 after fields written. Init: 0 */
// };

struct phase2rec_t {
    unsigned int        seq1;  /* Incremented before fields written. Init: 1 */
    unsigned long long  local; /* Ptr to localTail or localHead to CAS, EDIT: USE integer and then Cast to pointer */
    unsigned long long  cnt;   /* Counter value before global increment */
    unsigned int        seq2;  /* Set to seq1 after fields written. Init: 0 */
};

/*============================================================================
 * THREAD RECORD
 *
 * 128-byte aligned to avoid false sharing. Cache size dependent
 *
 * Fields declared `unsigned long long` are targets of atomicCAS.
 * Everything else is uint64_t or uint32_t.
*============================================================================*/

struct alignas(GPU_CACHELINE_BYTES) thrdrec_t {
    /* Private (used when helping other threads) */
    int32_t nextCheck; // countdown, inited with HELP_DELAY
    uint32_t nextTid; // Next peer to check: Init with own

    /* Phase 2 */
    phase2rec_t phase2;

    /*  Help request (shared, read by helpers) */
    // uint32_t seq1;
    // uint32_t enqueue; // 1=enq, 0=deq
    // uint32_t pending; // 1=active request
    // uint32_t _pad0;

    unsigned int seq1;
    unsigned int enqueue; // 1=enq, 0=deq
    unsigned int pending; // 1=active request
    uint32_t _pad0;

    /* Local counters (targets of atomicCas) */
    unsigned long long localTail; // [63..2]=cnt, [1]=INC, [0]=FIN
    unsigned long long initTail; // helpers start from this value, set to the global tail init
    unsigned long long localHead; // Same as localTail, for dequeue;
    unsigned long long initHead; 


    /* Payload */
    unsigned int index; /* 32 bit index being enqueued */
    unsigned int seq2; /*  Written LAST. Init: 0 */ 

};


static_assert(sizeof(thrdrec_t) <= (2 * GPU_CACHELINE_BYTES),
    "thrdrec_t must fit within 2 cache lines to avoid false sharing.");


/* Knobs */
#ifndef WF_HELP_DELAY
#define WF_HELP_DELAY 64
#endif
 
#ifndef WF_ENQ_PATIENCE
#define WF_ENQ_PATIENCE 128
#endif
 
#ifndef WF_DEQ_PATIENCE
#define WF_DEQ_PATIENCE 256
#endif


/*============================================================================
 * QUEUE INSTANCE
 *============================================================================*/
 
template <typename Config, uint32_t N_PARAM>
struct wf_t {
    static constexpr uint32_t N     = N_PARAM;
    static constexpr uint32_t SLOTS = 2 * N;
    static constexpr int32_t  THRESHOLD_MAX = 3 * (int32_t)N - 1;
 
    static_assert((N & (N - 1)) == 0, "N must be a power of two");
    static_assert(Config::MAX_THREADS <= N, "wf requires k <= n");
    static_assert(
        CYCLE_RANGE > 2u * (WF_HELP_DELAY * Config::MAX_THREADS / (2 * N) + 3),
        "Cycle bits insufficient for this thread/buffer configuration");
 
    unsigned long long Tail;            /* Global {Counter, ThrIdx}         */
    unsigned long long Head;
    int32_t            Threshold;       /* -1 = empty, 3n-1 after enqueue   */
    int32_t            _pad;
 
    unsigned long long Entry[SLOTS];    /* 2n packed entry slots            */
    thrdrec_t          Record[Config::MAX_THREADS];
};

/*============================================================================
 * INDIRECTION LAYER — Full MPMC queue with 64-bit payloads
 *============================================================================*/
 
template <typename Config, uint32_t N_PARAM>
struct wf_mpmc_t {
    using queue_type = wf_t<Config, N_PARAM>;
 
    queue_type aq;                /* Allocated queue: filled indices         */
    queue_type fq;                /* Free queue: available indices           */
    uint64_t   data[N_PARAM];    /* Payload array [0, N-1]                  */
};
 




template <typename Config, uint32_t N_PARAM>
__device__ void wf_init(wf_t<Config, N_PARAM>* q);
 
template <typename Config, uint32_t N_PARAM>
__device__ void wf_mpmc_init(wf_mpmc_t<Config, N_PARAM>* q);
 
/* Layer 1 */
template <typename Config, uint32_t N_PARAM>
__device__ bool Enqueue_Ptr(wf_mpmc_t<Config, N_PARAM>* q, uint64_t value);
 
template <typename Config, uint32_t N_PARAM>
__device__ bool Dequeue_Ptr(wf_mpmc_t<Config, N_PARAM>* q, uint64_t* out);
 
/* Layer 2 */
template <typename Config, uint32_t N_PARAM>
__device__ void Enqueue_wf(wf_t<Config, N_PARAM>* q, uint32_t index);
 
template <typename Config, uint32_t N_PARAM>
__device__ uint32_t Dequeue_wf(wf_t<Config, N_PARAM>* q);
 
/* Layer 3a: Fast path */
template <typename Config, uint32_t N_PARAM>
__device__ uint64_t try_enq(wf_t<Config, N_PARAM>* q, uint32_t index);
 
template <typename Config, uint32_t N_PARAM>
__device__ uint64_t try_deq(wf_t<Config, N_PARAM>* q, uint32_t* out_index);
 
template <typename Config, uint32_t N_PARAM>
__device__ void consume(wf_t<Config, N_PARAM>* q,
                        uint64_t h, uint32_t j, uint64_t entry_val);
 
template <typename Config, uint32_t N_PARAM>
__device__ void finalize_request(wf_t<Config, N_PARAM>* q, uint64_t h);
 
template <typename Config, uint32_t N_PARAM>
__device__ void catchup(wf_t<Config, N_PARAM>* q,
                        uint64_t tail, uint64_t head);
 
/* Layer 3b: Slow path */
template <typename Config, uint32_t N_PARAM>
__device__ void enqueue_slow(wf_t<Config, N_PARAM>* q, uint64_t tail,
                             uint32_t index, thrdrec_t* r);
 
template <typename Config, uint32_t N_PARAM>
__device__ void dequeue_slow(wf_t<Config, N_PARAM>* q,
                             uint64_t head, thrdrec_t* r);
 
template <typename Config, uint32_t N_PARAM>
__device__ bool try_enq_slow(wf_t<Config, N_PARAM>* q, uint64_t T,
                             uint32_t index, thrdrec_t* r);
 
template <typename Config, uint32_t N_PARAM>
__device__ bool try_deq_slow(wf_t<Config, N_PARAM>* q,
                             uint64_t H, thrdrec_t* r);
 
template <typename Config, uint32_t N_PARAM>
__device__ bool slow_faa(wf_t<Config, N_PARAM>* q,
                         unsigned long long* globalp,
                         unsigned long long* local,
                         uint64_t* v,
                         int32_t* thld);
 
__device__ void prepare_phase2(phase2rec_t* phase2,
                               unsigned long long* local,
                               uint64_t cnt);
 
template <typename Config>
__device__ uint64_t load_global_help_phase2(unsigned long long* globalp,
                                            unsigned long long* mylocal);
 
/* Layer 4: Helping */
template <typename Config, uint32_t N_PARAM>
__device__ void help_threads(wf_t<Config, N_PARAM>* q);
 
template <typename Config, uint32_t N_PARAM>
__device__ void help_enqueue(wf_t<Config, N_PARAM>* q, thrdrec_t* thr);
 
template <typename Config, uint32_t N_PARAM>
__device__ void help_dequeue(wf_t<Config, N_PARAM>* q, thrdrec_t* thr);
 
/* Layer 5: GPU-specific */
template <typename Config, uint32_t N_PARAM>
__device__ int warp_help_scan(wf_t<Config, N_PARAM>* q);
 
__device__ void warp_broadcast_request(uint64_t* init_val,
                                       uint32_t* index,
                                       uint32_t* is_enqueue,
                                       int source_lane);

#endif // __GWF_QUEUE_HPP__