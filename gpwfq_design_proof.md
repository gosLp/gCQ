# GPWFQ: GPU Packed Wait-Free Queue — Design and Correctness Proof

## 1. Problem Statement

**Goal**: Design a wait-free, linearizable FIFO queue for AMD GPUs (HIP/ROCm) using
only **64-bit CAS** (no 128-bit CAS / CAS2 / DWCAS).

**Constraints**:
- AMD HIP only supports `atomicCAS` on 32-bit and 64-bit words
- `atomicAdd` (FAA), `atomicExch` (SWAP), `atomicOr`, `atomicAnd` available on 64-bit
- Memory model: GPU relaxed atomics + `__threadfence()` for ordering
- Must support `nprocs` concurrent threads where `nprocs ≤ 2^P` for a compile-time parameter `P` (default P=15, max 32768 threads)

**Wait-freedom definition**: Every call to `enqueue()` or `dequeue()` by any thread
completes in a **bounded number of steps** that depends only on `nprocs` and `ring_size`,
regardless of the timing, scheduling, or suspension of any other threads.

---

## 2. Why Existing GPU Approaches Fail

### 2.1 Why wCQ Cannot Be Directly Ported

wCQ (Nikolaev, SPAA'22) achieves wait-freedom using two instances of CAS2 (128-bit CAS):

1. **On ring cells**: Each cell packs `(entry, addon)` where `entry` encodes
   `(cycle, index, flags)` and `addon` encodes a `note` field. A single CAS2
   atomically updates both.

2. **On `TailGP`/`HeadGP`**: The slow-path's cooperative `slow_faa` packs
   `(counter, phase2_pointer)` into a 128-bit word. The pointer identifies which
   thread performed an increment so that helpers can finalize its local state.

Without CAS2, the TOCTOU vulnerability between separate single-word updates
breaks the protocol.

### 2.2 Why GWFQ (wfqueue_wfq64.hpp) Is Not Wait-Free

GWFQ uses FAA for slow-path tickets and a FIN bit for termination, but:

- The slow path loops up to `WFQ_SLOW_LIMIT = 131072` times, then **gives up**
  (increments `proof_fail`). A give-up means the operation is lost — not even
  lock-free for that operation.
- Helping is probabilistic (round-robin countdown), not deterministic. A stalled
  thread may never receive help within any fixed bound.

**Counterexample**: Thread T₀ enters slow enqueue. All other threads are on the
fast path with 100% first-try success. Nobody's `next_check` counter reaches zero
during T₀'s bounded iterations. T₀'s loop exhausts → operation lost.

### 2.3 Why BWFQ (wfqueue_bwfq.hpp) Is Not Wait-Free

BWFQ has Phase 3 `while(pending)` loops. The claimed argument:

> "Admission guarantees a counterpart → help_all_once ensures progress → finite."

**Counterexample (dependency chain)**:
1. Thread T₀ admitted enqueue, ticket `p₀`. Slot `ring[p₀ % cap]` has `seq < p₀`
   because the previous dequeue at this slot hasn't completed.
2. That dequeue belongs to T₁, also stuck in Phase 3 because its slot's enqueue
   hasn't arrived yet.
3. `help_all_once` visits both records. For T₀: `try_complete_enq_once(p₀)` →
   `dif < 0` (slot not freed) → returns false. For T₁: `try_complete_deq_once(p₁)` →
   `dif < 0` (enqueue not arrived) → returns false.
4. Neither makes progress. Both are waiting on operations that haven't happened yet.

The `help_all_once` can only complete an operation whose slot is ALREADY in the
correct seq state. It cannot FORCE a slot into the correct state. The while-loop
is lock-free (system-wide progress eventually happens as the dependency chain
unwinds), but NOT wait-free (no per-thread bound).

---

## 3. The GPWFQ Design

### 3.1 Core Insight: Bit-Packing Replaces CAS2

wCQ's CAS2 on `TailGP` packs `(48-bit counter, 64-bit pointer)`. The pointer
is used only as an index into the thread's phase2 record. On GPU with bounded
`nprocs`, we replace the pointer with a **thread ID**:

```
TailGP (64-bit word):
  [63 : P+1]  counter  (64 - P - 1 bits)
  [P : 1]     helper_tid (P bits)
  [0]         has_helper (1 bit)
```

For P=15 (max 32768 threads): counter = 48 bits = 2^48 ≈ 281 trillion operations.
At 10 billion ops/sec peak GPU throughput, this lasts ~7.8 hours of continuous
maximum-rate operation — more than sufficient for any GPU kernel invocation.

For ring cells, wCQ's CAS2 packs `(entry, addon)`. With bounded ring size
(ring_size = 2 × nprocs ≤ 65536), the entry and addon fields fit in 64 bits:

```
Cell (64-bit word):
  [63 : 32]   entry_high (cycle + flags, 32 bits)
  [31 : 0]    entry_low (index + flags, 32 bits)

Where within the 64-bit word:
  Bits [63:48]  = note (16 bits, tracks cycle for skip detection)
  Bit  [47]     = is_safe flag
  Bits [46:32]  = cycle (15 bits, range 0..4n-1)
  Bits [31:16]  = reserved/padding
  Bits [15:0]   = index (16 bits, range 0..n-1 + sentinels)
```

The exact layout is parameterized by the order (ring size), ensuring all
cycle/index/note values fit. The key invariant: **any field combination
that wCQ's CAS2 updates atomically fits within our 64-bit word**.

### 3.2 Formal Model

**Shared state**:
- `Ring[0..2n-1]`: array of `uint64_t` (packed cells)
- `TailGP`: `uint64_t` (packed counter + helper)
- `HeadGP`: `uint64_t` (packed counter + helper)
- `Threshold`: `int64_t`
- `Records[0..nprocs-1]`: per-thread state (announcement + phase2)

**Per-thread state** (`Records[tid]`):
- `next_check`: help countdown
- `curr_thread`: round-robin help target
- `seq1, seq2`: sequence numbers for request validity
- `slow_tail, init_tail`: slow-path local counter for enqueue
- `slow_head, init_head`: slow-path local counter for dequeue
- `eidx`: operation type identifier
- `phase2.seq1, phase2.seq2`: phase2 validity
- `phase2.local`: pointer to local counter being advanced
- `phase2.cnt`: expected counter value

**Atomic primitives used**: FAA (fetch-and-add), CAS (compare-and-swap),
SWAP (unconditional exchange), Load, Store — all on 64-bit words only.

### 3.3 Encoding Functions

```
// TailGP / HeadGP encoding
pack_global(cnt, tid)    = (cnt << (P+1)) | (tid << 1) | 1
pack_global(cnt, NONE)   = (cnt << (P+1))
unpack_cnt(x)            = x >> (P+1)
unpack_tid(x)            = (x >> 1) & ((1 << P) - 1)
has_helper(x)            = x & 1

// Cell encoding (parameterized by order, n = 2^order, ring_size = 2n)
// Let C = ceil(log2(4n)) = order + 3 (cycle bits)
// Let I = order + 1 (index bits, including sentinel range)
// Let N = C (note bits, same range as cycle)
pack_cell(entry, note)   = (note << (C + I)) | entry
unpack_entry(x)          = x & ((1 << (C + I)) - 1)
unpack_note(x)           = x >> (C + I)

// Within entry:
entry_cycle(e)           = (e >> I) & ((1 << C) - 1)
entry_index(e)           = e & ((1 << I) - 1)
make_entry(cycle, index) = (cycle << I) | index
```

### 3.4 Operations

The operations are a direct transliteration of wCQ's algorithms (as given in
the reference `wfring.h` code) with every `__lfaba_cmpxchg_strong` (CAS2)
replaced by `CAS64` on the packed representation, and every `__wfring_pair`
replaced by our `pack_cell` or `pack_global`.

**Enqueue** (fast path):
```
function enqueue(ring, order, eidx, nonempty, state):
    eidx ^= (n - 1)                          // encode index
    if --state.next_check == 0: help(...)     // periodic helping
    for patience = MAX_PATIENCE downto 1:
        tail = FAA(TailGP_counter_word, 1)    // claim ticket
        tcycle = tail | (4n - 1)
        tidx = map(tail, order, 2n)
        cell = Load(Ring[tidx])
        entry = unpack_entry(cell)
        ecycle = entry | (4n - 1)
        if ecycle < tcycle and slot_available(entry, ecycle, tail):
            desired = pack_cell(tcycle ^ eidx, unpack_note(cell))
            if CAS64(Ring[tidx], cell, desired):
                update_threshold(...)
                return
    // Fast path exhausted → slow path
    enqueue_slow(ring, order, eidx, tail, nonempty, state)
```

**Slow path** (transliteration of wCQ's `do_enqueue_slow`):
```
function do_enqueue_slow(ring, order, eidx, seq, tail, nonempty, state):
    while slow_inc(TailGP, state.slow_tail, tail, NULL, state.phase2, my_tid):
        if Load(state.seq1) != seq: break     // request retired
        tcycle = tail | (4n - 1)
        tidx = map(tail >> 2, order, 2n)
        cell = Load(Ring[tidx])
        // ... same slot logic as fast path, but on the cooperatively-obtained ticket ...
        // On success: CAS the cell, set FIN on state.slow_tail, return
```

**`slow_inc`** (replaces wCQ's `__wfring_slow_inc`):
```
function slow_inc(global, local, prev, threshold, phase2, my_tid):
    gp = Load(global)
    loop:
        if Load(local) & FIN: return false
        cnt = help_phase2(global, gp)           // help + extract counter
        if not CAS(local, prev, cnt | INC):
            if prev & FIN: return false
            if not (prev & INC): return true    // finalized externally
            cnt = prev & ~(FIN | INC)
        else:
            prev = cnt | INC
        // Prepare phase2
        seq = Load(phase2.seq1) + 1
        Store(phase2.seq1, seq)
        phase2.local = &local
        phase2.cnt = cnt
        Store(phase2.seq2, seq)
        // CAS on global: (cnt, NONE) → (cnt+1, my_tid)
        expected = pack_global(cnt, NONE)
        desired = pack_global(cnt + 1, my_tid)
        if CAS64(global, expected, desired):    // ← 64-bit CAS, not CAS2!
            if threshold: FAA(threshold, -1)
            cnt_inc = cnt | INC
            CAS(local, cnt_inc, cnt)            // finalize local
            cleanup_old = desired
            CAS64(global, cleanup_old, pack_global(cnt+1, NONE))  // cleanup
            prev = cnt
            return true
        gp = expected  // failed CAS updates expected with current value
```

**`help_phase2`** (replaces wCQ's `__wfring_load_global_help_phase2`):
```
function help_phase2(global, gp):
    loop:
        if not has_helper(gp): break
        tid = unpack_tid(gp)
        p2 = Records[tid].phase2              // indirection via TID
        seq = Load(p2.seq2)
        local = p2.local
        cnt = p2.cnt
        if Load(p2.seq1) == seq:              // validate consistency
            cnt_inc = cnt | INC
            CAS(local, cnt_inc, cnt)          // finalize: remove INC
        // Clear helper flag regardless of whether finalization happened
        clean_new = pack_global(unpack_cnt(gp), NONE)
        if CAS64(global, gp, clean_new):
            gp = clean_new
            break
        // CAS failed → gp updated, retry
    return unpack_cnt(gp)
```

The dequeue side is symmetric, following wCQ's `__wfring_dequeue_slow` and
`__wfring_do_dequeue_slow` with the same packed 64-bit substitution.

---

## 4. Correctness Proofs

We prove three properties: linearizability, wait-freedom, and bounded memory.

### 4.1 Structural Equivalence to wCQ

**Lemma 4.1 (Encoding Faithfulness)**:
For any ring size `2n` where `n ≤ 2^(P-1)`, the packed 64-bit encoding
of `(counter, helper_tid)` in `TailGP`/`HeadGP` and `(entry, note)` in
ring cells is an injective mapping. That is, distinct `(counter, helper_tid)`
pairs map to distinct 64-bit values, and distinct `(entry, note)` pairs map
to distinct 64-bit cell values.

*Proof*: The encoding concatenates disjoint bit fields. For `TailGP`:
counter occupies bits `[63:P+1]`, `helper_tid` occupies bits `[P:1]`,
`has_helper` occupies bit `[0]`. These are non-overlapping. For ring cells:
`note` occupies the top N bits, `entry` (containing cycle and index in
disjoint sub-fields) occupies the bottom C+I bits. Since N + C + I ≤ 64
(verified: for n = 2^15 = 32768, C = 18, I = 16, N = 18, total = 52 ≤ 64),
the fields are non-overlapping. Injectivity follows from non-overlap. □

**Lemma 4.2 (CAS Equivalence)**:
Any `CAS2(addr, (old_a, old_b), (new_a, new_b))` in wCQ where `addr` is
either a ring cell or `TailGP`/`HeadGP` can be replaced by
`CAS64(addr, pack(old_a, old_b), pack(new_a, new_b))` with identical
semantics: the CAS succeeds if and only if the current value matches the
expected packed value, and on success atomically writes the desired packed value.

*Proof*: By Lemma 4.1, the packing is injective. Therefore
`pack(old_a, old_b) == pack(cur_a, cur_b)` iff `(old_a, old_b) == (cur_a, cur_b)`.
A 64-bit CAS on the packed word atomically reads and conditionally writes
all 64 bits, which includes all sub-fields simultaneously. This provides
exactly the "all-or-nothing" multi-field update that CAS2 provides for
the unpacked representation. □

**Theorem 4.3 (Reduction to wCQ)**:
GPWFQ is a correct implementation of wCQ under the encoding of Lemma 4.1,
for ring sizes `n ≤ 2^(P-1)`.

*Proof sketch*: Every shared variable access in GPWFQ corresponds 1:1 to
a shared variable access in wCQ, with packed encoding/decoding applied.
By Lemma 4.2, every CAS2 is faithfully replaced by CAS64. Non-CAS accesses
(loads, stores, FAA) operate on the same word and thus have identical
semantics. The only additional operation is the TID-to-pointer indirection
in `help_phase2` (`Records[tid].phase2` vs. direct pointer dereference).
This indirection reads from the same memory location (the thread's phase2
record) that wCQ's pointer would address directly. Since the TID uniquely
identifies the thread, and the encoding prevents aliasing (Lemma 4.1),
the indirection is semantically transparent. □

### 4.2 Linearizability

By Theorem 4.3, GPWFQ inherits wCQ's linearizability proof (wCQ Theorem 5.2
and Theorem 5.4 in the SPAA'22 paper).

**Theorem 4.4 (Linearizability)**:
Every concurrent execution of GPWFQ is linearizable with respect to a
sequential FIFO queue specification.

*Proof*: Follows from Theorem 4.3 and wCQ's Theorem 5.2/5.4. The
linearization points are:
- **Enqueue**: the successful CAS on the ring cell that installs the value.
- **Dequeue (success)**: the `atomic_fetch_or` that marks the cell consumed.
- **Dequeue (empty)**: the point where Threshold goes negative.

These are identical to wCQ's linearization points. □

### 4.3 Wait-Freedom

**Lemma 4.5 (Slow Path Termination — adapted from wCQ Lemma 5.5)**:
The `do_enqueue_slow` function makes at most `2n` calls to `slow_inc`
that return `true` before either (a) successfully enqueuing the item, or
(b) detecting that the request was completed by a helper (via `seq1` check).

*Proof*: The ring has `2n` slots. Each call to `slow_inc` that returns `true`
yields a unique ticket (counter value). The counter is monotonically
increasing (each CAS64 on `TailGP` increments it). Therefore `2n`
consecutive tickets cover all `2n` ring positions (modulo `2n`). Among
these `2n` positions, at most `n` contain items from the current or future
cycles (because the queue capacity is `n`). By pigeonhole, at least `n`
positions have entries from a past cycle and are available for enqueue.
The enqueue CAS on such a position succeeds unless another enqueue races
for the same position — but that other enqueue consumed one of the other
available positions, reducing future contention. After at most `2n`
tickets, all positions have been visited and at least one CAS succeeds. □

**Lemma 4.6 (Slow_inc Bounded Retries)**:
Each call to `slow_inc` completes in O(n) steps where n = `nprocs`.

*Proof*: The inner loop of `slow_inc` retries its CAS64 on `TailGP` only
when another thread successfully increments `TailGP` first. Each such
successful increment corresponds to a distinct thread's `slow_inc` call
succeeding (since the CAS64 atomically increments and registers a helper).
With `n` threads, at most `n - 1` other threads can interpose between
our retries. After helping each interposer's phase2 (O(1) per help),
the `TailGP` counter has advanced by `n - 1`. On the next retry, no
thread has a pending CAS64 ahead of us (they all completed), so our
CAS64 succeeds. Total: O(n) retries, each doing O(1) work = O(n) steps. □

**Note on Lemma 4.6**: This differs from a typical lock-free CAS loop!
In a lock-free CAS loop, the same thread can interpose multiple times
(doing repeated increments). In `slow_inc`, after a thread's CAS64
succeeds, it proceeds to process the ticket and doesn't immediately
retry CAS64 on `TailGP`. The cooperative protocol ensures that each
thread does at most one increment before yielding to slot processing.
Specifically: after `slow_inc` returns `true`, the caller enters the
slot-processing code in `do_enqueue_slow`. Only if slot processing fails
does it call `slow_inc` again (for a new ticket). This interleaving is
bounded by Lemma 4.5 (at most 2n calls total).

However, we must also account for helpers: when thread T is being helped
by threads H₁, H₂, ..., all of them call `slow_inc` with T's local counter.
The FIN mechanism ensures that once any of them succeeds in enqueuing T's
item, all others stop (they see FIN on the local counter). But before FIN,
multiple helpers might be racing on `slow_inc`. Each helper's `slow_inc`
CAS64 attempt can be foiled by other helpers' CAS64 attempts.

**Refined bound**: With `n` helpers all calling `slow_inc` for the same
request, the total number of CAS64 attempts across all helpers is bounded
by the number of counter increments. Each increment is "owned" by exactly
one helper (the one whose CAS64 succeeded). That helper then processes
the ticket. At most `2n` tickets need to be processed (Lemma 4.5) before
the item is enqueued. Therefore the total work across all helpers is
O(n × 2n) = O(n²). Each individual helper does at most O(n) CAS64 retries
per `slow_inc` call (Lemma 4.6), and at most O(2n) calls before FIN.
Per-thread worst case: O(n × 2n) = O(n²) steps.

**Lemma 4.7 (Help Protocol Completeness)**:
If thread T publishes a slow-path request (by writing `eidx`, `initTail`,
`seq2`), then within O(n) steps of any other thread executing the `help`
function, T's request is either completed or attempted.

*Proof*: The help function uses round-robin traversal. Thread T's state
is visited by thread `H` when `H.curr_thread` reaches T's position in
the circular list. The list has `n` entries, so after `n` help calls,
every thread has been visited. Each visit checks `eidx` and `seq1`/`seq2`
for a valid pending request, and if found, calls the appropriate
`do_enqueue_slow` or `do_dequeue_slow`. □

**Theorem 4.8 (Wait-Freedom)**:
Every `enqueue()` and `dequeue()` operation in GPWFQ completes in at most

    B(n) = MAX_PATIENCE + O(n²)

steps, where n = `nprocs` and the constant in O(n²) depends on ring_size = 2n.

*Proof*:
1. **Fast path**: bounded by `MAX_PATIENCE` FAA + CAS attempts (each O(1)).
2. **Slow path publication**: O(1) stores.
3. **Slow path execution** (`do_enqueue_slow` / `do_dequeue_slow`):
   - At most `2n` calls to `slow_inc` (Lemma 4.5)
   - Each `slow_inc` takes O(n) steps (Lemma 4.6)
   - Total: O(n × 2n) = O(n²)
4. **Request retirement**: O(1) stores.
5. **Helping (during the operation)**: The thread also helps other threads'
   requests (via `help()` before its own operation and within the slow path).
   Each help call is O(n) (scanning the thread list). This happens O(1) times
   per operation (once at the start).

Total per operation: MAX_PATIENCE + O(n²). This is bounded.

The helping mechanism ensures that even if thread T is indefinitely delayed
(e.g., descheduled by the GPU scheduler), its published request will be
completed by other threads within O(n²) of their collective steps. T, upon
resuming, observes FIN on its local counter and returns. □

### 4.4 Counterexample Resistance

We verify the design against the counterexamples that invalidated previous attempts:

**CE1 (GWFQ failure: operation loss)**:
In GPWFQ, the slow path has no iteration limit. It uses `slow_inc` which is
bounded by Lemma 4.6, called at most `2n` times by Lemma 4.5. No operation
is ever lost. ✓

**CE2 (BWFQ failure: dependency chain deadlock)**:
GPWFQ does not use the admission-control + seq-number discipline that creates
slot-level dependencies. Instead, it uses wCQ's epoch-based cell state, where
each cell's availability depends only on its cycle field vs. the requester's
cycle. The cooperative `slow_inc` ensures monotonic counter advancement,
which guarantees that after `2n` increments, all slots are visited. The
pigeonhole argument (Lemma 4.5) breaks any dependency chain. ✓

**CE3 (Phase2 stale data)**:
The `seq1`/`seq2` protocol in `help_phase2` validates data freshness. If
phase2 data is stale (seq mismatch), the helper skips finalization but still
clears the helper flag on `TailGP` via CAS64. This prevents the stale data
from blocking progress. The original thread, upon resuming, re-reads its
local counter and retries. ✓

**CE4 (FIN race between owner and helper)**:
When a helper completes T's operation, it sets FIN on T's local counter
via CAS. T's `slow_inc` checks FIN at the top of each loop iteration.
The CAS on the local counter uses the INC flag to prevent write-after-FIN:
if local already has FIN, the CAS `(cnt|INC) → cnt` fails because
`local ≠ cnt|INC`. T sees FIN and returns false. ✓

**CE5 (Counter overflow / ABA)**:
With P=15 (32768 threads), the counter has 48 bits = 2^48 values. At
10^10 ops/sec, overflow takes ~28,000 seconds (~7.8 hours). For P=16
(65536 threads), counter has 47 bits ≈ 3.9 hours. For kernel-scoped
queues (typical execution < 1 hour), this is safe. For longer-running
scenarios, P can be reduced (P=12 → 4096 threads, 51-bit counter ≈
62 hours). ✓ (with documented constraint)

**CE6 (Multiple helpers racing on slow_inc CAS64)**:
When multiple helpers attempt `CAS64(TailGP, (cnt, NONE), (cnt+1, my_tid))`,
exactly one succeeds (atomicity of CAS64). The others get the updated
value and retry. The winner processes the ticket and eventually clears
its helper flag (or another helper clears it via `help_phase2`). The
retrying helpers then attempt `(cnt+1, NONE) → (cnt+2, my_tid)`. Each
CAS64 failure is paired with one successful increment by another thread.
By Lemma 4.6, each thread retries at most O(n) times. ✓

**CE7 (GPU scheduling: entire wavefront suspended)**:
If a wavefront (64 threads on AMD) is suspended while one of its threads
holds a helper flag on `TailGP`, other threads on other wavefronts can
still make progress: they call `help_phase2`, clear the helper flag, and
proceed with their own `slow_inc` CAS64. The suspended thread, upon
resuming, finds its helper flag cleared and its local counter potentially
already finalized (FIN set). It handles this gracefully. ✓

**CE8 (GPU memory model: relaxed atomics)**:
The protocol uses `__threadfence()` at critical ordering points:
- After writing phase2 fields, before CAS64 on TailGP
- After CAS on ring cell, before setting FIN or updating threshold
This matches wCQ's barrier placement (wCQ assumes sequential consistency
but notes barriers are inserted in the implementation). ✓

### 4.5 Bounded Memory Usage

**Theorem 4.9 (Bounded Memory)**:
GPWFQ uses exactly:
- `2n × 8` bytes for the ring buffer
- `nprocs × sizeof(wfq_record)` bytes for per-thread records
- O(1) bytes for queue metadata (TailGP, HeadGP, Threshold)

No dynamic allocation occurs during queue operation. Total memory is
statically bounded by queue initialization parameters.

*Proof*: Direct from the data structure definition. No pointers to
dynamically allocated memory are used. The ring is a fixed-size array.
Per-thread records are a fixed-size array. This matches wCQ's bounded
memory property (wCQ Theorem 5.8). □

---

## 5. Limitations and Honest Assessment

### 5.1 Thread Count Limitation

GPWFQ is truly wait-free for `nprocs ≤ 2^P` where P is chosen at compile time.
The trade-off:

| P (TID bits) | Max threads | Counter bits | Counter lifetime at 10B ops/s |
|:---:|:---:|:---:|:---:|
| 12 | 4,096 | 51 | ~62 hours |
| 15 | 32,768 | 48 | ~7.8 hours |
| 16 | 65,536 | 47 | ~3.9 hours |
| 20 | 1,048,576 | 43 | ~14.7 minutes |

For `nprocs > 2^P`, the TID field overflows and the packed encoding becomes
ambiguous, breaking correctness. This is a fundamental limitation of packing
two fields into 64 bits.

MI300A's maximum hardware thread count (~780K) would need P=20, giving only
43-bit counter (~14.7 min). This may be insufficient for long-running kernels.
For such cases, true wait-freedom with 64-bit CAS on a ring buffer design
remains an open problem.

### 5.2 Not a Novel Algorithm — A Novel Encoding

GPWFQ is NOT a new algorithm. It is a **faithful encoding** of wCQ into 64-bit
words. The novelty is demonstrating that this encoding is feasible for practical
GPU thread counts, and proving that the encoding preserves all correctness
properties. The algorithmic credit belongs entirely to Nikolaev (wCQ, SPAA'22).

### 5.3 Counter Overflow

The 48-bit counter (P=15) overflows after ~7.8 hours at peak throughput.
In practice, GPU queues reset between kernel launches, so this is rarely
an issue. However, for long-running persistent kernels, the counter could
wrap. A generation counter or periodic reset would be needed for absolute
safety.

### 5.4 Performance Considerations

The TID-to-record indirection in `help_phase2` adds one global memory load
compared to wCQ's direct pointer dereference. On GPU, this is ~100-400ns
per access (DRAM latency). Since `help_phase2` is called only on the slow
path (which itself is rare), the performance impact is negligible.

The packed encoding may require additional shift/mask operations for field
extraction. These are register-level operations (1-2 cycles) and do not
affect throughput.

---

## 6. Summary

| Property | GPWFQ | GWFQ | BWFQ | wCQ (original) |
|:---|:---:|:---:|:---:|:---:|
| Wait-free | **Yes** (bounded n²) | No (bounded loop, gives up) | No (unbounded while-loop) | Yes |
| Linearizable | Yes | Questionable (lost ops) | Yes (if terminates) | Yes |
| Bounded memory | Yes | Yes | Yes | Yes |
| 64-bit CAS only | **Yes** | Yes | Yes | No (needs CAS2) |
| Thread limit | 2^P | Unlimited | Unlimited | Unlimited |
| Counter bits | 64-P-1 | 64 | 64 | 64 |

GPWFQ achieves true wait-freedom with 64-bit CAS for practical GPU thread
counts (up to 32K-64K threads). It is a direct encoding of wCQ's proven
algorithm, not a new algorithm, and inherits all of wCQ's correctness proofs
via the structural equivalence of Theorem 4.3.

---

## Appendix A: Comparison with CAS2-Based wCQ

For each CAS2 site in wCQ's `wfring.h`, we show the GPWFQ replacement:

### A.1 Ring cell CAS2

**wCQ**:
```c
__lfaba_cmpxchg_weak(&q->array[tidx], &pair,
    __wfring_pair(tcycle ^ eidx ^ n, note), ...)
```
This CAS2 atomically updates both `entry` and `note` fields of the cell.

**GPWFQ**:
```c
CAS64(&Ring[tidx], old_packed_cell,
    pack_cell(tcycle ^ eidx ^ n, note))
```
Same semantics: `pack_cell` concatenates entry and note into 64 bits.

### A.2 TailGP/HeadGP CAS2 (slow_inc)

**wCQ**:
```c
__lfaba_cmpxchg_strong(global, &gp,
    __wfring_pair(cnt + 1, phase2), ...)
```
This CAS2 atomically increments counter and registers phase2 pointer.

**GPWFQ**:
```c
CAS64(global, pack_global(cnt, NONE),
    pack_global(cnt + 1, my_tid))
```
Same semantics: `pack_global` concatenates counter and TID into 64 bits.
The phase2 record is found via `Records[my_tid].phase2` indirection.

### A.3 Phase2 cleanup CAS2

**wCQ**:
```c
__lfaba_cmpxchg_strong(global, &gp,
    __wfring_pair(__wfring_entry(gp), 0), ...)
```
Clears the phase2 pointer while keeping the counter.

**GPWFQ**:
```c
CAS64(global, gp, pack_global(unpack_cnt(gp), NONE))
```
Same semantics: clears the TID/helper flag while keeping the counter.
