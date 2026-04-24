# Final Benchmarks and Correctness

This directory contains three main GPU queue workflows:

- Raytracing (RT) benchmark (`concurrent_rt/rt_final.cpp`)
- Throughput benchmark (`throughout_bench.cpp`)
- History generation + FIFO linearizability checking (`correctness/history.cpp`, `correctness/lincheck/main.go`)
- BFS benchmark and sweeps (`concurrent_bfs/bfs_bench.cpp`, `concurrent_bfs/run_bfs_sweep.sh`)

All commands below are run from this directory (`wfq/final`).

## Queue names used here

- `SFQ`: Scogland-Feng Queue
- `WFQ`: G-WFQ-YMC (GPU adaptation of the YMC CPU design, as referenced in our paper)
- `GWFQ`: G-WFQ
- `GLFQ`: G-LFQ

Only these four queues are supported through this Makefile: `gwfq | glfq | wfq | sfq`.

## 1) Raytracing (RT) benchmark

Build + run:

```bash
make rt QUEUE=gwfq SCENE=0 THREADS=1024 BOUNCES=4
```

Key RT variables:

- `QUEUE`: `gwfq | glfq | wfq | sfq`
- `SCENE`: `0` (complex) or `1` (cornell)
- `THREADS`: worker threads per tile
- `BOUNCES`: max reflection depth
- `GPU_NAME`: optional label override; when unset, GPU name is auto-detected from HIP
- `MAX_THREADS`: compile-time ceiling used by RT build (default is `THREADS`)
- `QUEUE_N`: compile-time queue capacity macro for RT (default `32768`)
- `EXTRA_FLAGS`: extra compile flags appended to RT build

## 2) Throughput benchmark

Build + run (normal mode):

```bash
make tb QUEUE=gwfq
```

Build + run with per-test FIFO check enabled:

```bash
make tb-run-fifo QUEUE=gwfq
```

Queue support for throughput harness:

- `gwfq`, `glfq`, `wfq`, `sfq`

Key throughput variables:

- `TB_FIFO_MODE`: `0` (off) or `1` (on)
- `GPU_NAME`: optional label override for console/CSV; default is auto-detected GPU name
- `TB_RUN_MS`: timed run duration per case
- `TB_WARMUP_MS`: warmup duration per case
- `TB_CHUNK_OPS`: operations per kernel launch chunk
- `TB_BLOCK_SIZE`: GPU block size
- `TB_FIFO_OPS_PER_THREAD`: FIFO-check work per producer thread
- `TB_ONLY_BALANCED`: `1` to run only balanced test
- `TB_ONLY_SPLIT`: `1` to run only split producer/consumer test
- `TB_CSV_FILE`: output CSV path (default `benchmark_results.csv`)
- `TB_EXTRA_FLAGS`: extra compile flags appended to throughput build

Example tuned run:

```bash
make tb-run-fifo QUEUE=sfq TB_RUN_MS=1000 TB_WARMUP_MS=200 TB_CHUNK_OPS=128 TB_FIFO_OPS_PER_THREAD=32
```

## 3) Emit histories (JSONL)

Generate a single queue history JSONL:

```bash
make hist-run QUEUE=gwfq
```

Generate histories for all supported queues (`gwfq, glfq, wfq, sfq`):

```bash
make hist-run-all
```

All generated histories are written under:

- `correctness/histories`

Default output naming:

- `correctness/histories/history_<queue>_m<mode>_t<threads>_o<ops>.jsonl`

History tuning variables:

- `HIST_THREADS`: thread count used by history kernel
- `HIST_OPS`: operations per thread
- `HIST_MODE`: `0` alternating (`enq,deq`) or `1` split roles
- `HIST_PRODUCER_PERCENT`: used only when `HIST_MODE=1`
- `HIST_BLOCK_SIZE`: GPU block size
- `HIST_OUT_DIR`: output directory (default `correctness/histories`)
- `HIST_EXTRA_FLAGS`: extra compile flags for history binary

Example:

```bash
make hist-run QUEUE=wfq HIST_MODE=1 HIST_THREADS=128 HIST_OPS=32 HIST_PRODUCER_PERCENT=25
```

## 4) Linearizability check (Porcupine, Go)

Prerequisites:

- Go installed and available on `PATH`
- One or more history files already present in `correctness/histories`

Check FIFO linearizability for one history:

```bash
make lincheck LINCHECK_HISTORY=correctness/histories/history_gwfq_m0_t64_o16.jsonl
```

Check and also emit an HTML visualization:

```bash
make lincheck-html \
	LINCHECK_HISTORY=correctness/histories/history_gwfq_m0_t64_o16.jsonl \
	LINCHECK_HTML=correctness/histories/history_gwfq_m0_t64_o16.html
```

Expected checker output status:

- `Linearizable (FIFO)` -> success (exit code 0)
- `Not linearizable (FIFO)` or `Unknown (timeout)` -> failure (exit code 1)

## 5) Quick make help

```bash
make rt-help
```

## 6) BFS benchmark and sweeps

Build one BFS binary for a selected queue:

```bash
make bfs-build QUEUE=gwfq
```

Build all BFS queue binaries (`gwfq, glfq, wfq, sfq`) into:

- `concurrent_bfs/out`

Graph datasets are expected under `concurrent_bfs/graphs`.
The `.mtx` files are intentionally not tracked; see `concurrent_bfs/graphs/README.md` for required file names.

```bash
make bfs-build-all
```

Run one BFS case with a selected queue and graph:

```bash
make bfs-run \
	QUEUE=wfq \
	BFS_GRAPH=concurrent_bfs/graphs/road_usa.mtx \
	BFS_THREADS=8192 \
	BFS_BLOCK=256 \
	BFS_ITERS=10 \
	BFS_WARMUP=3
```

Run an automated sweep over all graphs in `concurrent_bfs/graphs`, all queues, and default thread counts (`512 1024 2048 4096 8192`):

```bash
make bfs-sweep
```

Sweep tuning variables:

- `BFS_QUEUES`: queue list (default `gwfq glfq wfq sfq`)
- `BFS_CHUNKS`: thread sizes (default `512 1024 2048 4096 8192`)
- `BFS_GRAPHS`: optional graph list (for example `ak2010 road_usa`)
- `GPU_NAME`: optional GPU label override for BFS console/CSV rows
- `BFS_GRAPH_DIR`: graph directory (default `concurrent_bfs/graphs`)
- `BFS_BUILD_DIR`: BFS binary directory (default `concurrent_bfs/out`)
- `BFS_LOG_DIR`: sweep log directory (default `concurrent_bfs/logs`)
- `BFS_CSV_NAME`: BFS CSV file path passed via environment
- `BFS_BLOCK`, `BFS_SRC_VERTEX`, `BFS_ITERS`, `BFS_WARMUP`
- `BFS_GPU_FAMILY`: label used in log filenames (for example `mi210` or `mi300a`)

Example custom sweep:

```bash
make bfs-sweep \
	BFS_GPU_FAMILY=mi300a \
	BFS_QUEUES="wfq sfq" \
	BFS_GRAPHS="ak2010 road_usa" \
	BFS_CHUNKS="512 1024 2048 4096 8192" \
	BFS_CSV_NAME=concurrent_bfs/bfs_bench_final.csv
```

## 7) Baselines

Baseline usage notes are documented in:

- `concurrent_bfs/README.md` for Gunrock BFS baseline (`-m` market input mode)
- `concurrent_rt/README.md` for the RT compaction baseline

RT compaction baseline source is in:

- `concurrent_rt/baselines/rt_compaction.cpp`

## 8) License

This artifact is released under the MIT License.

- `LICENSE`

