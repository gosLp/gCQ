HIPCXX ?= hipcc

RT_SRC ?= concurrent_rt/rt_final.cpp
RT_BUILD_DIR ?= out
TB_SRC ?= throughout_bench.cpp
TB_BUILD_DIR ?= out
HIST_SRC ?= correctness/history.cpp
HIST_BUILD_DIR ?= out
BFS_SRC ?= concurrent_bfs/bfs_bench.cpp
BFS_BUILD_DIR ?= concurrent_bfs/out
BFS_GRAPH_DIR ?= concurrent_bfs/graphs
BFS_SWEEP_SCRIPT ?= concurrent_bfs/run_bfs_sweep.sh

# Runtime args
QUEUE ?= gwfq
SCENE ?= 0
THREADS ?= 1024
BOUNCES ?= 4
GPU_NAME ?=

# Throughput-benchmark args
TB_FIFO_MODE ?= 0
TB_RUN_MS ?= 500
TB_WARMUP_MS ?= 100
TB_CHUNK_OPS ?= 64
TB_BLOCK_SIZE ?= 256
TB_FIFO_OPS_PER_THREAD ?= 64
TB_ONLY_BALANCED ?= 0
TB_ONLY_SPLIT ?= 0
TB_CSV_FILE ?= benchmark_results.csv
TB_EXTRA_FLAGS ?=

# History generator args
HIST_THREADS ?= 64
HIST_OPS ?= 16
HIST_MODE ?= 0
HIST_PRODUCER_PERCENT ?= 50
HIST_BLOCK_SIZE ?= 256
HIST_OUT_DIR ?= correctness/histories
HIST_EXTRA_FLAGS ?=

# Linearizability checker args
LINCHECK_DIR ?= correctness/lincheck
LINCHECK_HISTORY ?= $(HIST_JSONL)
LINCHECK_HTML ?=

# BFS args
BFS_GRAPH ?= $(BFS_GRAPH_DIR)/delaunay_n21.mtx
BFS_THREADS ?= 8192
BFS_BLOCK ?= 256
BFS_SRC_VERTEX ?= 0
BFS_ITERS ?= 10
BFS_WARMUP ?= 3
BFS_EXTRA_FLAGS ?=
BFS_QUEUES ?= gwfq glfq wfq sfq
BFS_CHUNKS ?= 512 1024 2048 4096 8192
BFS_GRAPHS ?=
BFS_LOG_DIR ?= concurrent_bfs/logs
BFS_CSV_NAME ?= bfs_bench_final.csv
BFS_GPU_FAMILY ?= local

# Compile-time args
MAX_THREADS ?= $(THREADS)
QUEUE_N ?= 32768
EXTRA_FLAGS ?=

ifeq ($(QUEUE),gwfq)
RT_QUEUE_DEF := -DUSE_GWFQ
RT_QUEUE_TAG := gwfq
else ifeq ($(QUEUE),glfq)
RT_QUEUE_DEF := -DUSE_GLFQ
RT_QUEUE_TAG := glfq
else ifeq ($(QUEUE),wfq)
RT_QUEUE_DEF := -DUSE_WFQ
RT_QUEUE_TAG := wfq
else ifeq ($(QUEUE),sfq)
RT_QUEUE_DEF := -DUSE_SFQ
RT_QUEUE_TAG := sfq
else
$(error Unsupported QUEUE='$(QUEUE)'. Use one of: gwfq glfq wfq sfq)
endif

RT_BIN := $(RT_BUILD_DIR)/rt_$(RT_QUEUE_TAG)
TB_BIN := $(TB_BUILD_DIR)/tb_$(RT_QUEUE_TAG)_fifo$(TB_FIFO_MODE)
HIST_BIN := $(HIST_BUILD_DIR)/history_$(RT_QUEUE_TAG)
HIST_JSONL := $(HIST_OUT_DIR)/history_$(RT_QUEUE_TAG)_m$(HIST_MODE)_t$(HIST_THREADS)_o$(HIST_OPS).jsonl
BFS_BIN := $(BFS_BUILD_DIR)/bfs_$(RT_QUEUE_TAG)

.PHONY: all rt rt-build rt-run rt-clean rt-help tb tb-build tb-run tb-build-fifo tb-run-fifo tb-help hist hist-build hist-run hist-run-all hist-help lincheck lincheck-html bfs bfs-build bfs-build-all bfs-run bfs-sweep

all: rt

rt: rt-run

rt-build:
	@mkdir -p $(RT_BUILD_DIR)
	@echo "[RT] Building $(RT_BIN) (queue=$(QUEUE), max_threads=$(MAX_THREADS), qN=$(QUEUE_N))"
	$(HIPCXX) -O3 -std=c++17 \
		$(RT_QUEUE_DEF) \
		-DRT_MAX_THREADS=$(MAX_THREADS) \
		-DRT_QUEUE_N=$(QUEUE_N) \
		$(EXTRA_FLAGS) \
		$(RT_SRC) -o $(RT_BIN)

rt-run: rt-build
	@echo "[RT] Running $(RT_BIN) scene=$(SCENE) bounces=$(BOUNCES) threads=$(THREADS)"
	GPU_NAME="$(GPU_NAME)" $(RT_BIN) $(SCENE) $(BOUNCES) $(THREADS)

tb: tb-run

tb-build:
	@mkdir -p $(TB_BUILD_DIR)
	@echo "[TB] Building $(TB_BIN) (queue=$(QUEUE), fifo_mode=$(TB_FIFO_MODE))"
	$(HIPCXX) -O3 -std=c++17 \
		$(RT_QUEUE_DEF) \
		-DRUN_MS=$(TB_RUN_MS) \
		-DWARMUP_MS=$(TB_WARMUP_MS) \
		-DCHUNK_OPS=$(TB_CHUNK_OPS) \
		-DBLOCK_SIZE=$(TB_BLOCK_SIZE) \
		-DONLY_BALANCED=$(TB_ONLY_BALANCED) \
		-DONLY_SPLIT=$(TB_ONLY_SPLIT) \
		-DMODE_FIFO=$(TB_FIFO_MODE) \
		-DFIFO_OPS_PER_THREAD=$(TB_FIFO_OPS_PER_THREAD) \
		-DCSV_FILE=\"$(TB_CSV_FILE)\" \
		$(TB_EXTRA_FLAGS) \
		$(TB_SRC) -o $(TB_BIN)

tb-run: tb-build
	@echo "[TB] Running $(TB_BIN)"
	GPU_NAME="$(GPU_NAME)" $(TB_BIN)

tb-build-fifo: TB_FIFO_MODE=1
tb-build-fifo: tb-build

tb-run-fifo: TB_FIFO_MODE=1
tb-run-fifo: tb-run

hist: hist-run

hist-build:
	@mkdir -p $(HIST_BUILD_DIR)
	@if [ "$(QUEUE)" != "gwfq" ] && [ "$(QUEUE)" != "glfq" ] && [ "$(QUEUE)" != "wfq" ] && [ "$(QUEUE)" != "sfq" ]; then \
		echo "[HIST] QUEUE=$(QUEUE) not supported by correctness/history.cpp (use gwfq|glfq|wfq|sfq)"; \
		exit 2; \
	fi
	@echo "[HIST] Building $(HIST_BIN) (queue=$(QUEUE))"
	$(HIPCXX) -O3 -std=c++17 \
		$(RT_QUEUE_DEF) \
		-DHIST_THREADS=$(HIST_THREADS) \
		-DHIST_OPS=$(HIST_OPS) \
		-DHIST_MODE=$(HIST_MODE) \
		-DHIST_PRODUCER_PERCENT=$(HIST_PRODUCER_PERCENT) \
		-DHIST_BLOCK_SIZE=$(HIST_BLOCK_SIZE) \
		$(HIST_EXTRA_FLAGS) \
		$(HIST_SRC) -o $(HIST_BIN)

hist-run: hist-build
	@mkdir -p $(HIST_OUT_DIR)
	@echo "[HIST] Writing $(HIST_JSONL)"
	$(HIST_BIN) $(HIST_JSONL) $(HIST_THREADS) $(HIST_OPS) $(HIST_MODE) $(HIST_PRODUCER_PERCENT)

hist-run-all:
	@mkdir -p $(HIST_OUT_DIR)
	@for q in gwfq glfq wfq sfq; do \
		echo "[HIST] queue=$$q"; \
		$(MAKE) --no-print-directory hist-run QUEUE=$$q \
			HIST_THREADS=$(HIST_THREADS) \
			HIST_OPS=$(HIST_OPS) \
			HIST_MODE=$(HIST_MODE) \
			HIST_PRODUCER_PERCENT=$(HIST_PRODUCER_PERCENT) \
			HIST_BLOCK_SIZE=$(HIST_BLOCK_SIZE) \
			HIST_OUT_DIR=$(HIST_OUT_DIR) \
			HIST_EXTRA_FLAGS='$(HIST_EXTRA_FLAGS)'; \
		done

bfs: bfs-run

bfs-build:
	@mkdir -p $(BFS_BUILD_DIR)
	@echo "[BFS] Building $(BFS_BIN) (queue=$(QUEUE))"
	$(HIPCXX) -O3 -std=c++17 \
		$(RT_QUEUE_DEF) \
		$(BFS_EXTRA_FLAGS) \
		$(BFS_SRC) -o $(BFS_BIN)

bfs-build-all:
	@mkdir -p $(BFS_BUILD_DIR)
	@for q in gwfq glfq wfq sfq; do \
		echo "[BFS] queue=$$q"; \
		$(MAKE) --no-print-directory bfs-build QUEUE=$$q \
			BFS_BUILD_DIR='$(BFS_BUILD_DIR)' \
			BFS_EXTRA_FLAGS='$(BFS_EXTRA_FLAGS)'; \
		done

bfs-run: bfs-build
	@if [ ! -f "$(BFS_GRAPH)" ]; then \
		echo "[BFS] graph not found: $(BFS_GRAPH)"; \
		exit 2; \
	fi
	@echo "[BFS] Running $(BFS_BIN) graph=$(BFS_GRAPH) threads=$(BFS_THREADS) block=$(BFS_BLOCK)"
	CSV_NAME=$(BFS_CSV_NAME) GPU_NAME="$(GPU_NAME)" $(BFS_BIN) \
		--graph "$(BFS_GRAPH)" \
		--threads "$(BFS_THREADS)" \
		--block "$(BFS_BLOCK)" \
		--src "$(BFS_SRC_VERTEX)" \
		--iters "$(BFS_ITERS)" \
		--warmup "$(BFS_WARMUP)"

bfs-sweep: bfs-build-all
	@mkdir -p $(BFS_LOG_DIR)
	@QUEUES="$(BFS_QUEUES)" \
	CHUNKS="$(BFS_CHUNKS)" \
	GRAPHS="$(BFS_GRAPHS)" \
	BFS_BUILD_DIR="$(BFS_BUILD_DIR)" \
	BFS_GRAPH_DIR="$(BFS_GRAPH_DIR)" \
	BLOCK="$(BFS_BLOCK)" \
	SRC="$(BFS_SRC_VERTEX)" \
	ITER="$(BFS_ITERS)" \
	WARM="$(BFS_WARMUP)" \
	GPU_NAME="$(GPU_NAME)" \
	LOG_DIR="$(BFS_LOG_DIR)" \
	CSV_NAME="$(BFS_CSV_NAME)" \
	bash "$(BFS_SWEEP_SCRIPT)" "$(BFS_GPU_FAMILY)"

lincheck:
	@if ! command -v go >/dev/null 2>&1; then \
		echo "[LINCHECK] Go is not installed or not on PATH"; \
		exit 2; \
	fi
	@if [ ! -f "$(LINCHECK_HISTORY)" ]; then \
		echo "[LINCHECK] history file not found: $(LINCHECK_HISTORY)"; \
		exit 2; \
	fi
	@echo "[LINCHECK] Checking $(LINCHECK_HISTORY)"
	@cd $(LINCHECK_DIR) && go run . ../../$(LINCHECK_HISTORY)

lincheck-html:
	@if ! command -v go >/dev/null 2>&1; then \
		echo "[LINCHECK] Go is not installed or not on PATH"; \
		exit 2; \
	fi
	@if [ ! -f "$(LINCHECK_HISTORY)" ]; then \
		echo "[LINCHECK] history file not found: $(LINCHECK_HISTORY)"; \
		exit 2; \
	fi
	@if [ -z "$(LINCHECK_HTML)" ]; then \
		echo "[LINCHECK] set LINCHECK_HTML, e.g. LINCHECK_HTML=correctness/histories/check.html"; \
		exit 2; \
	fi
	@mkdir -p $$(dirname "$(LINCHECK_HTML)")
	@echo "[LINCHECK] Checking $(LINCHECK_HISTORY), writing $(LINCHECK_HTML)"
	@cd $(LINCHECK_DIR) && go run . ../../$(LINCHECK_HISTORY) ../../$(LINCHECK_HTML)

rt-clean:
	@rm -rf $(RT_BUILD_DIR)
	@echo "Removed $(RT_BUILD_DIR)"

rt-help:
	@echo "RT benchmark targets (run inside final/):"
	@echo "  make rt QUEUE=gwfq SCENE=0 THREADS=1024 BOUNCES=4"
	@echo ""
	@echo "Throughput benchmark targets:"
	@echo "  make tb QUEUE=gwfq"
	@echo "  make tb-run-fifo QUEUE=gwfq"
	@echo ""
	@echo "Variables:"
	@echo "  QUEUE        : gwfq | glfq | wfq | sfq"
	@echo "  SCENE        : 0 (complex) | 1 (cornell)"
	@echo "  THREADS      : worker threads per tile"
	@echo "  BOUNCES      : max reflection depth"
	@echo "  MAX_THREADS  : compile-time ceiling (default = THREADS)"
	@echo "  QUEUE_N      : compile-time queue capacity (default 32768)"
	@echo "  EXTRA_FLAGS  : extra compile flags, e.g. -DSFQ_QUEUE_LENGTH=65536"
	@echo "  GPU_NAME     : optional GPU label override for RT/TB/BFS outputs"
	@echo "  TB_FIFO_MODE : 0 (off) | 1 (on)"
	@echo "  TB_RUN_MS, TB_WARMUP_MS, TB_CHUNK_OPS, TB_BLOCK_SIZE"
	@echo "  TB_FIFO_OPS_PER_THREAD : FIFO checker work per producer"
	@echo "  TB_ONLY_BALANCED, TB_ONLY_SPLIT"
	@echo ""
	@echo "History targets:"
	@echo "  make hist-run QUEUE=gwfq"
	@echo "  make hist-run-all"
	@echo ""
	@echo "History vars:"
	@echo "  HIST_THREADS, HIST_OPS"
	@echo "  HIST_MODE : 0 (alternating) | 1 (split)"
	@echo "  HIST_PRODUCER_PERCENT : used when HIST_MODE=1"
	@echo "  HIST_BLOCK_SIZE"
	@echo "  HIST_OUT_DIR : default correctness/histories"
	@echo ""
	@echo "Linearizability targets:"
	@echo "  make lincheck LINCHECK_HISTORY=correctness/histories/history_gwfq_m0_t64_o16.jsonl"
	@echo "  make lincheck-html LINCHECK_HISTORY=... LINCHECK_HTML=correctness/histories/check.html"
	@echo ""
	@echo "BFS targets:"
	@echo "  make bfs-build QUEUE=gwfq"
	@echo "  make bfs-build-all"
	@echo "  make bfs-run QUEUE=gwfq BFS_GRAPH=concurrent_bfs/graphs/delaunay_n21.mtx"
	@echo "  make bfs-sweep"
	@echo ""
	@echo "BFS vars:"
	@echo "  BFS_BUILD_DIR : default concurrent_bfs/out"
	@echo "  BFS_GRAPH_DIR : default concurrent_bfs/graphs"
	@echo "  BFS_QUEUES    : default 'gwfq glfq wfq sfq'"
	@echo "  BFS_CHUNKS    : default '512 1024 2048 4096 8192'"
	@echo "  BFS_GRAPHS    : optional graph list, e.g. 'ak2010 road_usa'"
	@echo "  BFS_BLOCK, BFS_SRC_VERTEX, BFS_ITERS, BFS_WARMUP"
	@echo "  BFS_LOG_DIR, BFS_CSV_NAME, BFS_GPU_FAMILY"
