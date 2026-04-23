#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash concurrent_bfs/run_bfs_sweep.sh mi210
#   bash concurrent_bfs/run_bfs_sweep.sh mi300a
#
# Optional overrides:
#   QUEUES="gwfq sfq" CHUNKS="512 1024 2048" ITER=5 WARM=2 \
#   bash concurrent_bfs/run_bfs_sweep.sh mi210

GPU_FAMILY="${1:-local}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FINAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

resolve_path() {
  local p="$1"
  if [[ "$p" = /* ]]; then
    printf '%s\n' "$p"
  else
    printf '%s\n' "${FINAL_DIR}/${p}"
  fi
}

BFS_BUILD_DIR_RAW="${BFS_BUILD_DIR:-concurrent_bfs/out}"
BFS_GRAPH_DIR_RAW="${BFS_GRAPH_DIR:-concurrent_bfs/graphs}"
LOG_DIR_RAW="${LOG_DIR:-concurrent_bfs/logs}"

BIN_DIR="$(resolve_path "${BFS_BUILD_DIR_RAW}")"
GRAPH_DIR="$(resolve_path "${BFS_GRAPH_DIR_RAW}")"
LOG_DIR="$(resolve_path "${LOG_DIR_RAW}")"

QUEUES_STR="${QUEUES:-gwfq glfq wfq sfq}"
CHUNKS_STR="${CHUNKS:-512 1024 2048 4096 8192}"
BLOCK="${BLOCK:-256}"
SRC="${SRC:-0}"
ITER="${ITER:-10}"
WARM="${WARM:-3}"
CSV_NAME="${CSV_NAME:-bfs_bench_final.csv}"
GPU_NAME="${GPU_NAME:-}"
GRAPHS_STR="${GRAPHS:-}"

read -r -a QUEUES <<< "${QUEUES_STR}"
read -r -a CHUNKS <<< "${CHUNKS_STR}"

if [[ ! -d "${GRAPH_DIR}" ]]; then
  echo "Error: BFS graph directory not found: ${GRAPH_DIR}" >&2
  exit 2
fi

declare -a GRAPH_FILES=()
if [[ -n "${GRAPHS_STR}" ]]; then
  read -r -a GINPUT <<< "${GRAPHS_STR}"
  for g in "${GINPUT[@]}"; do
    if [[ "$g" = /* ]]; then
      GRAPH_FILES+=("$g")
    else
      if [[ "$g" == *.mtx ]]; then
        GRAPH_FILES+=("${GRAPH_DIR}/${g}")
      else
        GRAPH_FILES+=("${GRAPH_DIR}/${g}.mtx")
      fi
    fi
  done
else
  while IFS= read -r -d '' g; do
    GRAPH_FILES+=("$g")
  done < <(find "${GRAPH_DIR}" -maxdepth 1 -type f -name '*.mtx' -print0 | sort -z)
fi

if [[ ${#GRAPH_FILES[@]} -eq 0 ]]; then
  echo "Error: no .mtx graphs found in ${GRAPH_DIR}" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}"

echo "GPU family : ${GPU_FAMILY}"
echo "Binary dir : ${BIN_DIR}"
echo "Graph dir  : ${GRAPH_DIR}"
echo "Queues     : ${QUEUES[*]}"
echo "Threads    : ${CHUNKS[*]}"
echo "Block      : ${BLOCK}"
echo "Src        : ${SRC}"
echo "Warmup     : ${WARM}"
echo "Iters      : ${ITER}"
echo "CSV file   : ${CSV_NAME}"
if [[ -n "${GPU_NAME}" ]]; then
  echo "GPU name   : ${GPU_NAME}"
fi
echo "Logs dir   : ${LOG_DIR}"
echo

export CSV_NAME
export GPU_NAME

for graph_path in "${GRAPH_FILES[@]}"; do
  if [[ ! -f "${graph_path}" ]]; then
    echo "Warning: graph file not found, skipping: ${graph_path}" >&2
    continue
  fi

  graph_file="$(basename "${graph_path}")"
  graph_base="${graph_file%.mtx}"

  for q in "${QUEUES[@]}"; do
    case "$q" in
      gwfq|glfq|wfq|sfq) ;;
      *)
        echo "Warning: unsupported queue '${q}', skipping" >&2
        continue
        ;;
    esac

    bin="${BIN_DIR}/bfs_${q}"
    if [[ ! -x "${bin}" ]]; then
      echo "Error: missing BFS binary for ${q}: ${bin}" >&2
      echo "Build with: make bfs-build QUEUE=${q}" >&2
      exit 2
    fi

    for t in "${CHUNKS[@]}"; do
      echo "============================================================"
      echo "Graph   : ${graph_base}"
      echo "Queue   : ${q}"
      echo "Threads : ${t}"
      echo "Running : ${bin}"
      echo "============================================================"

      log_file="${LOG_DIR}/bfs_${q}_${GPU_FAMILY}_${graph_base}_${t}.log"
      "${bin}" \
        --graph "${graph_path}" \
        --threads "${t}" \
        --block "${BLOCK}" \
        --src "${SRC}" \
        --iters "${ITER}" \
        --warmup "${WARM}" \
        2>&1 | tee "${log_file}"

      echo
    done
  done
done

echo "Done."
