# BFS Benchmarks and Baselines

This folder contains the queue-based BFS benchmark used in the artifact:

- bfs_bench.cpp
- run_bfs_sweep.sh
- graphs/ (local .mtx datasets; not tracked)

## Queue BFS (artifact benchmark)

Build binaries from final/:

make bfs-build-all

Run one case:

make bfs-run \
  QUEUE=gwfq \
  BFS_GRAPH=concurrent_bfs/graphs/ak2010.mtx \
  BFS_THREADS=512 \
  BFS_BLOCK=256 \
  BFS_ITERS=3 \
  BFS_WARMUP=1

## Gunrock baseline BFS

Install and build Gunrock following the official project instructions.

For Eg. on AMD Hip machines use the `hip-develop` branch on gunrock repo : [gunrock](https://github.com/gunrock/gunrock.git)

Then run Gunrock BFS on Matrix Market input using -m (market format). Example:

/path/to/gunrock/build/bin/bfs -m /absolute/path/to/graph.mtx

Use this baseline timing output for comparison against the queue-based BFS runs above.
