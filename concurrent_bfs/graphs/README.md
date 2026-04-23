# BFS Graph Datasets

This directory is used by BFS benchmarks and sweeps.

Dataset files with extension .mtx are intentionally not tracked by Git.
Add them locally using the file names below.

Expected graph files:

- ak2010.mtx
- belgium_osm.mtx
- delaunay_n21.mtx
- delaunay_n24.mtx
- europe_osm.mtx
- hollywood-2009.mtx
- kron_g500-logn21.mtx
- roadNet-CA.mtx
- road_usa.mtx

Examples:

Run one BFS case on a specific graph:

```bash
make bfs-run \
  QUEUE=gwfq \
  BFS_GRAPH=concurrent_bfs/graphs/ak2010.mtx \
  BFS_THREADS=512 \
  BFS_BLOCK=256 \
  BFS_ITERS=3 \
  BFS_WARMUP=1
```

Run sweep over selected graphs:

```bash
make bfs-sweep \
  BFS_GRAPHS="ak2010 road_usa" \
  BFS_CHUNKS="512 1024 2048 4096 8192"
```
