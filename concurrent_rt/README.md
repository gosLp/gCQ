# RT Benchmark and Baselines

This folder contains:

- rt_final.cpp: main queue-based persistent wavefront RT benchmark
- baselines/rt_compaction.cpp: compaction baseline

## Queue RT benchmark (artifact benchmark)

From final/:

make rt QUEUE=gwfq SCENE=1 THREADS=1024 BOUNCES=4

Supported queues in this artifact flow:

- gwfq
- glfq
- wfq
- sfq

## Compaction baseline

The file baselines/rt_compaction.cpp is copied from:

- wfq/native/rtrace/baselines/raytrace_compact.cpp

Build baseline from final/:

hipcc -O3 -std=c++17 concurrent_rt/baselines/rt_compaction.cpp -o out/rt_compaction

Run baseline (example):

./out/rt_compaction

Use the baseline runtime and throughput output for comparison against rt_final.cpp runs.
