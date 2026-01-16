# Benchmark Results

Note: Benchmarks run on CPU unless GPU is available in the environment.

| Batch Size | Standard Time (s) | Standard Throughput (samples/s) | Clipped Time (s) | Clipped Throughput (samples/s) |
|---|---|---|---|---|
| 16 | 0.0809 | 197.84 | 0.0830 | 192.82 |
| 32 | 0.1217 | 262.95 | 0.1460 | 219.24 |
| 64 | 0.2126 | 301.10 | 0.2796 | 228.90 |

## Summary

The benchmark compares the performance of standard gradient computation (with optimizer update) versus differentially private training (gradient clipping + noise addition + optimizer update) on a simple Transformer model using Flax NNX.

Each mode was run separately to ensure isolation. The results show the time and throughput for one training step (averaged over 50 iterations).

As expected, DP training (`clipped` mode) incurs overhead, which becomes more pronounced at larger batch sizes where the cost of per-example gradient clipping dominates.
