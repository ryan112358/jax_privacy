# Benchmark Results

Note: Benchmarks run on CPU unless GPU is available in the environment.

| Batch Size | Standard Time (s) | Standard Throughput (samples/s) | Clipped Time (s) | Clipped Throughput (samples/s) |
|---|---|---|---|---|
| 16 | 0.0983 | 162.82 | 0.0688 | 232.46 |
| 32 | 0.2247 | 142.44 | 0.1548 | 206.66 |
| 64 | 0.2975 | 215.14 | 0.3190 | 200.66 |

## Summary

The benchmark compares the performance of standard gradient computation using `jax.grad` versus privacy-preserving gradient computation using `jax_privacy.clipped_grad` on a simple Transformer model using Flax NNX.

Each mode was run separately to ensure isolation. The results show the time and throughput for computing gradients.
