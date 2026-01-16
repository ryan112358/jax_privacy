# Benchmark Results

Note: Benchmarks run on CPU unless GPU is available in the environment.

## JAX / Flax NNX

| Batch Size | Standard Time (s) | Standard Throughput (samples/s) | Clipped Time (s) | Clipped Throughput (samples/s) |
|---|---|---|---|---|
| 16 | 0.0437 | 366.20 | 0.0688 | 232.41 |
| 32 | 0.0632 | 506.62 | 0.1733 | 184.69 |
| 64 | 0.1145 | 558.96 | 0.2714 | 235.82 |

## PyTorch / Opacus

| Batch Size | Standard Time (s) | Standard Throughput (samples/s) | Clipped Time (s) | Clipped Throughput (samples/s) |
|---|---|---|---|---|
| 16 | 0.0360 | 444.09 | 0.0883 | 181.27 |
| 32 | 0.0574 | 557.74 | 0.1340 | 238.88 |
| 64 | 0.1031 | 620.70 | 0.2285 | 280.14 |

## Summary

The benchmark compares the performance of standard gradient computation (with optimizer update) versus differentially private training (gradient clipping + noise addition + optimizer update) on a simple Transformer model.

- **JAX (Flax NNX)**: `clipped` mode overhead varies. At batch size 32, throughput dropped significantly compared to standard.
- **PyTorch (Opacus)**: Standard training is slightly faster than JAX in this setup. Opacus shows good scaling for clipped training, outperforming JAX at larger batch sizes (32 and 64).

Each mode was run separately to ensure isolation. The results show the time and throughput for one training step (averaged over 50 iterations).
