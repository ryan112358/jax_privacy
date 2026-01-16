# Benchmark Results

Note: Benchmarks run on CPU unless GPU is available in the environment.

## Transformer

| Batch Size | Standard Time (s) | Standard Throughput (samples/s) | Clipped Time (s) | Clipped Throughput (samples/s) |
|---|---|---|---|---|
| 16 | 0.0811 | 197.28 | 0.0936 | 170.96 |
| 32 | 0.1248 | 256.40 | 0.1721 | 185.92 |
| 64 | 0.2464 | 259.70 | 0.3044 | 210.24 |

## CNN

| Batch Size | Standard Time (s) | Standard Throughput (samples/s) | Clipped Time (s) | Clipped Throughput (samples/s) |
|---|---|---|---|---|
| 16 | 0.0134 | 1198.02 | 0.0573 | 279.20 |
| 32 | 0.0359 | 890.24 | 0.1210 | 264.42 |
| 64 | 0.0644 | 993.28 | 0.2280 | 280.76 |

## Summary

The benchmark compares the performance of standard gradient computation (with optimizer update) versus differentially private training (gradient clipping + noise addition + optimizer update) on two models: a simple Transformer and a basic CNN, both using Flax NNX.
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

### Observations

1.  **Overhead of Privacy:** As expected, DP training (`clipped` mode) incurs overhead due to the per-example gradient clipping.
2.  **Transformer vs. CNN:**
    *   The **Transformer** shows a moderate performance drop in clipped mode.
    *   The **CNN** shows a much larger relative drop in throughput when switching to clipped mode (e.g., from ~1198 to ~279 samples/s at batch size 16). This is likely because the standard forward/backward pass for the small CNN is extremely fast, making the overhead of per-example gradient computation (vectorized or not) proportionally much larger.
3.  **Batch Size Scaling:** Throughput generally increases or stays stable with batch size for standard training, but the relationship is more complex in clipped mode due to memory and computation scaling characteristics of the specific `clipped_grad` implementation (e.g., whether it uses `vmap` or other techniques).
Each mode was run separately to ensure isolation. The results show the time and throughput for one training step (averaged over 50 iterations).
