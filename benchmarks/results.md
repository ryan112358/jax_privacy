# Benchmark Results

## System Information
- **Platform**: Linux-6.8.0-x86_64-with-glibc2.39
- **Processor**: x86_64
- **Python Version**: 3.12.12
- **GPU**: N/A (CPU only or nvidia-smi not found)

## Discussion / Interpretation

The following results demonstrate the performance characteristics of JAX and PyTorch (Opacus) across different models and sizes.

**Key Observations:**
1. **Throughput Scaling:** Throughput generally increases with batch size until memory saturation.
2. **Privacy Overhead:** 'Clipped' mode (DP-SGD) incurs significant overhead compared to 'Standard' training. This is due to the per-sample gradient computation required for privacy, preventing certain batch optimizations.
3. **Framework Comparison:** Performance varies between JAX and PyTorch depending on the model and batch size. JAX often shows strong performance on TPUs/GPUs with XLA compilation, while PyTorch/Opacus has its own optimizations.
4. **Memory Limits:** Larger models and batch sizes eventually lead to OOM errors, cutting off the curves.

## Opacus Throughput Experiments (Small Models, Batch Size 32)

We evaluated the throughput of `standard` vs `clipped` (DP-SGD) modes using `main_opacus.py` for small variants of CNN, Transformer, and StateSpace models.

| Model       | Mode     | Throughput (samples/s) | Speedup (vs Standard) |
| :---        | :---     | :---                   | :---                  |
| CNN         | Standard | 1270.55                | 1.00x                 |
| CNN         | Clipped  | 230.63                 | 0.18x (5.5x slower)   |
| Transformer | Standard | 60.44                  | 1.00x                 |
| Transformer | Clipped  | 39.76                  | 0.66x (1.5x slower)   |
| StateSpace  | Standard | 73.14                  | 1.00x                 |
| StateSpace  | Clipped  | 4.51                   | 0.06x (16.2x slower)  |

### Interpretation of Overheads

The primary cause of the throughput drop in `clipped` mode is the computation of **per-sample gradients**, which are necessary for clipping individual gradients before aggregation.

*   **Transformer:** Shows the lowest overhead (1.5x). Opacus is highly optimized for `Linear` layers (which dominate Transformers) using batch outer products, avoiding full per-sample gradient materialization.
*   **CNN:** Shows moderate overhead (5.5x). `Conv2d` layers require expanding the input (e.g., via `unfold`/im2col) to compute per-sample gradients, increasing memory bandwidth and compute significantly.
*   **StateSpace:** Shows extreme overhead (>16x). This implementation uses a `Conv1d` layer with a kernel size equal to the sequence length (256) and depthwise grouping. Calculating per-sample gradients for such a large kernel convolution is computationally expensive and memory-intensive in the current Opacus implementation.

## Detailed Results

### Model: Transformer | Size: small | Params: 2.16M (2,158,568)

|   Batch Size |   jax (standard) |
|--------------|------------------|
|           16 |            16.28 |
|           32 |            19.57 |
|           64 |            27.97 |
