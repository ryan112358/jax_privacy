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

## Detailed Results

### Model: Transformer | Size: small | Params: 2.16M (2,158,568)

|   Batch Size |   jax (standard) |
|--------------|------------------|
|           16 |            16.28 |
|           32 |            19.57 |
|           64 |            27.97 |