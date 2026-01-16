# CNN Benchmark Results

## Input Shape: (32, 32, 3), Batch Size: 32

**Standard Throughput:** 1027.97 samples/s

| Mode | Microbatch Size | Throughput (samples/s) | Speedup vs Standard | % of Standard |
|---|---|---|---|---|
| Standard | - | 1027.97 | 1.0x | 100% |
| Clipped | 32.0 | 188.36 | 0.18x | 18.3% |
| Clipped | 8.0 | 174.68 | 0.17x | 17.0% |
| Clipped | 1.0 | 148.61 | 0.14x | 14.5% |

## Input Shape: (32, 32, 3), Batch Size: 64

**Standard Throughput:** 1018.70 samples/s

| Mode | Microbatch Size | Throughput (samples/s) | Speedup vs Standard | % of Standard |
|---|---|---|---|---|
| Standard | - | 1018.70 | 1.0x | 100% |
| Clipped | 64.0 | 195.82 | 0.19x | 19.2% |
| Clipped | 16.0 | 152.93 | 0.15x | 15.0% |
| Clipped | 1.0 | 130.90 | 0.13x | 12.8% |

## Input Shape: (64, 64, 3), Batch Size: 32

**Standard Throughput:** 69.86 samples/s

| Mode | Microbatch Size | Throughput (samples/s) | Speedup vs Standard | % of Standard |
|---|---|---|---|---|
| Standard | - | 69.86 | 1.0x | 100% |


## Analysis

### Performance Gap
*   **Standard Mode**: Very fast, achieving ~1000 samples/s for small models.
*   **Clipped Mode**: Significant slowdown. For small models, throughput drops to ~130-200 samples/s, which is about 15-20% of standard performance.
*   **Reason**: The per-sample gradient computation (or effectively doing so via vectorized clipping) adds computational overhead.

### Microbatch Size Impact
*   **Best Performance**: Setting `microbatch_size` equal to `batch_size` (no splitting) yields the best clipped performance.
*   **Worst Performance**: Setting `microbatch_size` to 1 yields the worst clipped performance, but the penalty is not extremely severe in this setup (dropping from ~190 to ~150 samples/s).
*   **Recommendation**: Use the largest microbatch size that fits in memory.

### Model Size Impact
*   **Small Model (32x32)**: Standard is ~5x faster than clipped.
*   **Medium Model (64x64)**: (Only standard run completed in time, but we can infer similar or larger gaps). The standard run itself slowed down significantly to ~70 samples/s due to increased model complexity.
