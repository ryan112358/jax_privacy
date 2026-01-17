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


## JAXPR Analysis

To investigate the performance difference, we captured the JAXPR (JAX's intermediate representation) for both the standard and clipped gradient computations.

### Standard Training JAXPR

In standard training, the gradient is computed on the mean of the loss. The key observation is how gradients for the weights (convolution kernels) are computed.

```
dl:f32[3,3,32,64] = conv_general_dilated[
  batch_group_count=1
  dimension_numbers=ConvDimensionNumbers(lhs_spec=(3, 0, 1, 2), rhs_spec=(3, 0, 1, 2), out_spec=(2, 3, 0, 1))
  feature_group_count=1
  ...
] r dh
```

Here, `dl` represents the gradient with respect to the weights of one of the layers.
*   **Structure**: This is a single convolution operation that reduces over the batch dimension immediately. The input `r` (activations) and `dh` (gradients from the layer above) both have the batch dimension.
*   **Efficiency**: The reduction happens *inside* this `conv_general_dilated` call (or matrix multiplication). This is highly efficient because it avoids storing per-sample gradients. It accumulates the sum directly.

### Clipped Training JAXPR

In clipped training (using `jax_privacy.clipped_grad`), the operation is vectorized over the batch dimension using `vmap`.

```
ga:f32[3,3,32,128] = conv_general_dilated[
  batch_group_count=1
  dimension_numbers=ConvDimensionNumbers(lhs_spec=(3, 0, 1, 2), rhs_spec=(3, 0, 1, 2), out_spec=(2, 3, 0, 1))
  feature_group_count=2
  ...
] fy fz
```

At first glance, this looks similar, but notice the `feature_group_count=2` (which corresponds to the batch size in our JAXPR capture experiment).
*   **Structure**: The `vmap` transformation pushes the batch dimension into the `feature_group_count` (or batch_group_count depending on configuration) of the convolution.
*   **Output Size**: The output `ga` has shape `[3, 3, 32, 128]`. In our tiny experiment, `128` is actually `64 (features) * 2 (batch_size)`. This means it is producing `batch_size` separate gradients for the weights.
*   **Overhead**:
    1.  **Memory**: Instead of outputting one weight gradient tensor `(K, K, Cin, Cout)`, it outputs `(Batch, K, K, Cin, Cout)`. This drastically increases memory write bandwidth.
    2.  **Computation**: While the FLOP count is similar, the lack of immediate reduction means we lose the arithmetic intensity benefits of the standard "Batch-to-Weight" gradient convolution.
    3.  **Post-Processing**: After producing these per-sample gradients, the system must compute the norm of *each* one (reading them all back), clip them, and then sum them (reading and writing again).

### Clarification on Gradient Propagation

A common question is: "Don't we need to pass unreduced gradients to the previous layer regardless of the mode?"

*   **Activation Gradients**: Yes, the gradient with respect to the *input/activations* of a layer (often denoted $\delta$ or $\nabla_x L$) must preserve the batch dimension to be passed to the previous layer. This step is identical in both standard and clipped training.
*   **Weight Gradients**: The performance difference arises from the computation of the gradient with respect to the *weights* ($\nabla_w L$).
    *   **Standard**: $\nabla_w L$ is computed by summing over the batch dimension immediately (e.g., `activations^T @ output_grads`). The result has no batch dimension.
    *   **Clipped**: We must compute per-sample weight gradients ($\nabla_{w}^{(i)} L$) first. This prevents the immediate summation and forces the materialization of a large tensor with an extra batch dimension, as seen in the JAXPR above.

### Conclusion

The performance gap is caused by the fundamental requirement of DP-SGD to access per-sample gradients.
1.  **Standard Backprop**: `Activations (B) x Gradients (B) -> Summed Weight Gradient (1)` (Reduction happens during compute).
2.  **Clipped Backprop**: `Activations (B) x Gradients (B) -> Per-Sample Weight Gradients (B) -> Clip -> Sum`.

The intermediate step of materializing `B` weight gradients creates a massive memory bottleneck, especially for CNNs where weights can be large and the convolution operation is computationally intensive. The `microbatch_size` argument mitigates the peak memory usage but does not remove the fundamental need to materialize these gradients before reduction.
