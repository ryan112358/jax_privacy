import jax
import jax.numpy as jnp
from flax import nnx
import time
import argparse
import json
from benchmarks.transformer import Transformer, TransformerConfig, generate_dummy_data
from jax_privacy.clipping import clipped_grad
import optax

def benchmark(mode, config, batch_size, num_iterations=10):
    print(f"Benchmarking mode='{mode}' with config: batch_size={batch_size}")

    key = jax.random.key(0)
    key_model, key_data = jax.random.split(key)

    model = Transformer(config, rngs=nnx.Rngs(key_model))

    # Split model
    graphdef, state, others = nnx.split(model, nnx.Param, ...)

    def loss_fn(state, params):
        """
        Unified loss function for both standard and clipped gradients.

        Args:
            state: The differentiable model parameters (and potentially other state).
            params: The batch of data (inputs, targets).
        """
        # Merge state back into the model graph
        m = nnx.merge(graphdef, state, others)

        x, y = params

        # Check rank of x. If rank 1 (seq_len), expand to (1, seq_len)
        if x.ndim == 1:
            x = jnp.expand_dims(x, 0)

        logits = m(x)

        # If output is (1, seq_len, vocab), reshape correctly
        logits = logits.reshape(-1, logits.shape[-1])
        y = y.reshape(-1)
        one_hot = jax.nn.one_hot(y, logits.shape[-1])
        loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot).mean()
        return loss

    # Generate dummy data
    data = generate_dummy_data(batch_size, config.max_len, config.vocab_size, key_data)
    targets = generate_dummy_data(batch_size, config.max_len, config.vocab_size, key_data)
    batch = (data, targets)

    if mode == 'standard':
        # jax.grad defaults: argnums=0 (state).
        grad_fn = jax.grad(loss_fn)

        @jax.jit
        def train_step(state, batch):
            return grad_fn(state, batch)

    elif mode == 'clipped':
        # clipped_grad defaults: argnums=0 (state), batch_argnums=1 (params/batch).
        grad_clipped = clipped_grad(
            loss_fn,
            l2_clip_norm=1.0,
            keep_batch_dim=True,
        )

        @jax.jit
        def train_step(state, batch):
            return grad_clipped(state, batch)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Warmup
    grads = train_step(state, batch)
    jax.block_until_ready(grads)

    start_time = time.time()
    for _ in range(num_iterations):
        grads = train_step(state, batch)
        jax.block_until_ready(grads)
    end_time = time.time()

    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    throughput = batch_size / avg_time

    print(f"{mode}: Avg time: {avg_time:.4f}s, Throughput: {throughput:.2f} samples/s")

    return {
        "mode": mode,
        "batch_size": batch_size,
        "avg_time": avg_time,
        "throughput": throughput
    }

def main():
    parser = argparse.ArgumentParser(description='Benchmark Transformer gradients.')
    parser.add_argument('--mode', type=str, required=True, choices=['standard', 'clipped'],
                        help='Benchmark mode: standard or clipped')
    args = parser.parse_args()

    config = TransformerConfig(
        vocab_size=1000,
        hidden_size=128,
        num_heads=4,
        num_layers=2,
        max_len=64,
        dropout_rate=0.0
    )

    batch_sizes = [16, 32, 64]
    results = []

    for bs in batch_sizes:
        res = benchmark(args.mode, config, bs)
        results.append(res)

    # Print results as JSON for easy parsing or just reading
    print("RESULTS_JSON=" + json.dumps(results))

if __name__ == "__main__":
    main()
