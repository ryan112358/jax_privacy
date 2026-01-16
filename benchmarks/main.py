import jax
import jax.numpy as jnp
from flax import nnx
import time
import pandas as pd
from benchmarks.transformer import Transformer, TransformerConfig, generate_dummy_data
from jax_privacy.clipping import clipped_grad
import optax
import functools

def benchmark(config, batch_size, num_iterations=10):
    print(f"Benchmarking with config: {config.__dict__}, batch_size={batch_size}")

    key = jax.random.key(0)
    key_model, key_data = jax.random.split(key)

    model = Transformer(config, rngs=nnx.Rngs(key_model))

    # Generate dummy data
    data = generate_dummy_data(batch_size, config.max_len, config.vocab_size, key_data)
    targets = generate_dummy_data(batch_size, config.max_len, config.vocab_size, key_data)

    # 1. Standard jax.grad
    print("Benchmarking jax.grad...")

    def loss_fn(model, x, y):
        logits = model(x)
        logits = logits.reshape(-1, logits.shape[-1])
        y = y.reshape(-1)
        one_hot = jax.nn.one_hot(y, logits.shape[-1])
        loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot).mean()
        return loss

    @nnx.jit
    def train_step_standard(model, x, y):
        grad_fn = nnx.grad(loss_fn)
        grads = grad_fn(model, x, y)
        return grads

    # Warmup
    grads_standard = train_step_standard(model, data, targets)
    # Block on grads to ensure compilation and execution is done
    jax.tree.map(lambda x: x.block_until_ready(), grads_standard)

    start_time = time.time()
    for _ in range(num_iterations):
        grads_standard = train_step_standard(model, data, targets)
        jax.tree.map(lambda x: x.block_until_ready(), grads_standard)
    end_time = time.time()

    total_time_standard = end_time - start_time
    avg_time_standard = total_time_standard / num_iterations
    throughput_standard = batch_size / avg_time_standard

    print(f"Standard: Avg time: {avg_time_standard:.4f}s, Throughput: {throughput_standard:.2f} samples/s")

    # 2. jax_privacy.clipped_grad (Flax stateless API)
    print("Benchmarking jax_privacy.clipped_grad...")

    # Split model into params and others (e.g. rng counts, though we shouldn't have any dynamic ones now)
    graphdef, params, others = nnx.split(model, nnx.Param, ...)

    def loss_fn_stateless_args(params, x, y):
        m = nnx.merge(graphdef, params, others)
        logits = m(x)

        logits = logits.reshape(-1, logits.shape[-1])
        y = y.reshape(-1)
        one_hot = jax.nn.one_hot(y, logits.shape[-1])
        loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot).mean()
        return loss

    grad_clipped = clipped_grad(
        loss_fn_stateless_args,
        argnums=0,
        batch_argnums=(1, 2),
        l2_clip_norm=1.0,
        keep_batch_dim=True,
    )

    @jax.jit
    def train_step_clipped(params, x, y):
        grads = grad_clipped(params, x, y)
        return grads

    # Warmup
    grads_clipped = train_step_clipped(params, data, targets)
    jax.tree.map(lambda x: x.block_until_ready(), grads_clipped)

    start_time = time.time()
    for _ in range(num_iterations):
        grads_clipped = train_step_clipped(params, data, targets)
        jax.tree.map(lambda x: x.block_until_ready(), grads_clipped)
    end_time = time.time()

    total_time_clipped = end_time - start_time
    avg_time_clipped = total_time_clipped / num_iterations
    throughput_clipped = batch_size / avg_time_clipped

    print(f"Clipped: Avg time: {avg_time_clipped:.4f}s, Throughput: {throughput_clipped:.2f} samples/s")

    return {
        "config": config.__dict__,
        "batch_size": batch_size,
        "standard_time": avg_time_standard,
        "standard_throughput": throughput_standard,
        "clipped_time": avg_time_clipped,
        "clipped_throughput": throughput_clipped,
        "overhead_factor": avg_time_clipped / avg_time_standard
    }

def main():
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
        res = benchmark(config, bs)
        results.append(res)

    # Save results to markdown
    with open("benchmarks/results.md", "w") as f:
        f.write("# Benchmark Results\n\n")
        f.write("Note: Benchmarks run on CPU unless GPU is available in the environment.\n\n")
        f.write("| Batch Size | Standard Time (s) | Standard Throughput (samples/s) | Clipped Time (s) | Clipped Throughput (samples/s) | Overhead Factor |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['batch_size']} | {r['standard_time']:.4f} | {r['standard_throughput']:.2f} | {r['clipped_time']:.4f} | {r['clipped_throughput']:.2f} | {r['overhead_factor']:.2f} |\n")

    print("Benchmark finished. Results saved to benchmarks/results.md")

if __name__ == "__main__":
    main()
