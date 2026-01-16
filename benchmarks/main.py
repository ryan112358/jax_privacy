import jax
import jax.numpy as jnp
from flax import nnx
import time
import argparse
import json
from benchmarks.transformer import Transformer, TransformerConfig, generate_dummy_data
from jax_privacy.clipping import clipped_grad
from jax_privacy import noise_addition
import optax

def benchmark(mode, config, batch_size, num_iterations=50):
    print(f"Benchmarking mode='{mode}' with config: batch_size={batch_size}")

    key = jax.random.key(0)
    key_model, key_data, key_noise = jax.random.split(key, 3)

    model = Transformer(config, rngs=nnx.Rngs(key_model))

    graphdef, params, others = nnx.split(model, nnx.Param, ...)

    optimizer = optax.adamw(learning_rate=1e-4)
    opt_state = optimizer.init(params)

    data = generate_dummy_data(batch_size, config.max_len, config.vocab_size, key_data)
    targets = generate_dummy_data(batch_size, config.max_len, config.vocab_size, key_data)
    batch = (data, targets)

    def loss_fn(params, batch):
        m = nnx.merge(graphdef, params, others)
        x, y = batch
        if x.ndim == 1:
            x = jnp.expand_dims(x, 0)
        logits = m(x)
        logits = logits.reshape(-1, logits.shape[-1])
        y = y.reshape(-1)
        one_hot = jax.nn.one_hot(y, logits.shape[-1])
        loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot).mean()
        return loss

    if mode == 'standard':
        grad_fn = jax.grad(loss_fn)

        @jax.jit
        def train_step(params, opt_state, batch):
            grads = grad_fn(params, batch)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state

        # Warmup
        params, opt_state = train_step(params, opt_state, batch)
        jax.block_until_ready(params)

        start_time = time.time()
        for _ in range(num_iterations):
            params, opt_state = train_step(params, opt_state, batch)
        jax.block_until_ready(params)
        end_time = time.time()

    elif mode == 'clipped':
        privatizer = noise_addition.gaussian_privatizer(stddev=0.1, prng_key=key_noise)
        noise_state = privatizer.init(params)

        grad_clipped = clipped_grad(
            loss_fn,
            l2_clip_norm=1.0,
            keep_batch_dim=True,
            normalize_by=batch_size
        )

        @jax.jit
        def train_step(params, opt_state, noise_state, batch):
            grads = grad_clipped(params, batch)
            noisy_grads, new_noise_state = privatizer.update(grads, noise_state)
            updates, new_opt_state = optimizer.update(noisy_grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, new_noise_state

        # Warmup
        params, opt_state, noise_state = train_step(params, opt_state, noise_state, batch)
        jax.block_until_ready(params)

        start_time = time.time()
        for _ in range(num_iterations):
            params, opt_state, noise_state = train_step(params, opt_state, noise_state, batch)
        jax.block_until_ready(params)
        end_time = time.time()

    else:
        raise ValueError(f"Unknown mode: {mode}")

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

    print("RESULTS_JSON=" + json.dumps(results))

if __name__ == "__main__":
    main()
