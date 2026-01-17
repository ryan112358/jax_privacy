import jax
import jax.numpy as jnp
from flax import nnx
import time
import argparse
import json
import functools
import os
from .transformer import TransformerConfig
from .cnn import CNNConfig
from .state_space import StateSpaceConfig
from jax_privacy.clipping import clipped_grad
from jax_privacy import noise_addition
import optax
import numpy as np

def benchmark(grad_fn, optimizer, config, batch_size, microbatch_size=None, num_iterations=50):
    print(f"Benchmarking with config: batch_size={batch_size}")

    key = jax.random.key(0)
    key_model, key_data = jax.random.split(key, 2)

    model = config.make(rngs=nnx.Rngs(key_model))

    graphdef, params, others = nnx.split(model, nnx.Param, ...)

    opt_state = optimizer.init(params)

    batch_numpy = config.generate_dummy_data(batch_size, seed=0)

    # Convert numpy batch to JAX array
    if isinstance(batch_numpy, tuple):
        batch = tuple(jnp.array(x) for x in batch_numpy)
    else:
        batch = jnp.array(batch_numpy)

    def loss_fn(params, batch):
        m = nnx.merge(graphdef, params, others)
        x, y = batch

        if isinstance(x, (tuple, list)):
            logits = m(*x)
        else:
            logits = m(x)

        if jnp.issubdtype(y.dtype, jnp.floating):
             # MSE loss for diffusion
             loss = jnp.mean((logits - y) ** 2)
        else:
            if logits.ndim == 3: # Sequence task (Transformer)
                logits = logits.reshape(-1, logits.shape[-1])
                y = y.reshape(-1)

            one_hot = jax.nn.one_hot(y, logits.shape[-1])
            loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot).mean()
        return loss

    gradient_computer = grad_fn(loss_fn)

    @jax.jit(donate_argnums=(0, 1))
    def train_step(params, opt_state, batch):
        grads = gradient_computer(params, batch)
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

    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    throughput = batch_size / avg_time

    print(f"Avg time: {avg_time:.4f}s, Throughput: {throughput:.2f} samples/s")

    res = {
        "batch_size": batch_size,
        "avg_time": avg_time,
        "throughput": throughput,
        "model": config.__class__.__name__.replace("Config", ""),
    }

    if microbatch_size is not None:
        res["microbatch_size"] = microbatch_size

    return res

def main():
    parser = argparse.ArgumentParser(description='Benchmark Transformer and CNN gradients.')
    parser.add_argument('--mode', type=str, required=True, choices=['standard', 'clipped'],
                        help='Benchmark mode: standard or clipped')
    parser.add_argument('--model', type=str, default='transformer', choices=['transformer', 'cnn', 'state_space'],
                        help='Model to benchmark: transformer, cnn, or state_space')
    parser.add_argument('--size', type=str, default='small', choices=['small', 'medium', 'large'],
                        help='Model size: small, medium, large')

    # Args from origin/main
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size')
    parser.add_argument('--microbatch_size', type=int, help='Microbatch size for clipped mode')
    parser.add_argument('--output_file', type=str, default='results.json')

    args = parser.parse_args()

    if args.model == 'transformer':
        config = TransformerConfig.build(args.size)
    elif args.model == 'cnn':
        config = CNNConfig.build(args.size)
    elif args.model == 'state_space':
        config = StateSpaceConfig.build(args.size)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    config.framework = 'jax'

    if args.mode == 'standard':
        grad_fn = jax.grad
        optimizer = optax.adamw(learning_rate=1e-4)
    elif args.mode == 'clipped':
         grad_fn = functools.partial(
            clipped_grad,
            l2_clip_norm=1.0,
            keep_batch_dim=True,
            normalize_by=args.batch_size,
            microbatch_size=args.microbatch_size
        )
         privatizer = noise_addition.gaussian_privatizer(stddev=0.1, prng_key=jax.random.key(1337))
         optimizer = optax.chain(
             privatizer,
             optax.adamw(learning_rate=1e-4)
         )

    res = benchmark(grad_fn, optimizer, config, args.batch_size, args.microbatch_size)
    res['mode'] = args.mode
    res['microbatch_size'] = args.microbatch_size

    # Append to results file
    with open(args.output_file, 'a') as f:
        f.write(json.dumps(res) + '\n')

    print(f"Result appended to {args.output_file}")

if __name__ == "__main__":
    main()
