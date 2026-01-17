import jax
import jax.numpy as jnp
from flax import nnx
import time
import argparse
import json
import functools
import os
from benchmarks.transformer import Transformer, TransformerConfig, generate_dummy_data as generate_transformer_data
from benchmarks.cnn import CNN, CNNConfig, generate_dummy_data as generate_cnn_data
from benchmarks.state_space import StateSpaceModel, StateSpaceConfig, generate_dummy_data as generate_state_space_data
from jax_privacy.clipping import clipped_grad
from jax_privacy import noise_addition
import optax
import numpy as np

def benchmark(model_class, data_gen_fn, grad_fn, optimizer, config, batch_size, microbatch_size=None, num_iterations=50):
    print(f"Benchmarking with config: batch_size={batch_size}, model={model_class.__name__}")

    key = jax.random.key(0)
    key_model, key_data = jax.random.split(key, 2)

    model = model_class(config, rngs=nnx.Rngs(key_model))

    graphdef, params, others = nnx.split(model, nnx.Param, ...)

    opt_state = optimizer.init(params)

    # data_gen_fn signature adaptation
    # transformer: (bs, seq_len, vocab_size, key) -> (data, targets)
    # cnn: (bs, input_shape, num_classes, key) -> (data, targets)
    # The caller main() should wrap data_gen_fn to only take (bs, key) or pass config inside.
    # In main(), we wrapped it to take (bs, cfg, key) or similar.
    # Let's assume data_gen_fn passed here takes (batch_size, config, seed).
    # Since numpy random is seeded by seed argument in generate_dummy_data.

    batch_numpy = data_gen_fn(batch_size, config, 0)

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
        "model": model_class.__name__
    }

    if microbatch_size is not None:
        res["microbatch_size"] = microbatch_size

    # Add model specific config info
    if isinstance(config, TransformerConfig):
        res.update({
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "num_heads": config.num_heads,
            "num_layers": config.num_layers,
            "max_len": config.max_len
        })
    elif isinstance(config, CNNConfig):
        res.update({
            "input_shape": list(config.input_shape),
            "num_classes": config.num_classes,
            "features": list(config.features),
            "hidden_size": config.hidden_size
        })
    elif isinstance(config, StateSpaceConfig):
        res.update({
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "max_len": config.max_len
        })

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
        model_class = Transformer
        def data_gen_fn(bs, cfg, seed):
            data = generate_transformer_data(bs, cfg.max_len, cfg.vocab_size, seed=seed)
            targets = generate_transformer_data(bs, cfg.max_len, cfg.vocab_size, seed=seed+1)
            return (data, targets)

    elif args.model == 'cnn':
        config = CNNConfig.build(args.size)
        model_class = CNN
        def data_gen_fn(bs, cfg, seed):
            return generate_cnn_data(bs, cfg.input_shape, cfg.num_classes, seed=seed)

    elif args.model == 'state_space':
        config = StateSpaceConfig.build(args.size)
        model_class = StateSpaceModel
        def data_gen_fn(bs, cfg, seed):
            return generate_state_space_data(bs, cfg.max_len, cfg.vocab_size, seed=seed)

    else:
        raise ValueError(f"Unknown model: {args.model}")

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

    res = benchmark(model_class, data_gen_fn, grad_fn, optimizer, config, args.batch_size, args.microbatch_size)
    res['mode'] = args.mode
    res['microbatch_size'] = args.microbatch_size

    # Append to results file
    with open(args.output_file, 'a') as f:
        f.write(json.dumps(res) + '\n')

    print(f"Result appended to {args.output_file}")

if __name__ == "__main__":
    main()
