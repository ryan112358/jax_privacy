import jax
import jax.numpy as jnp
from flax import nnx
import time
import argparse
import json
import functools
import os
from benchmarks.transformer import Transformer, generate_dummy_data as generate_transformer_data
from benchmarks.cnn import CNN, generate_dummy_data as generate_cnn_data
from benchmarks.state_space import StateSpaceModel, StateSpaceConfig, generate_dummy_data as generate_state_space_data
from benchmarks.config import TransformerConfig, CNNConfig
from jax_privacy.clipping import clipped_grad
from jax_privacy import noise_addition
import optax

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
    # Let's assume data_gen_fn passed here takes (batch_size, config, key).

    batch = data_gen_fn(batch_size, config, key_data)

    def loss_fn(params, batch):
        m = nnx.merge(graphdef, params, others)
        x, y = batch
        # Handle 1D input if necessary
        if x.ndim == 1:
            x = jnp.expand_dims(x, 0)

        logits = m(x)

        if logits.ndim == 3: # Sequence task (Transformer)
            logits = logits.reshape(-1, logits.shape[-1])
            y = y.reshape(-1)

        one_hot = jax.nn.one_hot(y, logits.shape[-1])
        loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot).mean()
        return loss

    if isinstance(grad_fn, functools.partial) and grad_fn.func == clipped_grad:
         gradient_computer = grad_fn(loss_fn)
    elif grad_fn == jax.grad:
         gradient_computer = grad_fn(loss_fn)
    else:
         gradient_computer = grad_fn(loss_fn)

    @jax.jit
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
        "microbatch_size": microbatch_size,
        "avg_time": avg_time,
        "throughput": throughput,
        "model": model_class.__name__
    }

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
        if args.size == 'small':
            config = TransformerConfig.small()
        elif args.size == 'medium':
            config = TransformerConfig.medium()
        elif args.size == 'large':
            config = TransformerConfig.large()

        model_class = Transformer
        def data_gen_fn(bs, cfg, key):
            data = generate_transformer_data(bs, cfg.max_len, cfg.vocab_size, key)
            targets = generate_transformer_data(bs, cfg.max_len, cfg.vocab_size, key)
            return (data, targets)

    elif args.model == 'cnn':
        if args.size == 'small':
            config = CNNConfig.small()
        elif args.size == 'medium':
            config = CNNConfig.medium()
        elif args.size == 'large':
            config = CNNConfig.large()

        model_class = CNN
        def data_gen_fn(bs, cfg, key):
            return generate_cnn_data(bs, cfg.input_shape, cfg.num_classes, key)

    elif args.model == 'state_space':
        if args.size == 'small':
            config = StateSpaceConfig.small()
        elif args.size == 'medium':
            config = StateSpaceConfig.medium()
        elif args.size == 'large':
            config = StateSpaceConfig.large()

        model_class = StateSpaceModel
        def data_gen_fn(bs, cfg, key):
            return generate_state_space_data(bs, cfg.max_len, cfg.vocab_size, key=key)

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
