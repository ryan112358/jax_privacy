import jax
import jax.numpy as jnp
from flax import nnx
import time
import argparse
import json
import os
from benchmarks.transformer import Transformer, TransformerConfig, generate_dummy_data as generate_transformer_data
from benchmarks.cnn import CNN, CNNConfig, generate_dummy_data as generate_cnn_data
from jax_privacy.clipping import clipped_grad
from jax_privacy import noise_addition
import optax

def benchmark(model_class, data_gen_fn, mode, config, batch_size, microbatch_size=None, num_iterations=50):
    print(f"Benchmarking mode='{mode}' with config: batch_size={batch_size}, "
          f"microbatch_size={microbatch_size}, model={model_class.__name__}")

    key = jax.random.key(0)
    key_model, key_data, key_noise = jax.random.split(key, 3)

    model = model_class(config, rngs=nnx.Rngs(key_model))

    graphdef, params, others = nnx.split(model, nnx.Param, ...)

    optimizer = optax.adamw(learning_rate=1e-4)
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

        # Flatten logits for loss calculation if needed, but for standard classification:
        # logits: (batch, num_classes) or (batch, seq, vocab)
        # targets: (batch,) or (batch, seq)

        if logits.ndim == 3: # Sequence task (Transformer)
            logits = logits.reshape(-1, logits.shape[-1])
            y = y.reshape(-1)

        # Ensure y is one-hot or handled by loss.
        # generate_dummy_data returns indices.
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
            normalize_by=batch_size,
            microbatch_size=microbatch_size
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

    # Construct result dict
    res = {
        "mode": mode,
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

    return res

def main():
    parser = argparse.ArgumentParser(description='Benchmark Transformer and CNN gradients.')
    parser.add_argument('--mode', type=str, required=True, choices=['standard', 'clipped'],
                        help='Benchmark mode: standard or clipped')
    parser.add_argument('--model', type=str, default='transformer', choices=['transformer', 'cnn'],
                        help='Model to benchmark: transformer or cnn')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size')
    parser.add_argument('--microbatch_size', type=int, help='Microbatch size for clipped mode')
    parser.add_argument('--output_file', type=str, default='results.json')

    # Transformer args
    parser.add_argument('--vocab_size', type=int, default=1000)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--max_len', type=int, default=64)

    args = parser.parse_args()

    if args.model == 'transformer':
        config = TransformerConfig(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            max_len=args.max_len,
            dropout_rate=0.0
        )
        model_class = Transformer
        def data_gen_fn(bs, cfg, key):
            data = generate_transformer_data(bs, cfg.max_len, cfg.vocab_size, key)
            targets = generate_transformer_data(bs, cfg.max_len, cfg.vocab_size, key)
            return (data, targets)

    elif args.model == 'cnn':
        # CNN Config uses defaults for now as they weren't exposed in args,
        # but we could expose them if needed. Keeping it simple as per merge instructions.
        config = CNNConfig(
            input_shape=(32, 32, 3),
            num_classes=10,
            features=(32, 64),
            kernel_size=(3, 3),
            hidden_size=128
        )
        model_class = CNN
        def data_gen_fn(bs, cfg, key):
            return generate_cnn_data(bs, cfg.input_shape, cfg.num_classes, key)

    else:
        raise ValueError(f"Unknown model: {args.model}")

    # In clipped mode, pass microbatch_size if provided.
    microbatch_size = None
    if args.mode == 'clipped' and args.microbatch_size is not None:
        microbatch_size = args.microbatch_size

    res = benchmark(model_class, data_gen_fn, args.mode, config, args.batch_size, microbatch_size=microbatch_size)

    # Append to results file
    with open(args.output_file, 'a') as f:
        f.write(json.dumps(res) + '\n')

    print(f"Result appended to {args.output_file}")

if __name__ == "__main__":
    main()
