import jax
import jax.numpy as jnp
from flax import nnx
import time
import argparse
import json
import functools
from benchmarks.transformer import Transformer, generate_dummy_data as generate_transformer_data
from benchmarks.cnn import CNN, generate_dummy_data as generate_cnn_data
from benchmarks.config import TransformerConfig, CNNConfig
from jax_privacy.clipping import clipped_grad
from jax_privacy import noise_addition
import optax

def benchmark(model_class, data_gen_fn, grad_fn, optimizer, config, batch_size, num_iterations=50):
    print(f"Benchmarking with config: batch_size={batch_size}")

    key = jax.random.key(0)
    key_model, key_data = jax.random.split(key, 2)

    model = model_class(config, rngs=nnx.Rngs(key_model))

    graphdef, params, others = nnx.split(model, nnx.Param, ...)

    opt_state = optimizer.init(params)

    batch = data_gen_fn(batch_size, config, key_data)

    def loss_fn(params, batch):
        m = nnx.merge(graphdef, params, others)
        x, y = batch
        # Handle 1D input if necessary (though not expected with current generators)
        if x.ndim == 1:
            x = jnp.expand_dims(x, 0)
        logits = m(x)
        logits = logits.reshape(-1, logits.shape[-1])
        y = y.reshape(-1)
        one_hot = jax.nn.one_hot(y, logits.shape[-1])
        loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot).mean()
        return loss

    # Bind loss_fn to grad_fn if it's jax.grad or clipped_grad wrapper that expects func
    # The requirement says "consumes a grad_fn (like jax.grad, or functools.partial(jax_privacy.clipped_grad, ...)"
    # So we assume grad_fn takes (fun, ...) and returns a gradient function.
    # OR it consumes a ready-to-use gradient function?
    # "consumes a grad_fn (like jax.grad ...)" usually means a function transformation.
    # So we apply it to loss_fn.

    if isinstance(grad_fn, functools.partial) and grad_fn.func == clipped_grad:
         # partial(clipped_grad, ...)
         # clipped_grad(f, ...) returns a function that computes grads
         gradient_computer = grad_fn(loss_fn)
    elif grad_fn == jax.grad:
         gradient_computer = grad_fn(loss_fn)
    else:
         # Fallback, assume it acts like jax.grad
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

    return {
        "batch_size": batch_size,
        "avg_time": avg_time,
        "throughput": throughput
    }

def main():
    parser = argparse.ArgumentParser(description='Benchmark Transformer and CNN gradients.')
    parser.add_argument('--mode', type=str, required=True, choices=['standard', 'clipped'],
                        help='Benchmark mode: standard or clipped')
    parser.add_argument('--model', type=str, default='transformer', choices=['transformer', 'cnn'],
                        help='Model to benchmark: transformer or cnn')
    parser.add_argument('--size', type=str, default='small', choices=['small', 'medium', 'large'],
                        help='Model size: small, medium, large')
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

    else:
        raise ValueError(f"Unknown model: {args.model}")

    if args.mode == 'standard':
        grad_fn = jax.grad
        optimizer = optax.adamw(learning_rate=1e-4)
    elif args.mode == 'clipped':
        # Create a privatizer
        # We need a key for the privatizer.
        # But wait, the privatizer state is part of optimizer state in optax chain.
        # So we can create it here.
        key = jax.random.key(42)
        key_noise = jax.random.split(key)[0]

        # NOTE: In previous main.py, keep_batch_dim=True and normalize_by=batch_size were used.
        # We need to capture batch_size in the grad_fn construction if it's constant,
        # but here we iterate over batch_sizes.
        # The clipped_grad function accepts normalize_by.
        # We might need to handle batch_size dynamically or re-create grad_fn inside loop.
        # The prompt says: "Modify main so that it consumes a grad_fn ... and an optimizer"
        # Since batch_size varies, we should probably pass the partial, but normalize_by depends on batch_size.
        # If we pass a partial that expects (loss_fn), then inside benchmark we call it.
        # But clipped_grad(loss_fn, ..., normalize_by=BS).
        # We can pass a factory? Or just update it inside the loop.
        # But `benchmark` signature takes `grad_fn`.
        # Maybe `benchmark` should take `grad_fn_factory`?
        # Or simpler: The user prompt says "Use a single shared config ...".
        # It also says "Modify main so that it consumes a grad_fn... and an optimizer".
        # If I strictly follow "consumes a grad_fn", then `benchmark` takes a fixed `grad_fn`.
        # If `grad_fn` is fixed, then `normalize_by` must be fixed or handled inside.
        # `jax_privacy.clipped_grad` allows `normalize_by` to be an int.
        # If I want to sweep batch sizes, I need to change `normalize_by`.
        # So I should probably move the loop over batch sizes to `main` and call `benchmark` for each,
        # constructing `grad_fn` appropriately for each batch size.
        pass # Logic implemented below

    batch_sizes = [16, 32, 64]
    results = []

    for bs in batch_sizes:
        if args.mode == 'standard':
            grad_fn = jax.grad
            optimizer = optax.adamw(learning_rate=1e-4)
        elif args.mode == 'clipped':
             grad_fn = functools.partial(
                clipped_grad,
                l2_clip_norm=1.0,
                keep_batch_dim=True,
                normalize_by=bs # Use current batch size
            )
             privatizer = noise_addition.gaussian_privatizer(stddev=0.1, prng_key=jax.random.key(1337))
             optimizer = optax.chain(
                 privatizer,
                 optax.adamw(learning_rate=1e-4)
             )

        res = benchmark(model_class, data_gen_fn, grad_fn, optimizer, config, bs)
        res['mode'] = args.mode # Add mode back for consistency
        results.append(res)

    print("RESULTS_JSON=" + json.dumps(results))

if __name__ == "__main__":
    main()
