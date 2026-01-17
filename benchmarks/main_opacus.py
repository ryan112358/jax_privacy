import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
import argparse
import json
from benchmarks.transformer import TransformerTorch, TransformerConfig, generate_dummy_data as generate_transformer_data
from benchmarks.cnn import CNNTorch, CNNConfig, generate_dummy_data as generate_cnn_data
from benchmarks.state_space import StateSpaceModelTorch, StateSpaceConfig, generate_dummy_data as generate_state_space_data
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import numpy as np

class BenchmarkDataset(TensorDataset):
    def __init__(self, *tensors, total_length):
        super().__init__(*tensors)
        self.tensors = tensors
        self.total_length = total_length
        self.original_len = tensors[0].size(0)

    def __len__(self):
        return self.total_length

    def __getitem__(self, index):
        real_idx = index % self.original_len
        return tuple(tensor[real_idx] for tensor in self.tensors)

def run_benchmark(mode, model_name, config, batch_size, num_iterations=50):
    print(f"Benchmarking model='{model_name}', mode='{mode}' with config: batch_size={batch_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == 'Transformer':
        model = TransformerTorch(config).to(device)
        # Generate data
        d_np = generate_transformer_data(batch_size, config.max_len, config.vocab_size, seed=42)
        t_np = generate_transformer_data(batch_size, config.max_len, config.vocab_size, seed=43)

        d = torch.from_numpy(d_np).to(device)
        t = torch.from_numpy(t_np).to(device)
        data_batch, targets_batch = d, t

        def loss_fn(output, targets):
             return nn.CrossEntropyLoss()(output.view(-1, config.vocab_size), targets.view(-1).long())

    elif model_name == 'CNN':
        model = CNNTorch(config).to(device)
        d_np, t_np = generate_cnn_data(batch_size, config.input_shape, config.num_classes, seed=42)

        # Transpose image to NCHW for PyTorch
        # d_np is NHWC from generate_dummy_data
        d_np = np.transpose(d_np, (0, 3, 1, 2))

        d = torch.from_numpy(d_np).to(device)
        t = torch.from_numpy(t_np).long().to(device)

        data_batch, targets_batch = d, t

        def loss_fn(output, targets):
             return nn.CrossEntropyLoss()(output, targets)

    elif model_name == 'StateSpace':
        model = StateSpaceModelTorch(config).to(device)
        d_np, t_np = generate_state_space_data(batch_size, config.max_len, config.vocab_size, seed=42)

        d = torch.from_numpy(d_np).to(device)
        t = torch.from_numpy(t_np).long().to(device)

        data_batch, targets_batch = d, t

        def loss_fn(output, targets):
             return nn.CrossEntropyLoss()(output.view(-1, config.vocab_size), targets.view(-1))
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if mode == 'clipped':
        model = ModuleValidator.fix(model)
        ModuleValidator.validate(model, strict=False)

    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    if mode == 'standard':
        # Warmup
        optimizer.zero_grad()
        if isinstance(data_batch, (list, tuple)):
            output = model(*data_batch)
        else:
            output = model(data_batch)
        loss = loss_fn(output, targets_batch)
        loss.backward()
        optimizer.step()
        if device.type == 'cuda':
            torch.cuda.synchronize()

        start_time = time.time()
        for _ in range(num_iterations):
            optimizer.zero_grad()
            if isinstance(data_batch, (list, tuple)):
                output = model(*data_batch)
            else:
                output = model(data_batch)
            loss = loss_fn(output, targets_batch)
            loss.backward()
            optimizer.step()

        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()

    elif mode == 'clipped':
        if isinstance(data_batch, (list, tuple)):
            all_tensors = list(data_batch) + [targets_batch]
        else:
            all_tensors = [data_batch, targets_batch]

        dataset = BenchmarkDataset(*all_tensors, total_length=batch_size * (num_iterations + 10))
        dataloader = DataLoader(dataset, batch_size=batch_size)

        privacy_engine = PrivacyEngine()

        noise_multiplier = 0.1 * batch_size
        max_grad_norm = 1.0

        model, optimizer, dataloader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=dataloader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
        )

        iter_loader = iter(dataloader)

        def unpack_batch(batch):
            if len(batch) == 2:
                return batch[0], batch[1]
            elif len(batch) > 2:
                # Assume last is target, rest are inputs
                return batch[:-1], batch[-1]
            else:
                 raise ValueError("Unexpected batch size")

        # Warmup
        batch_tensors = next(iter_loader)
        d, t = unpack_batch(batch_tensors)

        optimizer.zero_grad()
        if isinstance(d, (list, tuple)):
            output = model(*d)
        else:
            output = model(d)
        loss = loss_fn(output, t)
        loss.backward()
        optimizer.step()

        if device.type == 'cuda':
            torch.cuda.synchronize()

        start_time = time.time()
        for _ in range(num_iterations):
            try:
                batch_tensors = next(iter_loader)
            except StopIteration:
                break

            d, t = unpack_batch(batch_tensors)

            optimizer.zero_grad()
            if isinstance(d, (list, tuple)):
                output = model(*d)
            else:
                output = model(d)
            loss = loss_fn(output, t)
            loss.backward()
            optimizer.step()

        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()

    else:
        raise ValueError(f"Unknown mode: {mode}")

    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    throughput = batch_size / avg_time

    print(f"{mode}: Avg time: {avg_time:.4f}s, Throughput: {throughput:.2f} samples/s")

    return {
        "model": model_name,
        "mode": mode,
        "batch_size": batch_size,
        "avg_time": avg_time,
        "throughput": throughput
    }

def main():
    parser = argparse.ArgumentParser(description='Benchmark gradients (PyTorch/Opacus).')
    parser.add_argument('--mode', type=str, required=True, choices=['standard', 'clipped'],
                        help='Benchmark mode: standard or clipped')
    parser.add_argument('--model', type=str, default='Transformer', choices=['Transformer', 'CNN', 'StateSpace'],
                        help='Model to benchmark')
    parser.add_argument('--size', type=str, default='small', choices=['small', 'medium', 'large'],
                        help='Model size for CNN/Diffusion')
    parser.add_argument('--batch_size', type=int, help='Batch size (optional, overrides default list)')
    parser.add_argument('--output_file', type=str, help='Output JSON file to append results')
    parser.add_argument('--max_len', type=int, default=64, help='Max sequence length for Transformer')
    parser.add_argument('--vocab_size', type=int, default=1000)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)

    args = parser.parse_args()

    if args.model == 'Transformer':
        config = TransformerConfig.build(args.size)
    elif args.model == 'CNN':
        config = CNNConfig.build(args.size)
    elif args.model == 'StateSpace':
        config = StateSpaceConfig.build(args.size)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Allow overriding config from CLI if needed (legacy) but sticking to size presets is better.
    # The user didn't ask to remove these CLI args, but standardize config construction.
    # I'll respect `args.size` first.

    batch_sizes = [args.batch_size] if args.batch_size else [32] # Default if not provided

    results = []

    for bs in batch_sizes:
        res = run_benchmark(args.mode, args.model, config, bs)

        # Add max_len to result if Transformer, to match JAX output
        if args.model == 'Transformer':
            res['max_len'] = args.max_len

        results.append(res)

        if args.output_file:
            with open(args.output_file, 'a') as f:
                f.write(json.dumps(res) + '\n')
            print(f"Result appended to {args.output_file}")

    if not args.output_file:
        print("RESULTS_JSON=" + json.dumps(results))

if __name__ == "__main__":
    main()
