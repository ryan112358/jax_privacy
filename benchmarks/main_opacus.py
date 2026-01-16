import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
import argparse
import json
from benchmarks.transformer_torch import Transformer, generate_dummy_data
from benchmarks.config import TransformerConfig
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

class BenchmarkDataset(TensorDataset):
    def __init__(self, data, targets, total_length):
        super().__init__(data, targets)
        self.data = data
        self.targets = targets
        self.total_length = total_length
        self.original_len = data.size(0)

    def __len__(self):
        return self.total_length

    def __getitem__(self, index):
        real_idx = index % self.original_len
        return self.data[real_idx], self.targets[real_idx]

def run_benchmark(mode, model_name, config, batch_size, num_iterations=50):
    print(f"Benchmarking model='{model_name}', mode='{mode}' with config: batch_size={batch_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == 'Transformer':
        model = Transformer(config).to(device)
        # Generate data
        d = generate_transformer_data(batch_size, config.max_len, config.vocab_size, seed=42).to(device)
        t = generate_transformer_data(batch_size, config.max_len, config.vocab_size, seed=43).to(device)
        data_batch, targets_batch = d, t

        def loss_fn(output, targets):
             return nn.CrossEntropyLoss()(output.view(-1, config.vocab_size), targets.view(-1))

    elif model_name == 'CNN':
        model = CNN(config).to(device)
        d, t = generate_cnn_data(batch_size, config.input_shape, config.num_classes, seed=42)
        data_batch, targets_batch = d.to(device), t.to(device)

        def loss_fn(output, targets):
             return nn.CrossEntropyLoss()(output, targets)
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
        output = model(data_batch)
        loss = loss_fn(output, targets_batch)
        loss.backward()
        optimizer.step()
        if device.type == 'cuda':
            torch.cuda.synchronize()

        start_time = time.time()
        for _ in range(num_iterations):
            optimizer.zero_grad()
            output = model(data_batch)
            loss = loss_fn(output, targets_batch)
            loss.backward()
            optimizer.step()

        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()

    elif mode == 'clipped':
        dataset = BenchmarkDataset(data_batch, targets_batch, batch_size * (num_iterations + 10))
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

        # Warmup
        d, t = next(iter_loader)
        optimizer.zero_grad()
        output = model(d)
        loss = loss_fn(output, t)
        loss.backward()
        optimizer.step()

        if device.type == 'cuda':
            torch.cuda.synchronize()

        start_time = time.time()
        for _ in range(num_iterations):
            try:
                d, t = next(iter_loader)
            except StopIteration:
                break

            optimizer.zero_grad()
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
    parser.add_argument('--size', type=str, default='small', choices=['small', 'medium', 'large'],
                        help='Model size: small, medium, large')
    args = parser.parse_args()

    if args.size == 'small':
        config = TransformerConfig.small()
    elif args.size == 'medium':
        config = TransformerConfig.medium()
    elif args.size == 'large':
        config = TransformerConfig.large()

    batch_sizes = [16, 32, 64]
    results = []

    for bs in batch_sizes:
        res = run_benchmark(args.mode, args.model, config, bs)
        results.append(res)

    print("RESULTS_JSON=" + json.dumps(results))

if __name__ == "__main__":
    main()
