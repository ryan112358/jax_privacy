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

# Dataset that wraps a single batch and returns it repeatedly (effectively)
# Since we want to benchmark "num_iterations" of updates on the same data.
# However, Opacus expects to iterate over a DataLoader.
# We will create a dataset that has (batch_size * num_iterations) length,
# but the data is just the same batch repeated.
# Actually, better to just generate random data for the whole run if memory permits,
# OR just repeat the same data to save memory but allow iteration.
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
        # Return item from the original batch using modulo
        real_idx = index % self.original_len
        return self.data[real_idx], self.targets[real_idx]

def benchmark(mode, config, batch_size, num_iterations=50):
    print(f"Benchmarking mode='{mode}' with config: batch_size={batch_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Transformer(config).to(device)

    if mode == 'clipped':
        model = ModuleValidator.fix(model)
        ModuleValidator.validate(model, strict=False)

    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Generate one batch of dummy data
    # To match main.py, we use seed logic but here we just use fixed seed
    data_batch = generate_dummy_data(batch_size, config.max_len, config.vocab_size, seed=42).to(device)
    targets_batch = generate_dummy_data(batch_size, config.max_len, config.vocab_size, seed=43).to(device)

    criterion = nn.CrossEntropyLoss()

    if mode == 'standard':
        # Simple loop
        # Warmup
        optimizer.zero_grad()
        output = model(data_batch)
        loss = criterion(output.view(-1, config.vocab_size), targets_batch.view(-1))
        loss.backward()
        optimizer.step()
        if device.type == 'cuda':
            torch.cuda.synchronize()

        start_time = time.time()
        for _ in range(num_iterations):
            optimizer.zero_grad()
            output = model(data_batch)
            loss = criterion(output.view(-1, config.vocab_size), targets_batch.view(-1))
            loss.backward()
            optimizer.step()

        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()

    elif mode == 'clipped':
        # For Opacus, we need a DataLoader
        # We simulate num_iterations steps.
        # Opacus updates per step (batch).
        dataset = BenchmarkDataset(data_batch, targets_batch, batch_size * num_iterations)
        dataloader = DataLoader(dataset, batch_size=batch_size) # default shuffle=False

        privacy_engine = PrivacyEngine()

        # Match noise parameters to main.py
        # main.py: noise_std = 0.1, clip_norm = 1.0, normalize_by = batch_size
        # The noise added to the average gradient is N(0, 0.1^2).
        # Opacus adds noise to the sum gradient: N(0, (noise_multiplier * clip_norm)^2)
        # Then averages by dividing by batch_size.
        # Resulting noise on average gradient: N(0, (noise_multiplier * clip_norm / batch_size)^2).
        # We want noise_multiplier * clip_norm / batch_size = 0.1
        # clip_norm = 1.0 -> noise_multiplier = 0.1 * batch_size

        noise_multiplier = 0.1 * batch_size
        max_grad_norm = 1.0

        model, optimizer, dataloader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=dataloader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
        )

        # Warmup (1 step)
        # We need to manually do one step or create a small loader
        # Let's just run one step on the data_batch manually?
        # No, make_private wraps the optimizer, so we must use the wrapped optimizer.
        # But wrapped optimizer expects step to be called in context?
        # Ideally we just iterate the dataloader.

        # To support warmup + measurement, let's create a loader that has num_iterations + 1 batches.
        dataset = BenchmarkDataset(data_batch, targets_batch, batch_size * (num_iterations + 1))
        dataloader = DataLoader(dataset, batch_size=batch_size)

        # We need to re-wrap because I created a new loader?
        # Actually Opacus wraps the loader just to check batch size/sample rate.
        # If I replace the loader, I should be fine as long as batch_size is same.
        # But make_private returns a new loader. I should use that if possible.
        # Or I can just pass the new loader to make_private again? No, double wrapping model.

        # Simpler: Just make the original dataset larger: num_iterations + 10 (buffer).
        # Then iterate manually.

        iter_loader = iter(dataloader)

        # Warmup
        d, t = next(iter_loader) # d, t are on device if we didn't move dataset to device?
        # BenchmarkDataset has data on device already.

        optimizer.zero_grad()
        output = model(d)
        loss = criterion(output.view(-1, config.vocab_size), t.view(-1))
        loss.backward()
        optimizer.step()

        if device.type == 'cuda':
            torch.cuda.synchronize()

        start_time = time.time()
        # Run exactly num_iterations
        for _ in range(num_iterations):
            try:
                d, t = next(iter_loader)
            except StopIteration:
                break

            optimizer.zero_grad()
            output = model(d)
            loss = criterion(output.view(-1, config.vocab_size), t.view(-1))
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
        "mode": mode,
        "batch_size": batch_size,
        "avg_time": avg_time,
        "throughput": throughput
    }

def main():
    parser = argparse.ArgumentParser(description='Benchmark Transformer gradients (PyTorch/Opacus).')
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
        res = benchmark(args.mode, config, bs)
        results.append(res)

    print("RESULTS_JSON=" + json.dumps(results))

if __name__ == "__main__":
    main()
