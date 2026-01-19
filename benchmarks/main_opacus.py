import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
import argparse
import json
from transformer import TransformerConfig
from cnn import CNNConfig
from state_space import StateSpaceConfig
from diffusion import DiffusionConfig
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

    model = config.make().to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")

    dummy_data = config.generate_dummy_data(batch_size)

    def to_device(data):
        if isinstance(data, np.ndarray):
            # If floating point, use default float (float32), else long for integers
            if np.issubdtype(data.dtype, np.floating):
                 return torch.from_numpy(data).float().to(device)
            else:
                 return torch.from_numpy(data).long().to(device)
        elif isinstance(data, (tuple, list)):
            return type(data)(to_device(x) for x in data)
        return data

    data_batch, targets_batch = to_device(dummy_data)

    if isinstance(config, TransformerConfig):
        def loss_fn(output, targets):
             return nn.CrossEntropyLoss()(output.view(-1, config.vocab_size), targets.view(-1).long())
    elif isinstance(config, CNNConfig):
         def loss_fn(output, targets):
             return nn.CrossEntropyLoss()(output, targets)
    elif isinstance(config, StateSpaceConfig):
         def loss_fn(output, targets):
             return nn.CrossEntropyLoss()(output.view(-1, config.vocab_size), targets.view(-1))
    elif isinstance(config, DiffusionConfig):
         def loss_fn(output, targets):
             return nn.MSELoss()(output, targets)
    else:
         raise ValueError(f"Unknown config type: {type(config)}")

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
                # For diffusion (x, t, target), we want ((x, t), target)
                if isinstance(config, DiffusionConfig):
                    # x is batch[0], t is batch[1], target is batch[2]
                     return (batch[0], batch[1]), batch[2]
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
        "batch_size": batch_size,
        "avg_time": avg_time,
        "throughput": throughput,
        "model": config.__class__.__name__.replace("Config", ""),
        "mode": mode,
        "num_params": num_params,
    }

def main():
    parser = argparse.ArgumentParser(description='Benchmark gradients (PyTorch/Opacus).')
    parser.add_argument('--mode', type=str, required=True, choices=['standard', 'clipped'],
                        help='Benchmark mode: standard or clipped')
    parser.add_argument('--model', type=str, default='transformer', choices=['transformer', 'cnn', 'state_space', 'diffusion'],
                        help='Model to benchmark')
    parser.add_argument('--size', type=str, default='small', choices=['small', 'medium', 'large'],
                        help='Model size')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size')
    parser.add_argument('--output_file', type=str, default='results.json', help='Output JSON file to append results')

    args = parser.parse_args()

    if args.model == 'transformer':
        config = TransformerConfig.build(args.size)
    elif args.model == 'cnn':
        config = CNNConfig.build(args.size)
    elif args.model == 'state_space':
        config = StateSpaceConfig.build(args.size)
    elif args.model == 'diffusion':
        config = DiffusionConfig.build(args.size)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    config.framework = 'torch'

    res = run_benchmark(args.mode, args.model, config, args.batch_size)
    res['framework'] = 'torch'
    res['size'] = args.size

    with open(args.output_file, 'a') as f:
        f.write(json.dumps(res) + '\n')
    print(f"Result appended to {args.output_file}")

if __name__ == "__main__":
    main()
