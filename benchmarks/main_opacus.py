import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
import argparse
import json
from benchmarks.transformer_models import TransformerTorch as Transformer, TransformerConfig, generate_dummy_data_torch as generate_dummy_data
from benchmarks.cnn_models import CNNTorch as CNN, CNNConfig, generate_dummy_data_torch as generate_cnn_data
from benchmarks.diffusion import TorchDiffusion, DiffusionConfig, generate_dummy_data_torch
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

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
        model = Transformer(config).to(device)
        # Generate data
        d = generate_dummy_data(batch_size, config.max_len, config.vocab_size, seed=42).to(device)
        t = generate_dummy_data(batch_size, config.max_len, config.vocab_size, seed=43).to(device)
        data_batch, targets_batch = d, t

        def loss_fn(output, targets):
             return nn.CrossEntropyLoss()(output.view(-1, config.vocab_size), targets.view(-1))

    elif model_name == 'CNN':
        model = CNN(config).to(device)
        d, t = generate_cnn_data(batch_size, config.input_shape, config.num_classes, seed=42)
        data_batch, targets_batch = d.to(device), t.to(device)

        def loss_fn(output, targets):
             return nn.CrossEntropyLoss()(output, targets)

    elif model_name == 'Diffusion':
        model = TorchDiffusion(config).to(device)
        d, t = generate_dummy_data_torch(batch_size, config, seed=42)
        # d is (x, t_emb), t is noise
        x_in, t_in = d
        data_batch = (x_in.to(device), t_in.to(device))
        targets_batch = t.to(device)

        def loss_fn(output, targets):
             return nn.MSELoss()(output, targets)
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
    parser.add_argument('--model', type=str, default='Transformer', choices=['Transformer', 'CNN', 'Diffusion'],
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

    config = None
    if args.model == 'Transformer':
        config = TransformerConfig(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            max_len=args.max_len,
            dropout_rate=0.0
        )
    elif args.model == 'CNN':
        if args.size == 'small':
            config = CNNConfig.small()
        elif args.size == 'medium':
            config = CNNConfig.medium()
        elif args.size == 'large':
            config = CNNConfig.large()
    elif args.model == 'Diffusion':
        if args.size == 'small':
            config = DiffusionConfig.small()
        elif args.size == 'medium':
            config = DiffusionConfig.medium()
        elif args.size == 'large':
            config = DiffusionConfig.large()

    if config is None:
        raise ValueError(f"Invalid model or size configuration: {args.model}, {args.size}")

    if args.batch_size:
        batch_sizes = [args.batch_size]
    else:
        batch_sizes = [16, 32, 64]

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
