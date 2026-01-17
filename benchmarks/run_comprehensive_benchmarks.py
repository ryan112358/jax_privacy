import argparse
import subprocess
import itertools
import json
import os
import sys

def run_command(cmd):
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        # Continue or exit? Let's continue to gather other results.
        pass

def main():
    parser = argparse.ArgumentParser(description='Run comprehensive benchmarks for JAX/JAX Privacy and PyTorch/Opacus.')
    parser.add_argument('--output_file', type=str, default='comprehensive_results.json', help='Output file for results.')
    parser.add_argument('--models', nargs='+', default=['transformer', 'cnn'], help='Models to benchmark (transformer, cnn, state_space)')
    parser.add_argument('--modes', nargs='+', default=['standard', 'clipped'], help='Modes to benchmark (standard, clipped)')
    parser.add_argument('--frameworks', nargs='+', default=['jax', 'torch'], help='Frameworks to benchmark (jax, torch)')
    parser.add_argument('--batch_sizes', nargs='+', type=int, default=[16, 32, 64], help='Batch sizes')
    parser.add_argument('--microbatch_sizes', nargs='+', type=int, default=[4, 8], help='Microbatch sizes (only for JAX clipped)')
    parser.add_argument('--sizes', nargs='+', default=['small'], help='Model sizes to benchmark (small, medium, large)')

    args = parser.parse_args()

    # Clear output file if it exists? Or just append? The underlying scripts append.
    # But maybe we want a fresh start for a comprehensive run.
    if os.path.exists(args.output_file):
        print(f"Output file {args.output_file} exists. Appending to it.")

    # Iterate over the grid
    for framework in args.frameworks:
        for model in args.models:
            for size in args.sizes:
                for mode in args.modes:
                    for batch_size in args.batch_sizes:

                        # Common arguments
                        cmd_base = [sys.executable, '-m']

                        if framework == 'jax':
                            cmd_module = ['benchmarks.main']

                            # JAX specific loop for microbatching in clipped mode
                            current_microbatch_sizes = [None]
                            if mode == 'clipped':
                                current_microbatch_sizes = [mbs for mbs in args.microbatch_sizes if mbs <= batch_size]
                                if not current_microbatch_sizes:
                                    print(f"Skipping {framework} {mode} {batch_size}: No valid microbatch size.")
                                    continue

                            for mbs in current_microbatch_sizes:
                                cmd = cmd_base + cmd_module + [
                                    '--mode', mode,
                                    '--model', model,
                                    '--size', size,
                                    '--batch_size', str(batch_size),
                                    '--output_file', args.output_file,
                                ]
                                if mbs:
                                    cmd += ['--microbatch_size', str(mbs)]
                                run_command(cmd)

                        elif framework == 'torch':
                            cmd_module = ['benchmarks.main_opacus']

                            # Torch/Opacus doesn't support explicit microbatching in the same way via CLI args in main_opacus.py yet.
                            # It handles batching internally or via DataLoader.
                            # So we skip the microbatch loop for Torch.

                            cmd = cmd_base + cmd_module + [
                                '--mode', mode,
                                '--model', model,
                                '--size', size,
                                '--batch_size', str(batch_size),
                                '--output_file', args.output_file
                            ]
                            run_command(cmd)

if __name__ == "__main__":
    main()
