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
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run comprehensive benchmarks for JAX/JAX Privacy and PyTorch/Opacus.')
    parser.add_argument('--output_file', type=str, default='comprehensive_results.json', help='Output file for results.')
    parser.add_argument('--models', nargs='+', default=['transformer', 'cnn', 'state_space'], help='Models to benchmark')
    parser.add_argument('--modes', nargs='+', default=['standard', 'clipped'], help='Modes to benchmark')
    parser.add_argument('--frameworks', nargs='+', default=['jax', 'torch'], help='Frameworks to benchmark')
    parser.add_argument('--sizes', nargs='+', default=['small', 'medium', 'large'], help='Model sizes to benchmark')

    args = parser.parse_args()

    # Clear output file if it exists
    if os.path.exists(args.output_file):
        os.remove(args.output_file)
        print(f"Removed existing output file {args.output_file}.")

    # Iterate over the grid
    for framework in args.frameworks:
        for model in args.models:
            for size in args.sizes:
                for mode in args.modes:
                    print(f"\n--- Benchmarking {framework} {model} {size} {mode} ---")

                    batch_size = 1
                    while True:
                        print(f"Testing batch size: {batch_size}")

                        cmd_base = [sys.executable, '-m']

                        if framework == 'jax':
                            cmd_module = ['main']
                            cmd = cmd_base + cmd_module + [
                                '--mode', mode,
                                '--model', model,
                                '--size', size,
                                '--batch_size', str(batch_size),
                                '--output_file', args.output_file,
                            ]

                        elif framework == 'torch':
                            cmd_module = ['main_opacus']
                            cmd = cmd_base + cmd_module + [
                                '--mode', mode,
                                '--model', model,
                                '--size', size,
                                '--batch_size', str(batch_size),
                                '--output_file', args.output_file
                            ]

                        success = run_command(cmd)

                        if success:
                            batch_size *= 2
                        else:
                            print(f"Failed at batch size {batch_size}. Stopping this configuration.")
                            break

                        # Safety break to avoid infinite loops if something weird happens,
                        # though memory should eventually run out.
                        if batch_size > 100000:
                            print("Batch size limit reached.")
                            break

if __name__ == "__main__":
    main()
