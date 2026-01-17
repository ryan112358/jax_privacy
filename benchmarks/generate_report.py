import json
import pandas as pd
import platform
import subprocess
import os
import sys
from tabulate import tabulate

def get_system_info():
    info = {
        "Platform": platform.platform(),
        "Processor": platform.processor(),
        "Python Version": platform.python_version(),
    }

    # Try to get GPU info
    try:
        nvidia_smi = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], encoding='utf-8')
        info["GPU"] = nvidia_smi.strip().replace('\n', '; ')
    except (subprocess.CalledProcessError, FileNotFoundError):
        info["GPU"] = "N/A (CPU only or nvidia-smi not found)"

    return info

def format_params(num_params):
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    else:
        return str(num_params)

def main():
    results_file = 'comprehensive_results.json'
    output_file = 'benchmarks/results.md'

    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found.")
        return

    data = []
    with open(results_file, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line}")

    if not data:
        print("No data found in results file.")
        return

    df = pd.DataFrame(data)

    # Create Markdown content
    md_lines = []
    md_lines.append("# Benchmark Results")
    md_lines.append("\n## System Information")

    sys_info = get_system_info()
    for k, v in sys_info.items():
        md_lines.append(f"- **{k}**: {v}")

    md_lines.append("\n## Discussion / Interpretation")
    md_lines.append("\nThe following results demonstrate the performance characteristics of JAX and PyTorch (Opacus) across different models and sizes.")
    md_lines.append("\n**Key Observations:**")
    md_lines.append("1. **Throughput Scaling:** Throughput generally increases with batch size until memory saturation.")
    md_lines.append("2. **Privacy Overhead:** 'Clipped' mode (DP-SGD) incurs significant overhead compared to 'Standard' training. This is due to the per-sample gradient computation required for privacy, preventing certain batch optimizations.")
    md_lines.append("3. **Framework Comparison:** Performance varies between JAX and PyTorch depending on the model and batch size. JAX often shows strong performance on TPUs/GPUs with XLA compilation, while PyTorch/Opacus has its own optimizations.")
    md_lines.append("4. **Memory Limits:** Larger models and batch sizes eventually lead to OOM errors, cutting off the curves.")

    md_lines.append("\n## Detailed Results")

    # Group by Model and Num Params (Size)

    models = sorted(df['model'].unique())
    sizes_order = ['small', 'medium', 'large']

    for model in models:
        for size in sizes_order:
            # Filter for this model and size
            sub_df = df[(df['model'] == model) & (df['size'] == size)]

            if sub_df.empty:
                continue

            num_params = sub_df['num_params'].iloc[0]
            param_str = format_params(num_params)

            md_lines.append(f"\n### Model: {model} | Size: {size} | Params: {param_str} ({num_params:,})")

            # Construct column keys
            sub_df = sub_df.copy() # Avoid SettingWithCopyWarning
            sub_df['key'] = sub_df['framework'] + " (" + sub_df['mode'] + ")"

            # Handle duplicates: take the latest run (last entry) for each batch_size + key
            # This handles cases where the benchmark was restarted or run multiple times
            sub_df = sub_df.drop_duplicates(subset=['batch_size', 'key'], keep='last')

            pivot = sub_df.pivot(index='batch_size', columns='key', values='throughput')

            # Sort by batch size
            pivot.sort_index(inplace=True)

            # Reset index to make batch_size a column
            pivot.reset_index(inplace=True)

            # Rename batch_size column
            pivot.rename(columns={'batch_size': 'Batch Size'}, inplace=True)

            # Fill NaNs with "-"
            pivot.fillna("-", inplace=True)

            # Format numbers
            for col in pivot.columns:
                if col != 'Batch Size':
                    pivot[col] = pivot[col].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)

            # Generate table
            table = tabulate(pivot, headers='keys', tablefmt='github', showindex=False)
            md_lines.append("\n" + table)

    with open(output_file, 'w') as f:
        f.write('\n'.join(md_lines))
    print(f"Report generated at {output_file}")

if __name__ == "__main__":
    main()
