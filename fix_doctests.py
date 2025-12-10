import sys

files = ['jax_privacy/batch_selection.py', 'jax_privacy/clipping.py', 'jax_privacy/noise_addition.py']

for filepath in files:
    with open(filepath, 'r') as f:
        content = f.read()

    lines = content.splitlines()
    new_lines = []
    for i, line in enumerate(lines):
        # Look for closing code fence
        # Check if previous line is non-empty and not an opening fence
        stripped = line.strip()
        if stripped == '```' and i > 0:
            prev_line = lines[i-1]
            prev_stripped = prev_line.strip()
            if prev_stripped and not prev_stripped.startswith('```'):
                # Insert blank line with same indentation as the closing fence
                indent = line[:line.find('`')]
                new_lines.append('')
        new_lines.append(line)

    with open(filepath, 'w') as f:
        f.write('\n'.join(new_lines) + '\n')
print("Finished fixing doctests.")
