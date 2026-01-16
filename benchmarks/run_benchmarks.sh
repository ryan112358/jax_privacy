#!/bin/bash

# Output file
OUTPUT_FILE="results.json"

# Clear previous results if needed, or keep appending.
# The python script appends, so if we want a fresh start we should delete it here.
if [ -f "$OUTPUT_FILE" ]; then
    rm "$OUTPUT_FILE"
fi

# Define configurations
BATCH_SIZES=(16 32)
MICROBATCH_SIZES=(4 8)
SEQ_LENS=(32 64)
# We can fix some model params to keep the runtime reasonable,
# or loop through them as well.
HIDDEN_SIZE=128
NUM_HEADS=4
NUM_LAYERS=2

echo "Starting benchmarks..."

# 1. Standard Mode
for BS in "${BATCH_SIZES[@]}"; do
    for SEQ in "${SEQ_LENS[@]}"; do
        echo "Running: mode=standard, batch_size=$BS, seq_len=$SEQ"
        python -m benchmarks.main \
            --mode standard \
            --batch_size $BS \
            --max_len $SEQ \
            --hidden_size $HIDDEN_SIZE \
            --num_heads $NUM_HEADS \
            --num_layers $NUM_LAYERS \
            --output_file $OUTPUT_FILE
    done
done

# 2. Clipped Mode
for BS in "${BATCH_SIZES[@]}"; do
    for MBS in "${MICROBATCH_SIZES[@]}"; do
        for SEQ in "${SEQ_LENS[@]}"; do
            # Ensure microbatch size is valid (<= batch size)
            if [ $MBS -le $BS ]; then
                echo "Running: mode=clipped, batch_size=$BS, microbatch_size=$MBS, seq_len=$SEQ"
                python -m benchmarks.main \
                    --mode clipped \
                    --batch_size $BS \
                    --microbatch_size $MBS \
                    --max_len $SEQ \
                    --hidden_size $HIDDEN_SIZE \
                    --num_heads $NUM_HEADS \
                    --num_layers $NUM_LAYERS \
                    --output_file $OUTPUT_FILE
            fi
        done
    done
done

echo "All benchmarks finished. Results saved to $OUTPUT_FILE"
