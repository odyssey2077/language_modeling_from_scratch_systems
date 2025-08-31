#!/bin/bash

# Helper script to run all memory profiling configurations for the assignment

echo "=========================================="
echo "Running Memory Profiling for 2.7B Model"
echo "=========================================="
echo ""

# Part (a) and (b): Profile forward pass and full training step for different context lengths
echo "Part (a)(b): Profiling forward pass and training step..."
echo "------------------------------------------"

# Forward pass only
echo "1. Forward pass only (no memory snapshots for table):"
python cs336_systems/scripts/memory_profile.py \
    --mode forward \
    --context-lengths 128 256 512

echo ""
echo "2. Forward pass with memory snapshot (for visualization):"
python cs336_systems/scripts/memory_profile.py \
    --mode forward \
    --context-lengths 256 \
    --profile-memory \
    --snapshot-prefix forward_pass

echo ""
echo "3. Full training step (no memory snapshots for table):"
python cs336_systems/scripts/memory_profile.py \
    --mode training \
    --context-lengths 128 256 512

echo ""
echo "4. Full training step with memory snapshot (for visualization):"
python cs336_systems/scripts/memory_profile.py \
    --mode training \
    --context-lengths 256 \
    --profile-memory \
    --snapshot-prefix training_step

# Part (c): Profile with mixed precision
echo ""
echo "Part (c): Profiling with mixed precision..."
echo "------------------------------------------"

echo "5. Forward pass with AMP:"
python cs336_systems/scripts/memory_profile.py \
    --mode forward \
    --context-lengths 128 256 512 \
    --use-amp

echo ""
echo "6. Full training step with AMP:"
python cs336_systems/scripts/memory_profile.py \
    --mode training \
    --context-lengths 128 256 512 \
    --use-amp

echo ""
echo "=========================================="
echo "Memory Profiling Complete!"
echo "=========================================="
echo ""
echo "Generated files:"
echo "  - memory_profile_results_forward.csv"
echo "  - memory_profile_results_training.csv"
echo "  - forward_pass_ctx256.pickle (for memory_viz)"
echo "  - training_step_ctx256.pickle (for memory_viz)"
echo ""
echo "To view memory snapshots:"
echo "  1. Open https://pytorch.org/memory_viz"
echo "  2. Drag and drop the .pickle files"
echo ""
echo "Part (d) Answer:"
echo "The size of a tensor of activations in the Transformer residual stream"
echo "for the 2.7B model is calculated as:"
echo "  - Shape: (batch_size=4, context_length, d_model=2560)"
echo "  - For context_length=128: 4 * 128 * 2560 * 4 bytes = 5.24 MB"
echo "  - For context_length=256: 4 * 256 * 2560 * 4 bytes = 10.49 MB"
echo "  - For context_length=512: 4 * 512 * 2560 * 4 bytes = 20.97 MB"