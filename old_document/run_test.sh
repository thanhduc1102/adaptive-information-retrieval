#!/bin/bash
# Quick test script - Run this in WSL

echo "=========================================="
echo "ADAPTIVE IR SYSTEM - QUICK TEST"
echo "=========================================="
echo ""

# Navigate to project
cd "/mnt/d/duc/hust/web mining/project/adaptive-ir-system" || exit 1

# Activate conda
echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate myenv || {
    echo "Error: Could not activate myenv"
    echo "Try: conda activate myenv"
    exit 1
}

echo "✓ Conda environment activated"
echo ""

# Run test
echo "Running comprehensive test..."
echo "=========================================="
python scripts/final_test.py

echo ""
echo "=========================================="
echo "Test complete!"
echo ""
echo "If all tests passed, you can now train:"
echo "  python train.py --config configs/msa_config.yaml"
echo "=========================================="
