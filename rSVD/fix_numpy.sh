#!/bin/bash
# Quick fix for NumPy 2.x compatibility issue with PyTorch

echo "========================================="
echo "Fixing NumPy version compatibility..."
echo "========================================="

echo "Current NumPy version:"
python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "NumPy not installed"

echo ""
echo "Downgrading NumPy to < 2.0.0 for PyTorch compatibility..."
pip install --force-reinstall "numpy>=1.24.0,<2.0.0"

echo ""
echo "Verifying NumPy version:"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"

echo ""
echo "Testing PyTorch import:"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')" && echo "✓ PyTorch import successful!"

echo ""
echo "========================================="
echo "NumPy fix complete!"
echo "========================================="

