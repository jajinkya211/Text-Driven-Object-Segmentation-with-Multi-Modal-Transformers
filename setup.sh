#!/bin/bash

# Setup script for Referring Expression Segmentation

echo "=========================================="
echo "Setting up Referring Expression Segmentation"
echo "=========================================="

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (modify based on your CUDA version)
echo "Installing PyTorch..."
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU only, use:
# pip install torch torchvision

# Install other dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install refer package
echo "Installing refer package..."
if [ ! -d "refer" ]; then
    git clone https://github.com/lichengunc/refer.git
fi
cd refer
pip install -e .
cd ..

# Create necessary directories
echo "Creating directories..."
mkdir -p checkpoints
mkdir -p logs
mkdir -p results
mkdir -p data

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "Next steps:"
echo "  1. Download RefCOCO data (see README.md)"
echo "  2. Train model: python train.py --data_root /path/to/data"
echo "  3. Evaluate: python evaluate.py --checkpoint checkpoints/best_model.pth --data_root /path/to/data"
echo ""
