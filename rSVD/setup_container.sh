#!/bin/bash
# Setup script for running bert_imdb_finetune.py in a container without Python
# This script installs Python, pip, dependencies, and sets up the environment

set -e  # Exit on error

echo "========================================="
echo "Setting up Python environment in container"
echo "========================================="

# Detect OS
if [ -f /etc/debian_version ]; then
    OS="debian"
elif [ -f /etc/alpine-release ]; then
    OS="alpine"
elif [ -f /etc/redhat-release ]; then
    OS="redhat"
else
    OS="unknown"
fi

echo "Detected OS: $OS"

# ============================================
# Install Python and pip
# ============================================

if [ "$OS" = "debian" ]; then
    echo "Installing Python and dependencies (Debian/Ubuntu)..."
    apt-get update
    apt-get install -y \
        python3 \
        python3-pip \
        python3-dev \
        build-essential \
        git \
        wget \
        curl
    
    # Make python3 the default python
    ln -sf /usr/bin/python3 /usr/bin/python || true
    
elif [ "$OS" = "alpine" ]; then
    echo "Installing Python and dependencies (Alpine)..."
    apk add --no-cache \
        python3 \
        py3-pip \
        python3-dev \
        gcc \
        g++ \
        make \
        linux-headers \
        git \
        wget \
        curl
    
    # Make python3 the default python
    ln -sf /usr/bin/python3 /usr/bin/python || true
    
elif [ "$OS" = "redhat" ]; then
    echo "Installing Python and dependencies (RedHat/CentOS)..."
    yum install -y \
        python3 \
        python3-pip \
        python3-devel \
        gcc \
        gcc-c++ \
        make \
        git \
        wget \
        curl
    
    ln -sf /usr/bin/python3 /usr/bin/python || true
else
    echo "Warning: Unknown OS. Attempting generic Python installation..."
    echo "Please install Python 3.8+ and pip manually"
fi

# Verify Python installation
echo "========================================="
echo "Verifying Python installation..."
python --version || python3 --version
pip --version || pip3 --version

# ============================================
# Upgrade pip and install build tools
# ============================================
echo "========================================="
echo "Upgrading pip and installing build tools..."
pip install --upgrade pip setuptools wheel

# ============================================
# Install PyTorch (CPU or CUDA)
# ============================================
echo "========================================="
echo "Installing PyTorch..."

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected. Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "No CUDA detected. Installing PyTorch (CPU)..."
    pip install torch torchvision torchaudio
fi

# ============================================
# Install Python dependencies
# ============================================
echo "========================================="
echo "Installing Python dependencies from requirements.txt..."

# Navigate to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Fix NumPy version compatibility (NumPy 2.x incompatible with PyTorch)
echo "Fixing NumPy version compatibility..."
pip install "numpy>=1.24.0,<2.0.0"

# Install requirements
pip install -r requirements.txt

# ============================================
# Verify installation
# ============================================
echo "========================================="
echo "Verifying installations..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import datasets; print(f'Datasets version: {datasets.__version__}')"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"

# ============================================
# Setup complete
# ============================================
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "To run the script:"
echo "  cd $SCRIPT_DIR"
echo "  python bert_imdb_finetune.py"
echo ""
echo "Or set PYTHONPATH to include the rSVD directory:"
echo "  export PYTHONPATH=\$PYTHONPATH:$SCRIPT_DIR"
echo "  python bert_imdb_finetune.py"

