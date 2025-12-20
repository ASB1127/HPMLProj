# Base image for CUDA libraries
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS cuda

# Base image with PyTorch and common deep learning tools
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# Copy CUDA libraries from the cuda stage to the final stage
COPY --from=cuda /usr/local/cuda/lib64 /usr/local/cuda/lib64
COPY --from=cuda /usr/local/cuda/compat /usr/local/cuda/compat

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    tree && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python packages required for the project
RUN pip install --upgrade pip
RUN pip install --no-cache-dir \
    transformers \
    peft \
    evaluate \
    scikit-learn \
    hf_transfer \
    huggingface_hub

