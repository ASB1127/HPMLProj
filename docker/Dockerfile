FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# -----------------------------
# System deps
# -----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git curl ca-certificates openssh-server \
    && rm -rf /var/lib/apt/lists/*

# SSH setup (optional)
RUN mkdir -p /var/run/sshd

EXPOSE 22

# -----------------------------
# Python deps
# -----------------------------
RUN pip3 install --no-cache-dir --upgrade pip

# Install PyTorch CUDA wheels (fast, official)
RUN pip3 install --no-cache-dir \
    torch==2.2.2+cu121 \
    torchvision==0.17.2+cu121 \
    torchaudio==2.2.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# HF + required libs
RUN pip3 install --no-cache-dir \
    transformers>=4.30.0 \
    datasets>=2.12.0 \
    evaluate \
    peft>=0.5.0 \
    numpy>=1.24.0 \
    tqdm>=4.65.0 \
    accelerate \
    huggingface_hub

# -----------------------------
# Copy project
# -----------------------------
WORKDIR /workspace
COPY . /workspace

# -----------------------------
# SSHD entrypoint
# -----------------------------
CMD ["/usr/sbin/sshd", "-D"]

