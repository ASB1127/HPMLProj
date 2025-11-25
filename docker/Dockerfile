# Stage 1: get CUDA developer tools from NVIDIA
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS cuda

# Stage 2: use RunPod image for entrypoint + environment
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# Copy CUDA toolkit into RunPod image
COPY --from=cuda /usr/local/cuda /usr/local/cuda

# Expose CUDA binaries and libraries inside container
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Optional developer tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends vim htop git && \
    rm -rf /var/lib/apt/lists/*

# Keep RunPod entrypoint so the container stays alive on launch
# (runpod/pytorch sets ENTRYPOINT ["/start.sh"])

