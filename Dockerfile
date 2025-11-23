# Dockerfile for Referring Expression Segmentation

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set python3 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip
RUN pip install --upgrade pip

# Set working directory
WORKDIR /workspace

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install refer package
RUN git clone https://github.com/lichengunc/refer.git && \
    cd refer && \
    pip install -e .

# Copy project files
COPY . .

# Create directories
RUN mkdir -p checkpoints logs results data

# Set default command
CMD ["/bin/bash"]

# Usage:
# Build: docker build -t referring-segmentation .
# Run: docker run --gpus all -it -v /path/to/data:/workspace/data referring-segmentation
# Train: docker run --gpus all -v /path/to/data:/workspace/data referring-segmentation \
#        python train.py --data_root /workspace/data --dataset refcoco
