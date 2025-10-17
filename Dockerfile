# Start from an official NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.1.1-cudatoolkit-ubuntu22.04

# Set environment variables to prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system dependencies like Python, pip, and git
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy and install Python requirements first to leverage Docker's layer caching
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy all your project scripts into the working directory
COPY . .

# Make the bash scripts executable
RUN chmod +x ./*.sh

# --- Pre-download a default model during the build process ---
# This makes the image ready-to-use without needing downloads at runtime.
# You can change "progen2-small" to a different default if you prefer.
RUN ./setup_progen_model.sh progen2-small

# Define volumes for mounting data and saving outputs at runtime
VOLUME ["/app/data", "/app/outputs"]

# Set a default command to show usage instructions
CMD ["echo", "ProGen2 Finetuner container. Run with 'finetune' or 'inference'."]
