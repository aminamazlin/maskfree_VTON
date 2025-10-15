# Use the specified base image
FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04

# Avoid interactive prompts during apt operations
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies (git for pulling models, libglib2.0-0 for opencv, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
	git wget curl ca-certificates libglib2.0-0 libsm6 libxext6 libxrender1 \
	&& rm -rf /var/lib/apt/lists/*

# Create workspace directory
WORKDIR /workspace

# Copy requirement files into /workspace
# It’s copied first because Docker can cache it — meaning if only your code changes (not the requirements), it won’t reinstall everything.
COPY requirements.txt ./

# Install python dependencies
# Use --ignore-installed to avoid errors trying to uninstall distutils packages (e.g., blinker in base image)
# and install in one layer to maximize cache efficiency. You can pin versions in requirements.txt for reproducibility.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --ignore-installed -r requirements.txt

# Copy the rest of the source code
COPY . .

# Environment vars for MLflow (override at run time if needed)
ENV MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
	MLFLOW_EXPERIMENT_NAME=maskfree_vton \
	PYTHONUNBUFFERED=1

# Expose MLflow default port (if running server inside container) and potential gradio / app port
EXPOSE 5000 7860

# Create an entrypoint script directory (script added separately)
RUN mkdir -p /workspace/scripts

# Default command prints help; override with entrypoint script
CMD ["python", "train_lora.py", "--help"]