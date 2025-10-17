# Scripts Documentation

This directory contains three main scripts with clear, separate responsibilities:

## 📋 Script Overview

### 1. `entrypoint.sh` - Container Initialization
**Purpose**: Starts the SSH server, MLflow, and keeps the container running  
**When**: Runs automatically when the container starts (via Dockerfile ENTRYPOINT)  
**What it does**:
- Configures and starts SSH server
- Installs extra pip packages (if `EXTRA_PIP` env is set)
- **Always starts MLflow and downloads model checkpoints** (via `start_services.sh`)
- Keeps container alive

**You don't need to run this manually** - it runs automatically when the container starts.

---

### 2. `start_services.sh` - Service Management
**Purpose**: Starts MLflow server and downloads model checkpoints  
**When**: Called automatically by `entrypoint.sh` - you typically don't need to run this manually  
**What it does**:
- Downloads GroundingDINO and SAM model checkpoints (if not present)
- Starts MLflow tracking server on port 5000
- Keeps MLflow running in the background

**Manual Usage** (only if you need to restart MLflow):
```bash
# Inside the container
/app/scripts/start_services.sh
```

**Environment variables**:
- `MLFLOW_HOST` (default: 0.0.0.0)
- `MLFLOW_PORT` (default: 5000)
- `MLFLOW_BACKEND_STORE` (default: /app/mlflow/mlruns)
- `MLFLOW_ARTIFACT_ROOT` (default: /app/mlflow/artifacts)
- `SKIP_CHECKPOINT_DOWNLOAD` (set to 1 to skip downloads)

---

### 3. `train_lora.sh` - Training Execution
**Purpose**: Runs the LoRA training with accelerate  
**When**: Run manually when you want to start training  
**What it does**:
- Configures training parameters
- Checks if MLflow is available (warns if not)
- Launches training using `accelerate launch`

**Usage**:
```bash
# Inside the container, with default parameters
/app/scripts/train_lora.sh

# Or with custom environment variables
MODEL_ID="your/model" \
COND_DIR="/path/to/cond" \
TARGET_DIR="/path/to/target" \
/app/scripts/train_lora.sh

# Or pass additional arguments
/app/scripts/train_lora.sh --num_train_epochs=50
```

**Environment variables**:
- `MODEL_ID` (default: stabilityai/stable-diffusion-xl-1.0-inpainting-0.1)
- `COND_DIR` (required: path to conditioning images)
- `TARGET_DIR` (required: path to target images)
- `CAPTION_FILE` (optional: path to captions JSON)
- `OUT_DIR` (default: sdxl-stage1-lora)
- `MLFLOW_TRACKING_URI` (default: http://localhost:5000)
- `MLFLOW_EXPERIMENT_NAME` (default: maskfree_vton)

---

## 🚀 Typical Workflow

### Scenario 1: Standard workflow (most common)
```bash
# Run container (SSH + MLflow start automatically)
docker run -p 22:22 -p 5000:5000 aminamzl421/maskfree-vton:latest

# SSH into container
ssh trainer@<container-ip>

# Start training (MLflow is already running)
cd /app
export COND_DIR="/path/to/cond"
export TARGET_DIR="/path/to/target"
./scripts/train_lora.sh
```

### Scenario 2: Development/debugging
```bash
# Run container interactively
docker run -it -p 22:22 -p 5000:5000 aminamzl421/maskfree-vton:latest bash

# Inside container - services are already starting in background
# Wait a moment for MLflow to be ready
sleep 5

# Run training script directly with accelerate
cd /app
accelerate launch train_lora.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-1.0-inpainting-0.1" \
  --cond_data_dir="/path/to/cond" \
  --target_data_dir="/path/to/target" \
  --output_dir="output"
```

---

## 📝 Notes

- **SSH Server**: ✅ Always starts automatically via `entrypoint.sh`
- **MLflow**: ✅ Always starts automatically via `entrypoint.sh` → `start_services.sh`
- **Model Checkpoints**: ✅ Always downloaded automatically (if not present)
- **Training**: Manual start only - gives you full control when to begin
- **Accelerate**: Training script always uses `accelerate launch` for optimal performance
