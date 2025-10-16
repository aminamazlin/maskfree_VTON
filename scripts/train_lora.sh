#!/usr/bin/env bash
set -euo pipefail

# Training script - runs the LoRA training with accelerate
# Configure these environment variables or set them here:

echo "[train] Starting LoRA training..."

# Training parameters (override via environment variables)
MODEL_ID=${MODEL_ID:-"diffusers/stable-diffusion-xl-1.0-inpainting-0.1"}
COND_DIR=${COND_DIR:-"workspace/app/data/training/new_tryon"}
TARGET_DIR=${TARGET_DIR:-"workspace/app/data/training/orig_tryon"}
OUT_DIR=${OUT_DIR:-"sdxl-stage1-lora"}
CAPTION_FILE=${CAPTION_FILE:-"workspace/app/data/tryon_data.json"}

# MLflow configuration
MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-"http://localhost:5000"}
MLFLOW_EXPERIMENT_NAME=${MLFLOW_EXPERIMENT_NAME:-"maskfree_vton"}

TRAIN_SCRIPT=${TRAIN_SCRIPT:-train_lora.py}

# Ensure we're in the correct directory
cd /workspace/app

# Check if MLflow is reachable (optional warning)
if ! curl -s "$MLFLOW_TRACKING_URI" > /dev/null 2>&1; then
  echo "[train][WARN] MLflow server not reachable at $MLFLOW_TRACKING_URI"
  echo "[train][WARN] Start it with: /app/scripts/start_services.sh"
fi

# Build training command
TRAIN_CMD=(
  --pretrained_model_name_or_path="$MODEL_ID"
  --cond_data_dir="$COND_DIR"
  --target_data_dir="$TARGET_DIR"
  --captions_file="$CAPTION_FILE"
  --output_dir="$OUT_DIR"
  --seed=42
  --height=1024
  --width=1024
  --center_crop
  --train_batch_size=16
  --num_train_epochs=32
  --gradient_accumulation_steps=1
  --gradient_checkpointing
  --learning_rate=1e-4
  --lr_scheduler=constant
  --lr_warmup_steps=0
  --use_8bit_adam
  --adam_beta1=0.9
  --adam_beta2=0.999
  --adam_weight_decay=1e-2
  --adam_epsilon=1e-08
  --max_grad_norm=1.0
  --checkpointing_steps=500
  --enable_xformers_memory_efficient_attention
  --mixed_precision=bf16
  --logging_dir=logs
  --lora_rank=16
  --lora_alpha=16
  --scale_lr
  --local_rank=-1
  --mlflow_tracking_uri="$MLFLOW_TRACKING_URI"
  --mlflow_experiment="$MLFLOW_EXPERIMENT_NAME"
)

# Add any extra arguments passed to this script
TRAIN_CMD+=("$@")

# Run training with accelerate (required for distributed training)
echo "[train] Launching training with accelerate: $TRAIN_SCRIPT"
accelerate launch \
  --mixed_precision=bf16 \
  --num_processes=1 \
  "$TRAIN_SCRIPT" "${TRAIN_CMD[@]}"

echo "[train] Training completed!"
