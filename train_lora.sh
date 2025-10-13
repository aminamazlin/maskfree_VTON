export MODEL_ID="stabilityai/stable-diffusion-xl-1.0-inpainting-0.1"

export COND_DIR="path/to/cond_images"
export TARGET_DIR="path/to/target_images"
export OUT_DIR="sdxl-stage1-lora"

accelerate launch train_lora.py \
  --pretrained_model_name_or_path="$MODEL_ID" \
  --cond_data_dir="$COND_DIR" \
  --target_data_dir="$TARGET_DIR" \
  --captions_file=null \
  --output_dir="$OUT_DIR" \
  --seed=42 \
  --height=1024 \
  --width=1024 \
  --center_crop \
  --train_batch_size=16 \
  --num_train_epochs=32 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --use_8bit_adam \
  --adam_beta1=0.9 \
  --adam_beta2=0.999 \
  --adam_weight_decay=1e-2 \
  --adam_epsilon=1e-08 \
  --max_grad_norm=1.0 \
  --checkpointing_steps=500 \
  --enable_xformers_memory_efficient_attention \
  --mixed_precision="bf16" \
  --logging_dir="logs" \
  --push_to_hub \
  --hub_token=null \
  --hub_model_id=null \
  --lora_rank=16 \
  --lora_alpha=16 \
  --scale_lr \
  --local_rank=-1 \
  --mlflow_tracking_uri=null \
  --mlflow_experiment=null