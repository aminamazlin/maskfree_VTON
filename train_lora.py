import argparse
import math
import os
from pathlib import Path
import json
import mlflow
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer ,CLIPTextModelWithProjection

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from peft import LoraConfig, get_peft_model


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.13.0.dev0")

logger = get_logger(__name__)


def tokenize_prompt(tokenizer, prompts):
    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


@torch.no_grad()
def encode_prompts(text_encoder, text_encoder_2, tokenizer, tokenizer_2, prompts: list[str] , device, dtype):
    """
    Returns:
      prompt_embeds: (1, seq, C1+C2)  -> concat of last_hidden_state from both encoders
      pooled_prompt_embeds: (1, Cpool) -> pooled/projection from text_encoder_2 (SDXL uses this)
    """
    # Tokenize
    input_ids1 = tokenize_prompt(tokenizer, prompts).to(device)
    input_ids_2 = tokenize_prompt(tokenizer_2, prompts).to(device)

    # Encode: CLIPTextModel
    out1 = text_encoder(input_ids1, output_hidden_states=False, return_dict=True)
    hidden1 = out1.last_hidden_state.to(dtype)

    # Encode: CLIPTextModelWithProjection
    out2 = text_encoder_2(input_ids_2, output_hidden_states=False, return_dict=True)
    hidden2 = out2.last_hidden_state.to(dtype)

    # pooled / projected text embedding used by SDXL as additional conditioning
    pooled = out2.text_embeds.to(dtype) if hasattr(out2, "text_embeds") else out2.pooler_output.to(dtype)

    # Align sequence lengths if different
    if hidden1.shape[1] != hidden2.shape[1]:
        # simple pad or crop to the min length
        L = min(hidden1.shape[1], hidden2.shape[1])
        hidden1 = hidden1[:, :L]
        hidden2 = hidden2[:, :L]

    prompt_embeds = torch.cat([hidden1, hidden2], dim=-1)
    return prompt_embeds, pooled


def compute_time_ids(height, width, crop=(0,0), target_size=None, device="cpu", dtype=torch.float32):
    if target_size is None:
        target_size = (height, width)

    # order: orig_h, orig_w, crop_y, crop_x, target_h, target_w
    return torch.tensor(
        [height, width, crop[0], crop[1], target_size[0], target_size[1]],
        device=device, dtype=dtype
    )
    

def collate_fn(examples):
    targets = torch.stack([ex["target_tensor"] for ex in examples])   # (B,3,H,W) [-1,1]
    conds   = torch.stack([ex["cond_tensor"]   for ex in examples])   # (B,3,H,W) [-1,1]
    prompts = [ex["prompt"] for ex in examples]

    # Stage-1 mask-free: zero mask
    B, _, H, W = targets.shape
    masks = torch.zeros((B, 1, H, W), dtype=targets.dtype)    
    # same as erased image
    masked = conds.clone()                                

    batch =  {
        "pixel_values": targets,
        "cond_values": conds,
        "masks": masks,
        "masked_images": masked,
        "prompts": prompts,
    }
    return batch

def parse_args():
    parser = argparse.ArgumentParser("SDXL-Inpaint LoRA Stage-1 (mask-free)")
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="stabilityai/stable-diffusion-xl-1.0-inpainting-0.1",
                        help="HF model id or local path")
    parser.add_argument("--cond_data_dir", type=str, required=True,
                        help="Folder with traceless-erased (conditioning) images")
    parser.add_argument("--target_data_dir", type=str, required=True,
                        help="Folder with target images (same filenames as cond)")
    parser.add_argument("--captions_file", type=str, default=None,
                        help="JSONL with {'file': 'xxx.jpg', 'prompt': '...'} per line")
    parser.add_argument("--output_dir", type=str, default="sdxl-inpaint-lora")
    parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument("--resolution", type=int, default=1024)
    # parse_args()
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width",  type=int, default=1024)
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no","fp16","bf16"])
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--scale_lr", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--mlflow_tracking_uri", type=str, default=None)
    parser.add_argument("--mlflow_experiment", type=str, default=None, help="MLFlow experiment name")
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args



class MaskFreeDataset(Dataset):
    """
    Returns:
      - target_tensor:  I_T  (normalized to [-1,1])
      - cond_tensor:    I_c  (normalized to [-1,1])
      - prompt:         short object phrase
    """
    def __init__(self, height: int, width: int, target_dir, cond_dir, captions_json=None, center_crop=False):
        self.target_dir = Path(target_dir)
        self.cond_dir   = Path(cond_dir)
        self.height, self.width = height, width
        self.center_crop = center_crop

        # file list from target_dir
        self.files = sorted([p.name for p in self.target_dir.iterdir()
                             if p.suffix.lower() in {".jpg",".jpeg",".png"}])
        
        self.prompts = {}
        if captions_json and os.path.exists(captions_json):
            with open(captions_json, "r", encoding="utf-8") as file:
                for line in file:
                    j = json.loads(line)
                    self.prompts[j["file"]] = j["prompt"]


        resize = transforms.Resize((self.height, self.width), interpolation=transforms.InterpolationMode.BICUBIC)
        crop   = transforms.CenterCrop((self.height, self.width)) if center_crop else transforms.RandomCrop((self.height, self.width))
        self.preprocess = transforms.Compose([
            #crop,
            # transforms.Resize(
            #     (self.height, self.width), 
            #     interpolation=transforms.InterpolationMode.BICUBIC
            # ),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])  # -> [-1,1]
        ])


    def _open_rgb(self, path: Path):
        img = Image.open(path)
        return img.convert("RGB") if img.mode != 'RGB' else img
    
    # Returns the number of samples
    def __len__(self):
        return len(self.files)
    
    # Returns a sample from the dataset
    def __getitem__(self, index):
        name = self.files[index]
        target_path = self.target_dir / name
        cond_path = self.cond_dir   / name
        if not cond_path.exists():
            raise FileNotFoundError(f"Missing cond image for {name}: {cond_path}")
        
        # load & preprocess
        target_tensor = self.preprocess(self._open_rgb(target_path))
        cond_tensor   = self.preprocess(self._open_rgb(cond_path))

        prompt = self.prompts.get(name, "")

        return {
            "target_tensor": target_tensor,
            "cond_tensor":   cond_tensor,
            "prompt":        prompt,
            "H": target_tensor.shape[-2],
            "W": target_tensor.shape[-1],
        }
    

def train():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit, project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="mlflow",
        project_config=accelerator_project_config,
    )

    if args.seed is not None:
        set_seed(args.seed)

    # Params not to log in mlflow
    ignore_params = {
        "logging_dir", "push_to_hub", "hub_token",
        "hub_model_id", "mlflow_tracking_uri", "mlflow_experiment"
    }

    # Setup MLFlow
    if accelerator.is_main_process:
        if args.mlflow_tracking_uri:
            # Connect to local mlflow server
            mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        # Set experiment name
        mlflow.set_experiment(args.mlflow_experiment)
        mlflow.start_run()

        # log params
        params = {
            k: (str(v) if not isinstance(v, (int, float, bool, str)) else v)
            for k, v in vars(args).items()
            if k not in ignore_params
        }
        mlflow.log_params(params)


    # Handle the repository creation
    if accelerator.is_main_process:     # Only run this process on the main (first) process, not all GPUs
        # create output dir
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        # Push to huggingface if arg given
        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizers
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")

    # Load SDXL components
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2")

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    # Freeze base weights (we'll train only LoRA on the UNet attention) 
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    unet.requires_grad_(False)

    # dtypes
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move modules to GPU
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")


    # insert LoRA in Unet 
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=[
            'proj_in', 'proj_out',
            # self-attention
            'attn1.to_k', 'attn1.to_q', 'attn1.to_v', 'attn1.to_out.0', 
            # cross-attention
            'attn2.to_k', 'attn2.to_q', 'attn2.to_v', 'attn2.to_out.0', 
            # MLP (feed-forward)
            'ff.net.0.proj', 'ff.net.2',
            'norm.linear','norm1.linear','norm2.linear','norm3.linear',
        ]
    )
    unet = get_peft_model(unet, lora_config)   # Injects small trainable matrices into the Unet
    if accelerator.is_main_process:
        unet.print_trainable_parameters()

    # Scale LR
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Optimizer
    optimizer_class = torch.optim.AdamW
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

    # Optimizer only updates trainable params
    optimizer = optimizer_class(
        filter(lambda p: p.requires_grad, unet.parameters()),   # Only keep trainable parameters
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # Dataset & loader
    H, W = args.height, args.width
    train_dataset = MaskFreeDataset(
        cond_dir=args.cond_data_dir,        # traceless erased tryon image
        target_dir=args.target_data_dir,    # original tryon
        captions_json=args.captions_file,   # instruct prompts
        height=H,
        width=W,
        center_crop=args.center_crop,
    )
    # returns samples per step based on batch
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn
    )

    # Compute num training steps
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)   
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Set learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare with accelerator
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    # accelerator.register_for_checkpointing(lr_scheduler)

    # If args.max_train_steps is not given, recompute epochs if needed
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # Compute num of epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Trackers
    if accelerator.is_main_process:
        accelerator.init_trackers("sdxl-inpaint-lora", config=vars(args))

    # Train
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")


    global_step = 0   # Training step across all epochs
    first_epoch = 0
    resume_step = 0   # Tracks training step inside one epoch

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            # If checkpoint to resume from is not the latest, construct the checkpoint path name
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            # Last checkpoint path
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            # Access the checkpoint to resume from
            accelerator.load_state(os.path.join(args.output_dir, path))
            # Access the training step
            global_step = int(path.split("-")[1])

            # Which batch nubmer across whole training (all epochs) we left on
            resume_global_step = global_step * args.gradient_accumulation_steps
            # Which epoch we left on
            first_epoch = global_step // num_update_steps_per_epoch
            # Which batch we left on inside the current epoch 
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)


    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        # Put model training mode
        unet.train()
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            # Handle gradiënt accumulation automatically
            with accelerator.accumulate(unet):
                # Convert images to latent space (target tryon img)
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
               
                # Convert masked images to latent space (traceless erased tryon img)
                masked_images = batch["masked_images"].to(accelerator.device, dtype=weight_dtype)
                masked_latents = vae.encode(masked_images).latent_dist.sample() * vae.config.scaling_factor

                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample timestep
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()

                # Add noise to the latents 
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Downsample mask to latent size (H/8, W/8)
                mask = F.interpolate(
                    batch["masks"].to(device=latents.device, dtype=weight_dtype),
                    size=(latents.shape[-2], latents.shape[-1]),
                    mode="nearest",
                )

                # Concat channels: [noisy_latents, mask, masked_latents]
                latent_model_input = torch.cat([noisy_latents, mask, masked_latents], dim=1)

                # Get the text embedding for conditioning
                enc_hidden, pooled = encode_prompts(
                    text_encoder, text_encoder_2, tokenizer, tokenizer_2,
                    prompts=batch["prompts"], device=accelerator.device, dtype=weight_dtype
                )

                # Compute time ids (training at fixed resolution)
                time_id = compute_time_ids(H, W, crop=(0,0), device=latents.device, dtype=weight_dtype)
                time_ids = time_id.unsqueeze(0).repeat(bsz, 1)

                assert unet.config.in_channels == latent_model_input.shape[1] == 9

                # Predict the noise residual (UNet forward)
                noise_pred = unet(
                    latent_model_input, 
                    timesteps, 
                    encoder_hidden_states=enc_hidden,
                    added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids}
                ).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Compute mean squared error loss
                loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                # Backpropagations
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                # Update weights
                optimizer.step()
                # Update learning rate
                lr_scheduler.step()
                # Reset gradiënts
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Save Checkpoint every args.checkpointing_steps 
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                        # Save intermediate lora 
                        mlflow.log_artifacts(save_path, artifact_path=f"checkpoints/step_{global_step}")

            # Log loss & learning rate
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            # Show logs on tqdm bar
            progress_bar.set_postfix(**logs)
            # Sent logs to mlflow/tensorboard
            accelerator.log(logs, step=global_step)

            # Log metrics in mlflow
            if accelerator.is_main_process:
                mlflow.log_metrics(
                    {"loss": float(logs["loss"]), "lr": lr_scheduler.get_last_lr()[0]},
                    step=global_step
                )

            if global_step >= args.max_train_steps:
                break

        # Sync all processes before next epoch
        accelerator.wait_for_everyone()

    # Save LoRA adapter
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)
        # PEFT: save adapter weights
        unet.save_pretrained(args.output_dir)

        # Save lora
        mlflow.log_artifacts(args.output_dir, artifact_path="final_lora")

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

        mlflow.end_run()
        
    accelerator.end_training()


if __name__ == "__main__":
    train()