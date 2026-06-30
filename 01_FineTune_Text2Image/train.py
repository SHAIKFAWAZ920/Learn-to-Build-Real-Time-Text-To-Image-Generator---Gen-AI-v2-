import os
import argparse
import logging
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

# HuggingFace Diffusers and Transformers
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from diffusers.loaders import LoraLoaderMixin
from diffusers.models.attention_processor import LoRAAttnProcessor

from src.dataset import DreamBoothDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Stable Diffusion with LoRA")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Set up output directories
    output_dir = config["training"]["output_dir"]
    logging_dir = config["training"]["logging_dir"]
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logging_dir, exist_ok=True)

    # Device & GPU Detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")

    # Set seed
    torch.manual_seed(config["training"]["seed"])

    # Load components
    model_path = config["model"]["pretrained_model_name_or_path"]
    logger.info(f"Loading pretrained models from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet").to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

    # Freeze VAE and Text Encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # Inject LoRA into UNet using PEFT
    logger.info("Injecting LoRA parameters into UNet attention layers using PEFT...")
    from peft import LoraConfig
    unet_lora_config = LoraConfig(
        r=config["training"]["lora_r"],
        lora_alpha=config["training"]["lora_alpha"],
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)
    
    # Extract LoRA parameters to optimize
    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        lora_layers,
        lr=float(config["training"]["learning_rate"]),
    )

    # Dataset & Dataloader
    dataset = DreamBoothDataset(
        instance_data_dir=config["dataset"]["instance_data_dir"],
        instance_prompt=config["dataset"]["instance_prompt"],
        tokenizer=tokenizer,
        size=config["dataset"]["resolution"],
    )
    
    train_dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["train_batch_size"],
        shuffle=True,
    )

    # TensorBoard
    writer = SummaryWriter(log_dir=logging_dir)

    # Mixed precision setup
    mp = config["training"]["mixed_precision"]
    scaler = torch.cuda.amp.GradScaler(enabled=(mp == "fp16"))

    # Training state
    global_step = 0
    max_train_steps = config["training"]["max_train_steps"]
    
    # Check for resuming training
    resume_path = config["training"]["resume_from_checkpoint"]
    if resume_path:
        if resume_path == "latest":
            # find latest checkpoint in output_dir
            ckpt_dirs = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
            if ckpt_dirs:
                ckpt_dirs.sort(key=lambda x: int(x.split("-")[1]))
                resume_path = os.path.join(output_dir, ckpt_dirs[-1])
            else:
                resume_path = None
        
        if resume_path and os.path.exists(resume_path):
            logger.info(f"Resuming training from checkpoint: {resume_path}")
            # load weights
            lora_state_dict = torch.load(os.path.join(resume_path, "lora_weights.pt"), map_location="cpu")
            # Set state dict
            # For simplicity in native diffusers, we restore by calling load_attn_procs on unet
            unet.load_attn_procs(resume_path, weight_name="pytorch_lora_weights.bin")
            global_step = int(os.path.basename(resume_path).split("-")[1])

    progress_bar = tqdm(range(global_step, max_train_steps), desc="Steps")
    
    unet.train()
    
    while global_step < max_train_steps:
        for step, batch in enumerate(train_dataloader):
            if global_step >= max_train_steps:
                break
                
            # Convert images to latent space
            latents = vae.encode(batch["instance_images"].to(device)).latent_dist.sample()
            latents = latents * 0.18215

            # Sample noise
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(batch["instance_prompt_ids"].to(device))[0]

            # Predict the noise residual
            with torch.cuda.amp.autocast(enabled=(mp == "fp16")):
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

            # Backpropagation
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            progress_bar.update(1)
            
            # TensorBoard logging
            writer.add_scalar("loss", loss.item(), global_step)
            
            if global_step % 10 == 0 or global_step == max_train_steps:
                logger.info(f"Step {global_step}/{max_train_steps} - Loss: {loss.item():.4f}")

            # Checkpointing
            if global_step % config["training"]["checkpointing_steps"] == 0 or global_step == max_train_steps:
                ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                # Save UNet LoRA weights
                # Native diffusers saving
                unet.save_attn_procs(ckpt_dir)
                logger.info(f"Saved checkpoint to {ckpt_dir}")

    writer.close()
    logger.info("Fine-tuning completed successfully!")

if __name__ == "__main__":
    main()
