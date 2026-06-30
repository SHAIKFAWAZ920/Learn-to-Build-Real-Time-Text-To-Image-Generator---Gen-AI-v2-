import os
import argparse
import logging
import yaml
import torch
from diffusers import StableDiffusionPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for LoRA Fine-tuned Stable Diffusion")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--prompt", type=str, default=None, help="Inference prompt (defaults to config instance_prompt)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint folder containing LoRA weights")
    parser.add_argument("--output_path", type=str, default="outputs/generated_sample.png", help="Path to save generated image")
    return parser.parse_args()

def main():
    args = parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model_path = config["model"]["pretrained_model_name_or_path"]
    logger.info(f"Loading base Stable Diffusion model: {model_path}")
    
    # Load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float32,  # use fp32 for CPU compatibility
    )
    
    # Load LoRA weights if provided
    if args.checkpoint:
        logger.info(f"Loading LoRA weights from checkpoint: {args.checkpoint}")
        pipe.unet.load_attn_procs(args.checkpoint)
    
    pipe.to(device)

    # Set up prompt
    prompt = args.prompt if args.prompt else config["dataset"]["instance_prompt"]
    logger.info(f"Generating image for prompt: '{prompt}'")

    # Run inference
    # Note: Using small steps/size for compatibility with tiny/mock models
    image = pipe(
        prompt,
        num_inference_steps=5 if "tiny" in model_path else 30,
        guidance_scale=7.5,
    ).images[0]

    # Save output
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    image.save(args.output_path)
    logger.info(f"Generated image saved successfully to: {args.output_path}")

if __name__ == "__main__":
    main()
