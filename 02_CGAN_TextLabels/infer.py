import os
import argparse
import logging
import yaml
import torch
from torchvision.utils import save_image

from src.generator import CGANGenerator
from src.dataset import SHAPE_TO_LABEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for Conditional GAN Shape Generator")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--label", type=str, default="star", choices=list(SHAPE_TO_LABEL.keys()), help="Shape label to generate")
    parser.add_argument("--checkpoint", type=str, default="models/generator.pth", help="Path to generator weights")
    parser.add_argument("--output_path", type=str, default="outputs/generated_shape.png", help="Path to save generated image")
    return parser.parse_args()

def main():
    args = parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load generator
    generator = CGANGenerator(
        latent_dim=config["model"]["latent_dim"],
        num_classes=config["model"]["num_classes"],
        embedding_dim=config["model"]["embedding_dim"],
        img_shape=tuple(config["model"]["img_shape"])
    ).to(device)

    if os.path.exists(args.checkpoint):
        logger.info(f"Loading weights from {args.checkpoint}...")
        generator.load_state_dict(torch.load(args.checkpoint, map_location=device))
    else:
        logger.warning(f"Weights not found at {args.checkpoint}. Running with random initialization.")

    generator.eval()

    # Generate
    z = torch.randn(1, config["model"]["latent_dim"], device=device)
    label_idx = SHAPE_TO_LABEL[args.label]
    label_tensor = torch.tensor([label_idx], dtype=torch.long, device=device)

    with torch.no_grad():
        generated_img = generator(z, label_tensor)

    # Save
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    save_image(generated_img, args.output_path, normalize=True)
    logger.info(f"Generated shape '{args.label}' successfully saved to: {args.output_path}")

if __name__ == "__main__":
    main()
