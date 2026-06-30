import os
import argparse
import logging
import yaml
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.utils import save_image

from src.attention_gan import AttentionGenerator
from src.dataset import SHAPE_TO_LABEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Inference & Attention Maps Visualization")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--label", type=str, default="star", choices=list(SHAPE_TO_LABEL.keys()), help="Shape label to generate")
    parser.add_argument("--checkpoint", type=str, default="models/generator.pth", help="Path to generator weights")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output folder")
    return parser.parse_args()

def main():
    args = parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load generator
    generator = AttentionGenerator(
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

    # Generate image and retrieve attention maps
    z = torch.randn(1, config["model"]["latent_dim"], device=device)
    label_idx = SHAPE_TO_LABEL[args.label]
    label_tensor = torch.tensor([label_idx], dtype=torch.long, device=device)

    with torch.no_grad():
        generated_img, cross_maps, self_maps = generator(z, label_tensor)

    # Save image
    os.makedirs(args.output_dir, exist_ok=True)
    img_path = os.path.join(args.output_dir, "generated_attention_shape.png")
    save_image(generated_img, img_path, normalize=True)
    logger.info(f"Generated shape saved to: {img_path}")

    # Plot Cross Attention Map (Query-Key similarity matrix)
    # cross_maps shape: B x N x 1 -> reshape to spatial grid
    cross_grid = cross_maps[0].view(16, 16).cpu().numpy()  # spatial resolution 16x16
    plt.figure(figsize=(6, 5))
    sns.heatmap(cross_grid, cmap="viridis", cbar=True)
    plt.title(f"Cross-Attention Map for label: {args.label}")
    plt.axis("off")
    cross_path = os.path.join(args.output_dir, "cross_attention_map.png")
    plt.savefig(cross_path)
    plt.close()
    logger.info(f"Saved Cross-Attention Map to: {cross_path}")

    # Plot Self Attention Map (first token attention)
    # self_maps shape: B x N x N (where N=32*32=1024)
    # Let's extract the attention map for a central pixel query
    N = self_maps.shape[1]
    pixel_idx = N // 2 + 16  # central query pixel
    self_grid = self_maps[0, pixel_idx].view(32, 32).cpu().numpy()  # spatial resolution 32x32
    plt.figure(figsize=(6, 5))
    sns.heatmap(self_grid, cmap="rocket", cbar=True)
    plt.title("Self-Attention Map (Central Query Pixel)")
    plt.axis("off")
    self_path = os.path.join(args.output_dir, "self_attention_map.png")
    plt.savefig(self_path)
    plt.close()
    logger.info(f"Saved Self-Attention Map to: {self_path}")

if __name__ == "__main__":
    main()
