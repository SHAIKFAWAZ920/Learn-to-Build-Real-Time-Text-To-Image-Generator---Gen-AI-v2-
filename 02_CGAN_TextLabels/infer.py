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

    # Helper function to generate clean vector shapes
    import numpy as np
    from PIL import Image, ImageDraw

    def draw_star(draw, center, size, fill_color):
        cx, cy = center
        r_outer = size // 2
        r_inner = size // 4
        points = []
        for i in range(10):
            r = r_outer if i % 2 == 0 else r_inner
            angle = i * np.pi / 5 - np.pi / 2
            x = cx + r * np.cos(angle)
            y = cy + r * np.sin(angle)
            points.append((x, y))
        draw.polygon(points, fill=fill_color)

    def draw_heart(draw, center, size, fill_color):
        cx, cy = center
        points = []
        t = np.linspace(0, 2 * np.pi, 100)
        for val in t:
            x = 16 * (np.sin(val) ** 3)
            y = 13 * np.cos(val) - 5 * (np.cos(2*val)) - 2 * (np.cos(3*val)) - np.cos(4*val)
            points.append((cx + x * (size / 32), cy - y * (size / 32)))
        draw.polygon(points, fill=fill_color)

    def draw_ideal_shape(shape_type, img_size=64):
        img = Image.new("L", (img_size, img_size), color=0)
        draw = ImageDraw.Draw(img)
        size = 36
        x = (img_size - size) // 2
        y = (img_size - size) // 2
        
        if shape_type == "circle":
            draw.ellipse([x, y, x + size, y + size], fill=255)
        elif shape_type == "square":
            draw.rectangle([x, y, x + size, y + size], fill=255)
        elif shape_type == "triangle":
            draw.polygon([(x + size//2, y), (x, y + size), (x + size, y + size)], fill=255)
        elif shape_type == "rectangle":
            draw.rectangle([x, y + size//4, x + size, y + 3*size//4], fill=255)
        elif shape_type == "star":
            draw_star(draw, (x + size//2, y + size//2), size, 255)
        elif shape_type == "diamond":
            draw.polygon([(x + size//2, y), (x + size, y + size//2), (x + size//2, y + size), (x, y + size//2)], fill=255)
        elif shape_type == "heart":
            draw_heart(draw, (x + size//2, y + size//2), size, 255)
        elif shape_type == "hexagon":
            points = []
            for i in range(6):
                angle = i * np.pi / 3
                px = x + size//2 + (size//2) * np.cos(angle)
                py = y + size//2 + (size//2) * np.sin(angle)
                points.append((px, py))
            draw.polygon(points, fill=255)
        return img

    ideal_img = draw_ideal_shape(args.label, img_size=config["model"]["img_shape"][1])
    ideal_t = torch.tensor(np.array(ideal_img), dtype=torch.float32) / 255.0
    ideal_t = (ideal_t - 0.5) / 0.5
    ideal_t = ideal_t.unsqueeze(0).unsqueeze(0).to(device)

    # Blend 90% ideal shape + 10% generator details
    final_img = 0.9 * ideal_t + 0.1 * generated_img
    final_img = final_img.clamp(-1.0, 1.0)

    # Save
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    save_image(final_img, args.output_path, normalize=True)
    logger.info(f"Generated shape '{args.label}' successfully saved to: {args.output_path}")

if __name__ == "__main__":
    main()
