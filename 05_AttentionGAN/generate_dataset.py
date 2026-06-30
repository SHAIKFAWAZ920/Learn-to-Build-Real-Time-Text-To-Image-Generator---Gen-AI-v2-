import os
import argparse
import random
import logging
from PIL import Image, ImageDraw
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SHAPES = ["circle", "square", "triangle", "rectangle", "star", "diamond", "heart", "hexagon"]

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
        y = 13 * np.cos(val) - 5 * np.cos(2*val) - 2 * np.cos(3*val) - np.cos(4*val)
        points.append((cx + x * (size / 32), cy - y * (size / 32)))
    draw.polygon(points, fill=fill_color)

def draw_shape(draw, shape_type, x, y, size, fill_color):
    if shape_type == "circle":
        draw.ellipse([x, y, x + size, y + size], fill=fill_color)
    elif shape_type == "square":
        draw.rectangle([x, y, x + size, y + size], fill=fill_color)
    elif shape_type == "triangle":
        draw.polygon([(x + size//2, y), (x, y + size), (x + size, y + size)], fill=fill_color)
    elif shape_type == "rectangle":
        draw.rectangle([x, y + size//4, x + size, y + 3*size//4], fill=fill_color)
    elif shape_type == "star":
        draw_star(draw, (x + size//2, y + size//2), size, fill_color)
    elif shape_type == "diamond":
        draw.polygon([(x + size//2, y), (x + size, y + size//2), (x + size//2, y + size), (x, y + size//2)], fill=fill_color)
    elif shape_type == "heart":
        draw_heart(draw, (x + size//2, y + size//2), size, fill_color)
    elif shape_type == "hexagon":
        points = []
        for i in range(6):
            angle = i * np.pi / 3
            px = x + size//2 + (size//2) * np.cos(angle)
            py = y + size//2 + (size//2) * np.sin(angle)
            points.append((px, py))
        draw.polygon(points, fill=fill_color)

def main():
    parser = argparse.ArgumentParser(description="Generate Attention GAN Shape Dataset")
    parser.add_argument("--output_dir", type=str, default="dataset", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=800, help="Number of total samples to generate")
    parser.add_argument("--size", type=int, default=64, help="Image size")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Generating synthetic shape dataset with {args.num_samples} samples...")

    for i in range(args.num_samples):
        shape_type = SHAPES[i % len(SHAPES)]
        
        img = Image.new("L", (args.size, args.size), color=0)
        draw = ImageDraw.Draw(img)
        
        size = random.randint(24, 48)
        x = (args.size - size) // 2
        y = (args.size - size) // 2
        
        draw_shape(draw, shape_type, x, y, size, 255)
        
        img_path = os.path.join(args.output_dir, f"{shape_type}_{i:04d}.png")
        img.save(img_path)

    logger.info("Dataset generated successfully!")

if __name__ == "__main__":
    main()
