import os
import random
import logging
import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision.utils import save_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def generate_shape_dataset(output_dir: str, num_images: int = 200, img_size: int = 64):
    """
    Generates a synthetic dataset of simple shapes (circles, squares)
    and saves them in the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Generating synthetic dataset with {num_images} images in {output_dir}...")
    
    shapes = ["circle", "square"]
    
    for i in range(num_images):
        img = Image.new("L", (img_size, img_size), color=0)
        draw = ImageDraw.Draw(img)
        
        shape = random.choice(shapes)
        # Random size and position
        size = random.randint(16, 40)
        x = random.randint(0, img_size - size)
        y = random.randint(0, img_size - size)
        
        if shape == "circle":
            draw.ellipse([x, y, x + size, y + size], fill=255)
        elif shape == "square":
            draw.rectangle([x, y, x + size, y + size], fill=255)
            
        img.save(os.path.join(output_dir, f"shape_{i:04d}.png"))
        
    logger.info("Dataset generation complete.")

class ShapeDataset(torch.utils.data.Dataset):
    """
    Loads shape images and normalizes them.
    """
    def __init__(self, dataset_dir: str, img_size: int = 64):
        self.dataset_dir = dataset_dir
        self.img_names = [f for f in os.listdir(dataset_dir) if f.endswith(".png")]
        self.img_size = img_size

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_dir, self.img_names[idx])
        img = Image.open(img_path).convert("L")
        
        # Convert to tensor and normalize to [-1, 1]
        img_t = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
        img_t = (img_t - 0.5) / 0.5
        img_t = img_t.unsqueeze(0)  # Add channel dim: 1x64x64
        return img_t
