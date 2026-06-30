import os
import logging
from PIL import Image, ImageDraw

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    dataset_dir = "dataset"
    os.makedirs(dataset_dir, exist_ok=True)
    
    logger.info(f"Setting up sample DreamBooth dataset in '{dataset_dir}'...")
    
    # Create 5 simple shape images to train on
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
    ]
    
    for i, color in enumerate(colors):
        img = Image.new("RGB", (64, 64), color=(30, 30, 30))
        draw = ImageDraw.Draw(img)
        # Draw a shape
        draw.ellipse([8, 8, 56, 56], fill=color)
        
        img_path = os.path.join(dataset_dir, f"shape_{i+1}.png")
        img.save(img_path)
        logger.info(f"Generated sample image: {img_path}")
        
    logger.info("Sample DreamBooth dataset setup complete!")

if __name__ == "__main__":
    main()
