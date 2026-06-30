import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

SHAPES = ["circle", "square", "triangle", "rectangle", "star", "diamond", "heart", "hexagon"]
SHAPE_TO_LABEL = {shape: idx for idx, shape in enumerate(SHAPES)}

class CGANShapeDataset(Dataset):
    """
    Loads shape images, extracts their label from the filename,
    and normalizes them.
    """
    def __init__(self, dataset_dir: str, img_size: int = 64):
        self.dataset_dir = dataset_dir
        self.img_size = img_size
        
        if not os.path.exists(dataset_dir) or len(os.listdir(dataset_dir)) == 0:
            raise FileNotFoundError(
                f"Dataset directory '{dataset_dir}' does not exist or is empty. "
                f"Please run 'generate_dataset.py' first."
            )
            
        self.img_names = [f for f in os.listdir(dataset_dir) if f.endswith(".png")]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        filename = self.img_names[idx]
        img_path = os.path.join(self.dataset_dir, filename)
        
        # Load grayscale image
        img = Image.open(img_path).convert("L")
        if img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
            
        # Convert to tensor and normalize to [-1, 1]
        img_t = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
        img_t = (img_t - 0.5) / 0.5
        img_t = img_t.unsqueeze(0)  # 1x64x64
        
        # Get label index
        shape_name = filename.split("_")[0]
        label = SHAPE_TO_LABEL[shape_name]
        
        return img_t, label
