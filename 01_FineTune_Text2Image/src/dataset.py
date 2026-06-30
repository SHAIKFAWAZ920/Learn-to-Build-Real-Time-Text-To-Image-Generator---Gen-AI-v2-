import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class DreamBoothDataset(Dataset):
    """
    A custom dataset loader for DreamBooth-style training.
    Loads images from a folder and returns them with tokenized prompt IDs.
    """
    def __init__(
        self,
        instance_data_dir: str,
        instance_prompt: str,
        tokenizer,
        size: int = 512,
    ):
        self.instance_data_dir = instance_data_dir
        self.instance_prompt = instance_prompt
        self.tokenizer = tokenizer
        self.size = size

        if not os.path.exists(instance_data_dir):
            os.makedirs(instance_data_dir, exist_ok=True)
            # Create a dummy image if empty for self-containment
            dummy_img = Image.new("RGB", (size, size), color=(255, 0, 0))
            dummy_img.save(os.path.join(instance_data_dir, "dummy_1.png"))

        self.instance_images_path = [
            os.path.join(instance_data_dir, path)
            for path in os.listdir(instance_data_dir)
            if path.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        self.num_instance_images = len(self.instance_images_path)
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self.num_instance_images

    def __getitem__(self, index):
        example = {}
        image_path = self.instance_images_path[index]
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
            
        example["instance_images"] = self.image_transforms(image)
        
        # Tokenize the prompt
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        
        return example
