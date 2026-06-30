import os
import json
import logging
from collections import Counter
import pandas as pd
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class DatasetAnalyzer:
    """
    Analyzes image and caption statistics for COCO, Oxford-102, or custom folders.
    """
    def __init__(self, dataset_path: str, dataset_format: str = "custom"):
        self.dataset_path = dataset_path
        self.dataset_format = dataset_format
        
        self.image_stats = []
        self.captions = []
        self.classes = []

    def analyze(self) -> dict:
        """
        Executes analysis on the dataset directory.
        """
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path, exist_ok=True)
            self._create_toy_data()

        logger.info(f"Analyzing dataset in {self.dataset_path} (Format: {self.dataset_format})...")

        if self.dataset_format == "custom":
            self._analyze_custom()
        elif self.dataset_format == "coco":
            self._analyze_coco()
        elif self.dataset_format == "oxford":
            self._analyze_oxford()
        else:
            raise ValueError(f"Unknown format: {self.dataset_format}")

        # Compute summary metrics
        metrics = self._compute_summary()
        return metrics

    def _analyze_custom(self):
        """
        Loads custom dataset (images + matching caption txt files).
        """
        for filename in os.listdir(self.dataset_path):
            file_path = os.path.join(self.dataset_path, filename)
            
            # Process images
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                try:
                    with Image.open(file_path) as img:
                        w, h = img.size
                        self.image_stats.append({
                            "filename": filename,
                            "width": w,
                            "height": h,
                            "aspect_ratio": w / h
                        })
                except Exception as e:
                    logger.error(f"Failed to read image {filename}: {e}")

                # Process matching caption file
                base_name = os.path.splitext(filename)[0]
                caption_path = os.path.join(self.dataset_path, base_name + ".txt")
                if os.path.exists(caption_path):
                    with open(caption_path, "r", encoding="utf-8") as f:
                        caption = f.read().strip()
                        self.captions.append(caption)
                else:
                    # Fallback label parsed from filename prefix
                    shape_name = filename.split("_")[0]
                    self.captions.append(f"A photo of a {shape_name}")
                    self.classes.append(shape_name)

    def _analyze_coco(self):
        """
        Loads COCO JSON file format.
        """
        json_path = os.path.join(self.dataset_path, "annotations.json")
        if not os.path.exists(json_path):
            # Write a dummy COCO JSON for execution
            self._create_toy_coco()
            
        with open(json_path, "r") as f:
            coco_data = json.load(f)
            
        # Parse images
        for img_info in coco_data.get("images", []):
            self.image_stats.append({
                "filename": img_info["file_name"],
                "width": img_info["width"],
                "height": img_info["height"],
                "aspect_ratio": img_info["width"] / img_info["height"]
            })
            
        # Parse annotations (captions)
        for ann in coco_data.get("annotations", []):
            if "caption" in ann:
                self.captions.append(ann["caption"])
            if "category_id" in ann:
                self.classes.append(str(ann["category_id"]))

    def _analyze_oxford(self):
        """
        Loads class folder structure (Oxford-102 style).
        """
        for class_name in os.listdir(self.dataset_path):
            class_dir = os.path.join(self.dataset_path, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                        img_path = os.path.join(class_dir, img_name)
                        try:
                            with Image.open(img_path) as img:
                                w, h = img.size
                                self.image_stats.append({
                                    "filename": os.path.join(class_name, img_name),
                                    "width": w,
                                    "height": h,
                                    "aspect_ratio": w / h
                                })
                            self.classes.append(class_name)
                            self.captions.append(f"A photo of a {class_name}")
                        except Exception as e:
                            logger.error(f"Error reading {img_name}: {e}")

    def _compute_summary(self) -> dict:
        total_images = len(self.image_stats)
        if total_images == 0:
            return {"status": "empty"}

        df_img = pd.DataFrame(self.image_stats)
        
        # Word tokenization for caption stats
        all_words = []
        caption_lengths = []
        for cap in self.captions:
            words = cap.lower().replace(".", "").replace(",", "").split()
            all_words.extend(words)
            caption_lengths.append(len(words))
            
        vocab = Counter(all_words)
        
        summary = {
            "total_images": total_images,
            "mean_width": df_img["width"].mean(),
            "mean_height": df_img["height"].mean(),
            "aspect_ratios": df_img["aspect_ratio"].tolist(),
            "vocab_size": len(vocab),
            "total_words": len(all_words),
            "most_common_words": vocab.most_common(20),
            "caption_count": len(self.captions),
            "mean_caption_length": np.mean(caption_lengths) if caption_lengths else 0,
            "max_caption_length": int(np.max(caption_lengths)) if caption_lengths else 0,
            "min_caption_length": int(np.min(caption_lengths)) if caption_lengths else 0,
            "caption_lengths": caption_lengths,
            "class_distribution": dict(Counter(self.classes)) if self.classes else {}
        }
        return summary

    def _create_toy_data(self):
        """
        Creates synthetic training folder of shape drawings + text captions.
        """
        shapes = ["circle", "square", "triangle", "star"]
        for i in range(12):
            shape = shapes[i % len(shapes)]
            img = Image.new("RGB", (64, 64), color=(100 + i * 10, 50, 150))
            img_path = os.path.join(self.dataset_path, f"{shape}_{i:03d}.png")
            img.save(img_path)
            
            # Caption text
            cap_path = os.path.join(self.dataset_path, f"{shape}_{i:03d}.txt")
            with open(cap_path, "w") as f:
                f.write(f"A perfect drawing of a {shape} shape in high quality.")

    def _create_toy_coco(self):
        """
        Generates dummy annotations.json.
        """
        coco = {
            "images": [
                {"id": i, "file_name": f"img_{i}.png", "width": 64, "height": 64}
                for i in range(5)
            ],
            "annotations": [
                {"id": i, "image_id": i, "caption": "A drawing of a geometric shape", "category_id": 1}
                for i in range(5)
            ],
            "categories": [{"id": 1, "name": "shape"}]
        }
        with open(os.path.join(self.dataset_path, "annotations.json"), "w") as f:
            json.dump(coco, f, indent=4)
        
        # generate dummy images
        for img in coco["images"]:
            img_p = os.path.join(self.dataset_path, img["file_name"])
            Image.new("RGB", (64, 64), color=(40, 80, 120)).save(img_p)
