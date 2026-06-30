import os
import sys
import logging
import numpy as np
import torch
from PIL import Image

# Import embedding generator from task 3
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "03_TextEmbeddingSoftware")))
from src.embedder import EmbeddingGenerator

# Import Attention Generator from task 5
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "05_AttentionGAN")))
from src.attention_gan import AttentionGenerator
from src.dataset import SHAPES, SHAPE_TO_LABEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class TextToImagePipeline:
    """
    End-to-End Modular Text-To-Image Pipeline.
    Supports Mode A (Semantic Text Embedding -> Attention GAN shape generation)
    and Mode B (Stable Diffusion with LoRA adapter).
    """
    def __init__(self, mode: str = "attention-gan", config: dict = None):
        self.mode = mode
        self.config = config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Initializing Text-to-Image Pipeline in mode: {mode}")
        
        if mode == "attention-gan":
            # Load text embedding encoder (Sentence-Transformer)
            self.embedder = EmbeddingGenerator(model_type="sentence-transformer")
            
            # Embed labels for similarity matching
            self.shape_names = SHAPES
            self.shape_embeddings = self.embedder.embed(self.shape_names)
            
            # Load Attention-GAN generator
            latent_dim = self.config.get("latent_dim", 100)
            num_classes = self.config.get("num_classes", 8)
            embedding_dim = self.config.get("embedding_dim", 50)
            img_shape = self.config.get("img_shape", (1, 64, 64))
            
            self.generator = AttentionGenerator(
                latent_dim=latent_dim,
                num_classes=num_classes,
                embedding_dim=embedding_dim,
                img_shape=img_shape
            ).to(self.device)
            
            weights_path = self.config.get("gan_weights")
            if weights_path and os.path.exists(weights_path):
                logger.info(f"Loading GAN weights from {weights_path}")
                self.generator.load_state_dict(torch.load(weights_path, map_location=self.device))
            else:
                logger.warning(f"GAN weights not found at {weights_path}. Running with random weights.")
                
            self.generator.eval()
            
        elif mode == "stable-diffusion":
            from diffusers import StableDiffusionPipeline
            sd_model = self.config.get("sd_model_name", "hf-internal-testing/tiny-stable-diffusion-torch")
            
            logger.info(f"Loading Stable Diffusion pipeline: {sd_model}")
            self.pipe = StableDiffusionPipeline.from_pretrained(sd_model)
            
            lora_path = self.config.get("sd_lora_weights")
            if lora_path and os.path.exists(lora_path):
                logger.info(f"Loading LoRA weights from {lora_path}")
                self.pipe.unet.load_attn_procs(lora_path)
                
            self.pipe.to(self.device)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def generate(self, prompt: str) -> tuple[Image.Image, dict]:
        """
        Runs the end-to-end pipeline to generate an image from text.
        """
        logger.info(f"Pipeline generating image for prompt: '{prompt}'")
        
        metadata = {"prompt": prompt, "mode": self.mode}
        
        if self.mode == "attention-gan":
            # 1. Text Input -> Tokenizer -> Embedder (Task 3)
            query_embedding = self.embedder.embed([prompt])[0]
            
            # 2. Semantic matching: Find shape label closest to prompt embedding
            similarities = [
                np.dot(query_embedding, label_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(label_emb))
                for label_emb in self.shape_embeddings
            ]
            best_idx = int(np.argmax(similarities))
            matched_shape = self.shape_names[best_idx]
            match_score = float(similarities[best_idx])
            
            logger.info(f"Matched prompt '{prompt}' to category '{matched_shape}' (similarity: {match_score:.4f})")
            
            metadata["matched_label"] = matched_shape
            metadata["similarity"] = match_score
            
            # 3. Attention Module + Generator (Task 5)
            # Create input tensors
            z = torch.randn(1, self.generator.latent_dim, device=self.device)
            label_idx = SHAPE_TO_LABEL[matched_shape]
            label_tensor = torch.tensor([label_idx], dtype=torch.long, device=self.device)
            
            with torch.no_grad():
                img_tensor, cross_maps, self_maps = self.generator(z, label_tensor)
                
            # Convert grayscale tensor [-1, 1] to PIL Image
            img_tensor = (img_tensor[0] * 0.5 + 0.5).clamp(0, 1)
            img_np = (img_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            
            # Grayscale 2D array
            pil_img = Image.fromarray(img_np[:, :, 0], mode="L")
            return pil_img, metadata
            
        elif self.mode == "stable-diffusion":
            # Native HuggingFace Diffusers inference
            steps = 5 if "tiny" in self.config.get("sd_model_name", "") else 30
            img = self.pipe(prompt, num_inference_steps=steps).images[0]
            return img, metadata
