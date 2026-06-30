import os
import json
import logging
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, CLIPTextModel, T5EncoderModel
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    Modular Text Embedding Generator supporting BERT, CLIP, T5,
    and Sentence Transformers models with batch processing and multiple export formats.
    """
    def __init__(self, model_type: str = "sentence-transformer", config: dict = None):
        self.model_type = model_type
        self.config = config or {}
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Embedding Generator loading model on device: {self.device}")
        
        # Load appropriate model and tokenizer
        if model_type == "bert":
            model_name = self.config.get("bert_model_name", "prajjwal1/bert-tiny")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
        elif model_type == "clip":
            model_name = self.config.get("clip_model_name", "hf-internal-testing/tiny-random-clip")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = CLIPTextModel.from_pretrained(model_name).to(self.device)
        elif model_type == "t5":
            model_name = self.config.get("t5_model_name", "google/t5-efficient-tiny")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = T5EncoderModel.from_pretrained(model_name).to(self.device)
        elif model_type == "sentence-transformer":
            model_name = self.config.get("sentence_transformer_name", "all-MiniLM-L6-v2")
            self.model = SentenceTransformer(model_name, device=str(self.device))
            self.tokenizer = self.model.tokenizer
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        logger.info(f"Successfully loaded '{model_type}' model: {model_name}")

    def embed(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        """
        Generate embeddings for a list of texts (supports batch processing).
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings_list = []
        
        if self.model_type == "sentence-transformer":
            # Sentence-Transformers handles batching internally
            embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=False)
            return np.array(embeddings)
            
        # Manually batch process for Hugging Face models
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                if self.model_type == "bert":
                    outputs = self.model(**inputs)
                    # Mean pooling over token embeddings
                    attention_mask = inputs["attention_mask"].unsqueeze(-1)
                    token_embeddings = outputs.last_hidden_state
                    sum_embeddings = torch.sum(token_embeddings * attention_mask, 1)
                    sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
                    batch_embeds = (sum_embeddings / sum_mask).cpu().numpy()
                elif self.model_type == "clip":
                    outputs = self.model(**inputs)
                    # Pooler output represents sentence embedding
                    batch_embeds = outputs.pooler_output.cpu().numpy()
                elif self.model_type == "t5":
                    outputs = self.model(**inputs)
                    # Mean pool T5 hidden states
                    token_embeddings = outputs.last_hidden_state
                    attention_mask = inputs["attention_mask"].unsqueeze(-1)
                    sum_embeddings = torch.sum(token_embeddings * attention_mask, 1)
                    sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
                    batch_embeds = (sum_embeddings / sum_mask).cpu().numpy()
                    
            embeddings_list.append(batch_embeds)
            
        return np.concatenate(embeddings_list, axis=0)

    @staticmethod
    def export_embeddings(embeddings: np.ndarray, texts: list[str], output_path: str, fmt: str = "numpy"):
        """
        Exports generated embeddings to NumPy, Torch, or JSON.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if fmt == "numpy":
            np.save(output_path, embeddings)
        elif fmt == "torch":
            torch.save(torch.tensor(embeddings), output_path)
        elif fmt == "json":
            data = [{"text": text, "embedding": embedding.tolist()} for text, embedding in zip(texts, embeddings)]
            with open(output_path, "w") as f:
                json.dump(data, f, indent=4)
        else:
            raise ValueError(f"Unknown format: {fmt}")
            
        logger.info(f"Exported embeddings to {output_path} (Format: {fmt})")

    @staticmethod
    def load_embeddings(input_path: str, fmt: str = "numpy"):
        """
        Loads embeddings from file.
        """
        if fmt == "numpy":
            return np.load(input_path)
        elif fmt == "torch":
            return torch.load(input_path).numpy()
        elif fmt == "json":
            with open(input_path, "r") as f:
                data = json.load(f)
            return np.array([item["embedding"] for item in data])
        else:
            raise ValueError(f"Unknown format: {fmt}")
