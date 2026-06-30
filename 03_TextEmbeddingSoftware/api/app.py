import os
import sys
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.embedder import EmbeddingGenerator

app = FastAPI(
    title="Text Embedding API",
    description="REST API to convert text descriptions into high-dimensional vector embeddings.",
    version="1.0.0"
)

# Load config
config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Global generator instance
generator = EmbeddingGenerator(
    model_type=config["model"]["default_type"],
    config=config["model"]
)

class EmbeddingRequest(BaseModel):
    texts: list[str]
    model_type: str = "sentence-transformer"
    batch_size: int = 16

class EmbeddingResponseItem(BaseModel):
    text: str
    embedding: list[float]

class EmbeddingResponse(BaseModel):
    embeddings: list[EmbeddingResponseItem]
    model_used: str

@app.post("/embed", response_model=EmbeddingResponse)
async def get_embeddings(request: EmbeddingRequest):
    """
    Generate embeddings for a list of text inputs.
    """
    if not request.texts:
        raise HTTPException(status_code=400, detail="Text list cannot be empty.")
        
    try:
        # Re-initialize generator if model_type changes
        global generator
        if request.model_type != generator.model_type:
            generator = EmbeddingGenerator(model_type=request.model_type, config=config["model"])
            
        embeddings = generator.embed(request.texts, batch_size=request.batch_size)
        
        response_items = [
            EmbeddingResponseItem(text=text, embedding=embedding.tolist())
            for text, embedding in zip(request.texts, embeddings)
        ]
        
        return EmbeddingResponse(embeddings=response_items, model_used=generator.model_type)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": generator.model_type}
