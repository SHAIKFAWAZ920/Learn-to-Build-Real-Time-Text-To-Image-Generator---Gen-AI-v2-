import os
import sys
import io
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Add local src path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.pipeline import TextToImagePipeline

app = FastAPI(
    title="End-to-End Text-to-Image Pipeline",
    description="REST API service to generate images from raw text descriptions.",
    version="1.0.0"
)

# Load configurations
config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Cache pipelines
pipelines = {}

def get_pipeline(mode):
    if mode not in pipelines:
        pipelines[mode] = TextToImagePipeline(mode=mode, config=config["pipeline"])
    return pipelines[mode]

class GenerationRequest(BaseModel):
    prompt: str
    mode: str = "attention-gan"

@app.post("/generate")
async def generate_image(request: GenerationRequest):
    """
    Generate an image from text. Returns binary image stream.
    """
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt query cannot be empty.")
        
    try:
        pipeline = get_pipeline(request.mode)
        pil_img, metadata = pipeline.generate(request.prompt)
        
        # Save image to byte stream
        img_byte_arr = io.BytesIO()
        pil_img.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)
        
        # Add metadata headers
        headers = {
            "X-Model-Mode": request.mode,
            "X-Matched-Label": metadata.get("matched_label", "none"),
            "X-Semantic-Similarity": str(metadata.get("similarity", 0.0))
        }
        
        return StreamingResponse(img_byte_arr, media_type="image/png", headers=headers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "running"}
