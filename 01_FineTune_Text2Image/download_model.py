import argparse
import logging
from diffusers import StableDiffusionPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Download Stable Diffusion weights from Hugging Face")
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="hf-internal-testing/tiny-stable-diffusion-torch",
        help="Hugging Face model ID (e.g. runwayml/stable-diffusion-v1-5)"
    )
    args = parser.parse_args()

    logger.info(f"Downloading model '{args.model_id}' from Hugging Face...")
    
    # Download pipeline and save to local Hugging Face cache
    pipe = StableDiffusionPipeline.from_pretrained(args.model_id)
    
    logger.info(f"Successfully downloaded and cached model '{args.model_id}'!")

if __name__ == "__main__":
    main()
