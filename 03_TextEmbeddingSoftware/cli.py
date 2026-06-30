import argparse
import logging
import yaml
import os

from src.embedder import EmbeddingGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Text Embedding CLI Tool")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config")
    parser.add_argument("--text", type=str, nargs="+", help="Direct text sentences to embed")
    parser.add_argument("--file", type=str, help="Path to a text file containing sentences (one per line)")
    parser.add_argument("--model_type", type=str, default="sentence-transformer", choices=["bert", "clip", "t5", "sentence-transformer"], help="Model type to use")
    parser.add_argument("--format", type=str, default="numpy", choices=["numpy", "torch", "json"], help="Output export format")
    parser.add_argument("--output", type=str, default="embeddings/output_embeddings", help="Base path for output file (extension added automatically)")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Resolve texts
    texts = []
    if args.text:
        texts = args.text
    elif args.file:
        if os.path.exists(args.file):
            with open(args.file, "r") as f:
                texts = [line.strip() for line in f if line.strip()]
        else:
            logger.error(f"Input file not found at: {args.file}")
            return
    else:
        logger.error("You must specify either --text or --file.")
        return

    logger.info(f"Loaded {len(texts)} texts for embedding.")

    # Generate
    generator = EmbeddingGenerator(model_type=args.model_type, config=config["model"])
    embeddings = generator.embed(texts)

    # Save
    ext_mapping = {"numpy": ".npy", "torch": ".pt", "json": ".json"}
    output_path = args.output + ext_mapping[args.format]
    
    generator.export_embeddings(embeddings, texts, output_path, fmt=args.format)
    logger.info("CLI operation completed successfully.")

if __name__ == "__main__":
    main()
