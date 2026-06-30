import time
import logging
import argparse
import yaml
import numpy as np
import torch

from embedder import EmbeddingGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_benchmark(config_path: str = "configs/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Test datasets
    test_sentences = [
        "A photo of a circle",
        "A beautiful red star with glowing edges",
        "A small green triangle shape against a solid black background",
        "A complex blue hexagon pattern",
        "A simple yellow square outline",
        "A pink heart symbol of love",
        "A shiny silver diamond shape",
        "A large gray rectangle shape"
    ] * 5  # 40 sentences total

    models_to_test = ["bert", "clip", "sentence-transformer"]
    logger.info(f"Starting performance benchmarks on {len(test_sentences)} samples...")

    print("\n" + "="*70)
    print(f"{'Model Type':<25} | {'Latency (ms)':<15} | {'Throughput (sen/sec)':<20}")
    print("="*70)

    for model_type in models_to_test:
        try:
            generator = EmbeddingGenerator(model_type=model_type, config=config["model"])
            
            # Warmup
            _ = generator.embed(test_sentences[:2])
            
            # Benchmark run
            start_time = time.perf_counter()
            embeddings = generator.embed(test_sentences, batch_size=16)
            end_time = time.perf_counter()
            
            total_time_ms = (end_time - start_time) * 1000
            avg_latency = total_time_ms / len(test_sentences)
            throughput = len(test_sentences) / (end_time - start_time)
            
            print(f"{model_type:<25} | {avg_latency:<15.2f} | {throughput:<20.2f}")
        except Exception as e:
            logger.error(f"Failed to benchmark model {model_type}: {e}")

    print("="*70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Text Embeddings")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config")
    args = parser.parse_args()
    
    run_benchmark(args.config)
