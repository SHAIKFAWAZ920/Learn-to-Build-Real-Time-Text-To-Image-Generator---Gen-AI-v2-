import os
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_cmd(cmd, cwd):
    logger.info(f"Running command: '{cmd}' in cwd: '{cwd}'")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Command failed with exit code {result.returncode}")
        logger.error(f"Stderr:\n{result.stderr}")
    else:
        logger.info(f"Command succeeded!")
        if result.stdout:
            logger.info(f"Stdout:\n{result.stdout[:500]}...")
    return result.returncode == 0

def main():
    root_dir = os.getcwd()
    logger.info(f"Starting output generation suite in {root_dir}...")

    # 1. Task 01: Stable Diffusion LoRA Fine-Tuning
    t1_dir = os.path.join(root_dir, "01_FineTune_Text2Image")
    logger.info("--- Executing Task 01 ---")
    run_cmd("python download_dataset.py", t1_dir)
    run_cmd("python download_model.py --model_id hf-internal-testing/tiny-stable-diffusion-torch", t1_dir)
    run_cmd("python train.py", t1_dir)
    run_cmd("python infer.py --checkpoint outputs/checkpoint-10 --output_path outputs/generated_sample.png", t1_dir)

    # 2. Task 02: Conditional GAN Shape Generator
    t2_dir = os.path.join(root_dir, "02_CGAN_TextLabels")
    logger.info("--- Executing Task 02 ---")
    run_cmd("python generate_dataset.py --num_samples 200", t2_dir)
    run_cmd("python train.py", t2_dir)
    run_cmd("python infer.py --label star --checkpoint models/generator.pth --output_path outputs/generated_star.png", t2_dir)

    # 3. Task 03: Text Embedding Generation Suite
    t3_dir = os.path.join(root_dir, "03_TextEmbeddingSoftware")
    logger.info("--- Executing Task 03 ---")
    run_cmd("python src/benchmark.py", t3_dir)
    run_cmd('python cli.py --text "A red circle" "A green star" --model_type sentence-transformer --format numpy --output outputs/test_embeds', t3_dir)

    # 4. Task 04: Dataset Explorer & Statistics Suite
    t4_dir = os.path.join(root_dir, "04_DatasetExplorer")
    logger.info("--- Executing Task 04 ---")
    run_cmd("python explore.py --dataset_dir dataset --format custom", t4_dir)

    # 5. Task 05: Attention-Augmented GAN (SAGAN + Cross-Attention)
    t5_dir = os.path.join(root_dir, "05_AttentionGAN")
    logger.info("--- Executing Task 05 ---")
    run_cmd("python generate_dataset.py --num_samples 200", t5_dir)
    run_cmd("python train.py", t5_dir)
    run_cmd("python infer.py --label heart --checkpoint models/generator.pth --output_dir outputs", t5_dir)

    # 6. Task 06: End-to-End Pipeline
    t6_dir = os.path.join(root_dir, "06_Text2ImagePipeline")
    logger.info("--- Executing Task 06 ---")
    run_cmd('python -c "import os; os.makedirs(\'outputs\', exist_ok=True); import sys; sys.path.append(\'src\'); from pipeline import TextToImagePipeline; pipe = TextToImagePipeline(); img, meta = pipe.generate(\'A glowing star\'); img.save(\'outputs/star_pipeline.png\'); print(meta)"', t6_dir)

    logger.info("Generation suite execution complete. Please review the output folders.")

if __name__ == "__main__":
    main()
