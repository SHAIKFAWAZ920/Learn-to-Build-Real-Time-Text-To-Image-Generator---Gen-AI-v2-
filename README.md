# End-to-End Text-to-Image Generation Workspace

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub Actions CI](https://github.com/workflows/test.yml/badge.svg)](06_Text2ImagePipeline/.github/workflows/test.yml)

Welcome to the **Advanced Text-to-Image AI/ML Engineering Internship Workspace**. This workspace contains **six self-contained project repositories** and a **baseline training project**, representing a complete pipeline from text encoding and dataset profiling to custom attention-based GAN shape generation and parameter-efficient diffusion fine-tuning.

---

## Workspace Navigation Links

*   [**Original Training Project (Baseline)**](./original_training_project) - Unconditional MLP-based Shape GAN
*   [**Task 01: LoRA Fine-Tuning**](./01_FineTune_Text2Image) - Stable Diffusion Fine-Tuning using memory-efficient LoRA adapters
*   [**Task 02: Conditional GAN**](./02_CGAN_TextLabels) - Class-conditional DC-CGAN shape generator
*   [**Task 03: Text Embedding Software**](./03_TextEmbeddingSoftware) - Text encoder CLI, API (FastAPI) and UI (Gradio) using SBERT/BERT/CLIP/T5
*   [**Task 04: Dataset Explorer**](./04_DatasetExplorer) - Statistical dashboard profiling sizes, counts, aspect ratios, and word freqs
*   [**Task 05: Attention-Augmented GAN**](./05_AttentionGAN) - Self & Cross Attention shape generator with IS and FID evaluators
*   [**Task 06: Text-to-Image Pipeline**](./06_Text2ImagePipeline) - E2E REST API, Gradio comparative UI, and multi-container Docker deployment

---

## System Architecture Overview

The following diagram shows how the sub-modules interface to create a unified system:

```mermaid
graph TD
    UserQuery[User Text Prompt] --> T3[Task 03: Embedding Generator]
    T3 --> Vector[High-Dim Embedding Vector]
    
    Vector --> T6[Task 06: End-to-End Pipeline]
    
    T6 -->|Mode A: GAN| T5[Task 05: Attention GAN]
    T6 -->|Mode B: Diffusion| T1[Task 01: SD LoRA Finetune]
    
    T5 --> ShapeImage[Shape Image Output]
    T1 --> PhotoImage[DreamBooth Image Output]
    
    T4[Task 04: Dataset Explorer] -->|Profile / Audit| Dataset[Custom Dataset / COCO]
    Dataset --> T1
    Dataset --> T5
```

---

## Methodology Progression

The codebase follows a clear academic and engineering path:
1.  **Baseline (`original_training_project`)**: We begin with a vanilla MLP architecture generating shapes unconditionally from raw noise.
2.  **Conditioning (`02_CGAN_TextLabels`)**: We introduce convolutional layers (DCGAN) and direct class-label conditioning.
3.  **Attention (`05_AttentionGAN`)**: We add Self-Attention layers in Generator & Discriminator to capture spatial geometry and Cross-Attention to map label vectors onto feature grids.
4.  **Scaling (`01_FineTune_Text2Image`)**: We scale to a billion-parameter Stable Diffusion architecture and utilize Low-Rank Adaptation (LoRA) for parameter-efficient adaptation.

---

## Global Dataset Information
Our synthetic shape dataset supports 8 categories: `circle`, `square`, `triangle`, `rectangle`, `star`, `diamond`, `heart`, and `hexagon`.
- **Images**: 800 grayscale 64x64 PNG drawings.
- **Image Resolution**: 64x64.
- **Caption mapping**: Automatic prefix mapping (`star_0001.png` maps to caption `"A photo of a star"`).
- **Train/Val/Test Split**: 80% / 10% / 10%.

---

## How to Set Up the Workspace

### 1. Clone & Install Dependencies
Ensure you have Python 3.11 installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Run All Tests
Verify the entire codebase:
```bash
pytest 01_FineTune_Text2Image/tests/
pytest 02_CGAN_TextLabels/tests/
pytest 03_TextEmbeddingSoftware/tests/
pytest 04_DatasetExplorer/tests/
pytest 05_AttentionGAN/tests/
pytest 06_Text2ImagePipeline/tests/
```

### 3. Build Docker Environment (Pipeline Task 6)
Build and run the entire suite in Docker containers:
```bash
cd 06_Text2ImagePipeline/docker
docker-compose up --build
```

---

## Expected Outputs and Logs
- **Checkpoints**: Saved in `models/` or `outputs/` folders under each respective directory.
- **TensorBoard**: Launch logging views by running `tensorboard --logdir .` at the workspace root.
- **Heatmaps**: Attention maps are exported to `05_AttentionGAN/outputs/` as PNG files.

---

## Global Framework & Pipeline Workings Summary

### Task 01: LoRA Fine-Tuning of Stable Diffusion
*   **Framework**: **PyTorch** & **Hugging Face PEFT** (`LoraConfig`, `add_adapter`), **Diffusers** (`StableDiffusionPipeline`, `UNet2DConditionModel`), **Transformers** (`CLIPTextModel`).
*   **Pipeline Working**: Loads custom images $\rightarrow$ tokenizes prompt $\rightarrow$ projects to latents via VAE $\rightarrow$ injects adapter into UNet cross-attention blocks $\rightarrow$ predicts added noise $\rightarrow$ backpropagates MSE loss to train adapters $\rightarrow$ inference loads adapter weights and saves shape output.

### Task 02: Conditional GAN Shape Generator
*   **Framework**: **PyTorch** (`torch.nn`, `torch.optim`) & **Pillow** (for dataset vector shape drawing).
*   **Pipeline Working**: Generates 2D shape dataset $\rightarrow$ feeds Generator with latent noise $z$ + class embedding $\rightarrow$ feeds Discriminator with image-label pair $\rightarrow$ updates weights using adversarial BCE Loss $\rightarrow$ blends outputs based on epoch progress to save clean sample grids.

### Task 03: Text Embedding Generation Suite
*   **Framework**: **Hugging Face Transformers** & **SentenceTransformers** (`all-MiniLM-L6-v2`), **FastAPI**, **Gradio**, **NumPy**.
*   **Pipeline Working**: Tokenizes input strings $\rightarrow$ extracts token-level vectors via MiniLM transformer layers $\rightarrow$ performs Mean Pooling over attention masks $\rightarrow$ L2-normalizes to yield unit-length 384-d vector $\rightarrow$ saves to `.npy` file.

### Task 04: Dataset Explorer & Profiling Tool
*   **Framework**: **Pandas** & **NumPy** (statistics engine), **Matplotlib** & **Seaborn** (plotting), **Fpdf2** (PDF report compilation), **Gradio** (dashboard UI).
*   **Pipeline Working**: Scans directories for image-caption files $\rightarrow$ computes metrics (aspect ratios, token counts, vocab distribution) $\rightarrow$ saves plots $\rightarrow$ compiles PDF/HTML/CSV tables under `outputs/`.

### Task 05: Attention-Augmented GAN
*   **Framework**: **PyTorch** (Self/Cross Attention blocks) & **Scipy** (for FID matrix math calculation).
*   **Pipeline Working**: Generator upsamples latents $\rightarrow$ maps pixel relations via Self-Attention $\rightarrow$ aligns pixels to category embeddings via Cross-Attention $\rightarrow$ Discriminator scores fake/real $\rightarrow$ computes IS/FID metrics $\rightarrow$ plots attention weight heatmaps.

### Task 06: End-to-End Pipeline
*   **Framework**: **SentenceTransformers**, **Attention GAN Generator**, **FastAPI**, **Gradio**, **Docker Compose**.
*   **Pipeline Working**: Ingests text prompt $\rightarrow$ extracts sentence embedding $\rightarrow$ maps similarity ranking across 8 categories $\rightarrow$ passes highest similarity index (e.g. `"star"`) to generator $\rightarrow$ blends output image with vector shape template to save a sharp `star_pipeline.png`.

---

## License
Licensed under the MIT License.
