# Original Training Project: Baseline Unconditional Shape GAN

This is the baseline training project that serves as the foundation for the text-to-image internship workspace. It implements a simple **Unconditional Generative Adversarial Network (GAN)** using Multi-Layer Perceptrons (MLPs).

## Objective
The goal is to generate 64x64 grayscale shape-like images (circles/squares) from a random 100-dimensional Gaussian noise vector, without any class conditioning or attention mechanisms.

## Model Architecture

### Generator (MLP)
- Input: Latent vector \(z \in \mathbb{R}^{100}\)
- Layers:
  - Linear(100, 128) \(\rightarrow\) LeakyReLU
  - Linear(128, 256) \(\rightarrow\) BatchNorm1d \(\rightarrow\) LeakyReLU
  - Linear(256, 512) \(\rightarrow\) BatchNorm1d \(\rightarrow\) LeakyReLU
  - Linear(512, 1024) \(\rightarrow\) BatchNorm1d \(\rightarrow\) LeakyReLU
  - Linear(1024, 4096) \(\rightarrow\) Tanh
- Output: Flattened image reshaped to \(1 \times 64 \times 64\)

### Discriminator (MLP)
- Input: Flattened image \(x \in \mathbb{R}^{4096}\)
- Layers:
  - Linear(4096, 512) \(\rightarrow\) LeakyReLU
  - Linear(512, 256) \(\rightarrow\) LeakyReLU
  - Linear(256, 1) \(\rightarrow\) Sigmoid
- Output: Validity probability \([0, 1]\)

## Training Configuration
- **Optimizer**: Adam (lr=0.0002, betas=(0.5, 0.999))
- **Loss**: Binary Cross-Entropy (BCE) Loss
- **Epochs**: 5 (toy run)
- **Batch Size**: 32

## How to Run
1. Navigate to this directory:
   ```bash
   cd original_training_project
   ```
2. Start training (this will generate a synthetic shape dataset automatically if none exists):
   ```bash
   python train.py --epochs 5
   ```
3. Check the generated samples in `outputs/` and final checkpoints in `models/`.
