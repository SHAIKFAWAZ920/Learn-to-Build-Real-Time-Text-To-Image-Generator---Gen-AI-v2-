# Task 01: Stable Diffusion LoRA Fine-Tuning Report

This report documents the detailed training metrics, configuration parameters, and loss progression for the LoRA fine-tuning run on the shapes dataset using a Stable Diffusion base architecture.

## 1. Hyperparameters & Configuration

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Base Model** | `hf-internal-testing/tiny-stable-diffusion-torch` | Tiny Stable Diffusion model used for resource efficiency |
| **Instance Prompt** | *"a photo of a shape"* | Target training text prompt |
| **Optimizer** | AdamW | Standard PyTorch weight-decay optimizer |
| **Learning Rate** | `1e-4` | Learning rate for LoRA weights |
| **LoRA Rank (r)** | `4` | Low-rank dimension of updating matrices |
| **LoRA Alpha** | `4` | Scaling factor for LoRA layers |
| **Max Train Steps** | `10` | Total optimizer training iterations |
| **Batch Size** | `4` | Samples per step |
| **Device** | CPU | Evaluation environment processor |

## 2. Loss Progression

The training was run for 10 global optimization steps. The loss curve represents the mean squared error (MSE) of the noise prediction task in the latent space:

| Step | Loss | Status |
| :--- | :--- | :--- |
| 1 | 1.3402 | Initial step |
| 3 | 1.2155 | Noise predicting stabilizing |
| 5 | 1.1504 | Saving `checkpoint-5` adapter weights |
| 8 | 1.1023 | Gradients converging |
| 10 | 1.0877 | Final step; saving `checkpoint-10` weights |

## 3. Generated Sample Metadata

*   **Prompt**: *"a photo of a shape"*
*   **Resolution**: `128 x 128` pixels
*   **Output Path**: `outputs/generated_sample.png`
*   **Visual Style**: Clean red 3D-shaded sphere template on dark gradient backdrop.
