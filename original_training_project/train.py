import os
import argparse
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from model import BaselineGenerator, BaselineDiscriminator
from utils import generate_shape_dataset, ShapeDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train Baseline Unconditional GAN")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--latent_dim", type=int, default=100, help="Latent dimension size")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Dataset directory")
    parser.add_argument("--outputs_dir", type=str, default="outputs", help="Output sample directory")
    parser.add_argument("--checkpoints_dir", type=str, default="models", help="Checkpoint directory")
    args = parser.parse_args()

    # Create directories
    os.makedirs(args.outputs_dir, exist_ok=True)
    os.makedirs(args.checkpoints_dir, exist_ok=True)

    # Generate synthetic shapes if not present
    if not os.path.exists(args.dataset_dir) or len(os.listdir(args.dataset_dir)) == 0:
        generate_shape_dataset(args.dataset_dir, num_images=100)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Dataset & Dataloader
    dataset = ShapeDataset(args.dataset_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Initialize model
    generator = BaselineGenerator(latent_dim=args.latent_dim).to(device)
    discriminator = BaselineDiscriminator().to(device)

    # Loss function
    adversarial_loss = nn.BCELoss()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Training loop
    for epoch in range(args.epochs):
        for i, imgs in enumerate(dataloader):
            batch_size = imgs.shape[0]
            imgs = imgs.to(device)

            # Ground truths
            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Sample noise
            z = torch.randn(batch_size, args.latent_dim, device=device)

            # Generate batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            if i % 10 == 0:
                logger.info(
                    f"[Epoch {epoch}/{args.epochs}] [Batch {i}/{len(dataloader)}] "
                    f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]"
                )

        # Save samples at end of epoch
        save_image(gen_imgs.data[:16], os.path.join(args.outputs_dir, f"epoch_{epoch}.png"), nrow=4, normalize=True)

    # Save final weights
    torch.save(generator.state_dict(), os.path.join(args.checkpoints_dir, "generator.pth"))
    torch.save(discriminator.state_dict(), os.path.join(args.checkpoints_dir, "discriminator.pth"))
    logger.info("Baseline training successfully complete. Checkpoints saved.")

if __name__ == "__main__":
    main()
