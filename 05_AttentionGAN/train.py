import os
import argparse
import logging
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from src.attention_gan import AttentionGenerator, AttentionDiscriminator
from src.dataset import CGANShapeDataset
from src.metrics import GANMetricsEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train Attention-Augmented GAN on Shapes")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    return parser.parse_args()

def main():
    args = parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Output directories
    outputs_dir = config["training"]["outputs_dir"]
    checkpoints_dir = config["training"]["checkpoints_dir"]
    logs_dir = config["training"]["logs_dir"]
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Seed
    torch.manual_seed(config["training"]["seed"])

    # Verify dataset exists (generates if empty)
    dataset_dir = config["training"]["dataset_dir"]
    if not os.path.exists(dataset_dir) or len(os.listdir(dataset_dir)) == 0:
        logger.info("Dataset empty or not found. Generating shapes...")
        os.system(f"python generate_dataset.py --output_dir {dataset_dir} --num_samples 200")

    # DataLoader
    dataset = CGANShapeDataset(dataset_dir)
    dataloader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True, drop_last=True)

    # Models
    generator = AttentionGenerator(
        latent_dim=config["model"]["latent_dim"],
        num_classes=config["model"]["num_classes"],
        embedding_dim=config["model"]["embedding_dim"],
        img_shape=tuple(config["model"]["img_shape"])
    ).to(device)

    discriminator = AttentionDiscriminator(
        num_classes=config["model"]["num_classes"],
        img_shape=tuple(config["model"]["img_shape"])
    ).to(device)

    # Metrics Evaluator
    evaluator = GANMetricsEvaluator(device=device)

    # Loss function
    adversarial_loss = nn.BCELoss()

    # Optimizers
    optimizer_G = torch.optim.Adam(
        generator.parameters(),
        lr=config["training"]["lr"],
        betas=(config["training"]["beta1"], config["training"]["beta2"])
    )
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(),
        lr=config["training"]["lr"],
        betas=(config["training"]["beta1"], config["training"]["beta2"])
    )

    writer = SummaryWriter(log_dir=logs_dir)

    epochs = config["training"]["epochs"]
    latent_dim = config["model"]["latent_dim"]
    num_classes = config["model"]["num_classes"]

    for epoch in range(epochs):
        for i, (imgs, labels) in enumerate(dataloader):
            batch_size = imgs.shape[0]
            
            real_imgs = imgs.to(device)
            labels = labels.to(device)

            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            z = torch.randn(batch_size, latent_dim, device=device)
            gen_labels = torch.randint(0, num_classes, (batch_size,), device=device)

            # Generator output includes attention maps
            gen_imgs, cross_maps, self_maps_g = generator(z, gen_labels)

            # D output includes attention maps
            validity, self_maps_d = discriminator(gen_imgs, gen_labels)
            g_loss = adversarial_loss(validity, valid)
            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            real_validity, _ = discriminator(real_imgs, labels)
            fake_validity, _ = discriminator(gen_imgs.detach(), gen_labels)
            
            real_loss = adversarial_loss(real_validity, valid)
            fake_loss = adversarial_loss(fake_validity, fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            # Logging
            if i % 5 == 0:
                logger.info(
                    f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] "
                    f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]"
                )
                step = epoch * len(dataloader) + i
                writer.add_scalar("Loss/Discriminator", d_loss.item(), step)
                writer.add_scalar("Loss/Generator", g_loss.item(), step)

        # Periodically evaluate IS and FID at the end of epoch
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            is_mean, is_std = evaluator.calculate_inception_score(gen_imgs, splits=1)
            fid_score = evaluator.calculate_fid(real_imgs[:16], gen_imgs[:16])
            
            logger.info(f"📈 Evaluation Epoch {epoch}: IS = {is_mean:.3f} | FID = {fid_score:.3f}")
            writer.add_scalar("Metrics/Inception_Score", is_mean, epoch)
            writer.add_scalar("Metrics/FID", fid_score, epoch)

        # Save sample outputs
        with torch.no_grad():
            eval_noise = torch.randn(num_classes, latent_dim, device=device)
            eval_labels = torch.arange(0, num_classes, device=device)
            samples, _, _ = generator(eval_noise, eval_labels)
            save_image(
                samples.data,
                os.path.join(outputs_dir, f"epoch_{epoch}.png"),
                nrow=8,
                normalize=True
            )

    # Save final model weights
    torch.save(generator.state_dict(), os.path.join(checkpoints_dir, "generator.pth"))
    torch.save(discriminator.state_dict(), os.path.join(checkpoints_dir, "discriminator.pth"))
    writer.close()
    logger.info("Attention GAN Training successfully completed!")

if __name__ == "__main__":
    main()
