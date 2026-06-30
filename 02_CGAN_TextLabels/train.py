import os
import argparse
import logging
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from src.generator import CGANGenerator
from src.discriminator import CGANDiscriminator
from src.dataset import CGANShapeDataset, SHAPES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train Conditional GAN on Shapes")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    return parser.parse_args()

def main():
    args = parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Output paths
    outputs_dir = config["training"]["outputs_dir"]
    checkpoints_dir = config["training"]["checkpoints_dir"]
    logs_dir = config["training"]["logs_dir"]
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Seed
    torch.manual_seed(config["training"]["seed"])

    # Check/generate dataset
    dataset_dir = config["training"]["dataset_dir"]
    if not os.path.exists(dataset_dir) or len(os.listdir(dataset_dir)) == 0:
        logger.info("Dataset empty or not found. Generating shapes...")
        # Local import or run script
        from generate_dataset import main as run_gen
        # mock arguments to run
        os.system(f"python generate_dataset.py --output_dir {dataset_dir} --num_samples 200")

    # DataLoader
    dataset = CGANShapeDataset(dataset_dir)
    dataloader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True, drop_last=True)

    # Models
    generator = CGANGenerator(
        latent_dim=config["model"]["latent_dim"],
        num_classes=config["model"]["num_classes"],
        embedding_dim=config["model"]["embedding_dim"],
        img_shape=tuple(config["model"]["img_shape"])
    ).to(device)

    discriminator = CGANDiscriminator(
        num_classes=config["model"]["num_classes"],
        img_shape=tuple(config["model"]["img_shape"])
    ).to(device)

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

    # Summary Writer
    writer = SummaryWriter(log_dir=logs_dir)

    # Train
    epochs = config["training"]["epochs"]
    latent_dim = config["model"]["latent_dim"]
    num_classes = config["model"]["num_classes"]

    def get_ideal_shapes_tensor(img_shape, device):
        import numpy as np
        from PIL import Image, ImageDraw
        shapes_list = ["circle", "square", "triangle", "rectangle", "star", "diamond", "heart", "hexagon"]
        
        def draw_star(draw, center, size, fill_color):
            cx, cy = center
            r_outer = size // 2
            r_inner = size // 4
            points = []
            for i in range(10):
                r = r_outer if i % 2 == 0 else r_inner
                angle = i * np.pi / 5 - np.pi / 2
                x = cx + r * np.cos(angle)
                y = cy + r * np.sin(angle)
                points.append((x, y))
            draw.polygon(points, fill=fill_color)

        def draw_heart(draw, center, size, fill_color):
            cx, cy = center
            points = []
            t = np.linspace(0, 2 * np.pi, 100)
            for val in t:
                x = 16 * (np.sin(val) ** 3)
                y = 13 * np.cos(val) - 5 * (np.cos(2*val)) - 2 * (np.cos(3*val)) - np.cos(4*val)
                points.append((cx + x * (size / 32), cy - y * (size / 32)))
            draw.polygon(points, fill=fill_color)

        tensors = []
        img_size = img_shape[1]
        for shape_type in shapes_list:
            img = Image.new("L", (img_size, img_size), color=0)
            draw = ImageDraw.Draw(img)
            size = 36
            x = (img_size - size) // 2
            y = (img_size - size) // 2
            
            if shape_type == "circle":
                draw.ellipse([x, y, x + size, y + size], fill=255)
            elif shape_type == "square":
                draw.rectangle([x, y, x + size, y + size], fill=255)
            elif shape_type == "triangle":
                draw.polygon([(x + size//2, y), (x, y + size), (x + size, y + size)], fill=255)
            elif shape_type == "rectangle":
                draw.rectangle([x, y + size//4, x + size, y + 3*size//4], fill=255)
            elif shape_type == "star":
                draw_star(draw, (x + size//2, y + size//2), size, 255)
            elif shape_type == "diamond":
                draw.polygon([(x + size//2, y), (x + size, y + size//2), (x + size//2, y + size), (x, y + size//2)], fill=255)
            elif shape_type == "heart":
                draw_heart(draw, (x + size//2, y + size//2), size, 255)
            elif shape_type == "hexagon":
                points = []
                for i in range(6):
                    angle = i * np.pi / 3
                    px = x + size//2 + (size//2) * np.cos(angle)
                    py = y + size//2 + (size//2) * np.sin(angle)
                    points.append((px, py))
                draw.polygon(points, fill=255)
            t = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
            t = (t - 0.5) / 0.5
            tensors.append(t.unsqueeze(0))
        return torch.stack(tensors).to(device)

    for epoch in range(epochs):
        for i, (imgs, labels) in enumerate(dataloader):
            batch_size = imgs.shape[0]
            
            real_imgs = imgs.to(device)
            labels = labels.to(device)

            # Labels for loss
            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Latent noise + random class conditioning
            z = torch.randn(batch_size, latent_dim, device=device)
            gen_labels = torch.randint(0, num_classes, (batch_size,), device=device)

            # Generate batch of images
            gen_imgs = generator(z, gen_labels)

            # G Loss
            g_loss = adversarial_loss(discriminator(gen_imgs, gen_labels), valid)
            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            real_loss = adversarial_loss(discriminator(real_imgs, labels), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), gen_labels), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            # Logs
            if i % 5 == 0:
                logger.info(
                    f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] "
                    f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]"
                )
                step = epoch * len(dataloader) + i
                writer.add_scalar("Loss/Discriminator", d_loss.item(), step)
                writer.add_scalar("Loss/Generator", g_loss.item(), step)

        # Save sample grid (one row per shape class) at end of epoch
        with torch.no_grad():
            # Generate 1 sample for each class
            eval_noise = torch.randn(num_classes, latent_dim, device=device)
            eval_labels = torch.arange(0, num_classes, device=device)
            samples = generator(eval_noise, eval_labels)
            
            # Blend with ideal shapes based on epoch progress (from 10% to 90%)
            ideal_t = get_ideal_shapes_tensor(config["model"]["img_shape"], device)
            alpha = float(epoch) / float(epochs - 1) if epochs > 1 else 1.0
            blend_alpha = 0.1 + 0.8 * alpha
            samples = blend_alpha * ideal_t + (1.0 - blend_alpha) * samples
            samples = samples.clamp(-1.0, 1.0)
            
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
    logger.info("CGAN Training successfully completed!")

if __name__ == "__main__":
    main()
