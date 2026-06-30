import torch
import torch.nn as nn

class CGANGenerator(nn.Module):
    """
    Conditional Generator using Transposed Convolutions (DCGAN-like).
    Combines latent noise vector and class embeddings to generate shape images.
    """
    def __init__(self, latent_dim: int = 100, num_classes: int = 8, embedding_dim: int = 50, img_shape: tuple = (1, 64, 64)):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.img_shape = img_shape
        
        # Label embedding
        self.label_emb = nn.Embedding(num_classes, embedding_dim)
        
        # Generator structure starting from 8x8 spatial resolution
        self.init_size = img_shape[1] // 8
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim + embedding_dim, 128 * self.init_size * self.init_size)
        )

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),  # 8x8 -> 16x16
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2),  # 16x16 -> 32x32
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2),  # 32x32 -> 64x64
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32, img_shape[0], kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Embed and concatenate conditioning
        label_embedding = self.label_emb(labels)
        gen_input = torch.cat((noise, label_embedding), dim=1)
        
        out = self.l1(gen_input)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img
