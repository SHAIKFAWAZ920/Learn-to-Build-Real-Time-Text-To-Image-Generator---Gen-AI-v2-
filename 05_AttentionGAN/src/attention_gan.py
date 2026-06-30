import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """
    Self-Attention module for feature maps (SAGAN-style).
    Computes spatial attention weights to capture global context.
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.chanel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        
        # Learnable scale parameter
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B x N x C
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B x C x N
        
        energy = torch.bmm(proj_query, proj_key)  # B x N x N
        attention = F.softmax(energy, dim=-1)  # B x N x N
        
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B x C x N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C x N
        out = out.view(m_batchsize, C, width, height)
        
        out = self.gamma * out + x
        return out, attention


class CrossAttention(nn.Module):
    """
    Cross-Attention module.
    Allows image feature maps (queries) to attend to label conditioning embeddings (keys/values).
    """
    def __init__(self, in_dim: int, emb_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.emb_dim = emb_dim
        
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_linear = nn.Linear(emb_dim, in_dim // 8)
        self.value_linear = nn.Linear(emb_dim, in_dim)
        
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        m_batchsize, C, width, height = x.size()
        
        # Query comes from spatial features: B x N x C_query
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        
        # Key & Value come from class embedding: B x 1 x C_key
        proj_key = self.key_linear(embeddings).unsqueeze(1)  # B x 1 x C_key
        proj_value = self.value_linear(embeddings).unsqueeze(1)  # B x 1 x C_value
        
        # Cross energy
        energy = torch.bmm(proj_query, proj_key.permute(0, 2, 1))  # B x N x 1
        attention = F.softmax(energy, dim=-1)  # B x N x 1
        
        out = torch.bmm(proj_value.permute(0, 2, 1), attention.permute(0, 2, 1))  # B x C x N
        out = out.view(m_batchsize, C, width, height)
        
        out = self.gamma * out + x
        return out, attention


class AttentionGenerator(nn.Module):
    """
    Generator utilizing both Cross-Attention (conditioning) and Self-Attention (spatial structures).
    """
    def __init__(self, latent_dim: int = 100, num_classes: int = 8, embedding_dim: int = 50, img_shape: tuple = (1, 64, 64)):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.num_classes = num_classes
        
        self.label_emb = nn.Embedding(num_classes, embedding_dim)
        self.init_size = img_shape[1] // 8
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, 128 * self.init_size * self.init_size)
        )

        self.block1 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),  # 8x8 -> 16x16
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Cross Attention layer in G
        self.cross_attn = CrossAttention(in_dim=128, emb_dim=embedding_dim)
        
        self.block2 = nn.Sequential(
            nn.Upsample(scale_factor=2),  # 16x16 -> 32x32
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Self Attention layer in G
        self.self_attn = SelfAttention(in_dim=64)
        
        self.block3 = nn.Sequential(
            nn.Upsample(scale_factor=2),  # 32x32 -> 64x64
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, img_shape[0], kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Project noise
        out = self.l1(noise)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        
        out = self.block1(out)
        
        # Embed classes and apply cross attention
        emb = self.label_emb(labels)
        out, cross_maps = self.cross_attn(out, emb)
        
        out = self.block2(out)
        
        # Apply self attention
        out, self_maps = self.self_attn(out)
        
        img = self.block3(out)
        return img, cross_maps, self_maps


class AttentionDiscriminator(nn.Module):
    """
    Discriminator incorporating Self-Attention to capture spatial dependencies.
    """
    def __init__(self, num_classes: int = 8, img_shape: tuple = (1, 64, 64)):
        super().__init__()
        self.img_shape = img_shape
        self.label_emb = nn.Embedding(num_classes, img_shape[1] * img_shape[2])

        self.block1 = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Self Attention in D
        self.self_attn = SelfAttention(in_dim=64)
        
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Flatten(),
            nn.Linear(128 * (img_shape[1] // 8) * (img_shape[2] // 8), 1),
            nn.Sigmoid()
        )

    def forward(self, img: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Concatenate label channel
        label_embedding = self.label_emb(labels).view(img.size(0), 1, self.img_shape[1], self.img_shape[2])
        d_in = torch.cat((img, label_embedding), dim=1)
        
        out = self.block1(d_in)
        out = self.block2(out)
        
        # Self-Attention
        out, self_maps = self.self_attn(out)
        
        validity = self.block3(out)
        return validity, self_maps
