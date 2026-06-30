import torch
import torch.nn as nn

class BaselineGenerator(nn.Module):
    """
    Baseline Unconditional Generator using Multi-Layer Perceptron (MLP).
    Generates 1x64x64 grayscale shape-like images from a random noise vector.
    """
    def __init__(self, latent_dim: int = 100, img_shape: tuple = (1, 64, 64)):
        super().__init__()
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class BaselineDiscriminator(nn.Module):
    """
    Baseline Unconditional Discriminator using Multi-Layer Perceptron (MLP).
    Classifies a 1x64x64 image as real or fake.
    """
    def __init__(self, img_shape: tuple = (1, 64, 64)):
        super().__init__()
        self.img_shape = img_shape
        
        self.model = nn.Sequential(
            nn.Linear(int(torch.prod(torch.tensor(img_shape))), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity
