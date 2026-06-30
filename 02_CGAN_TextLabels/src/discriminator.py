import torch
import torch.nn as nn

class CGANDiscriminator(nn.Module):
    """
    Conditional Discriminator using Convolutions (DCGAN-like).
    Validates shapes using concatenated class embedding channels.
    """
    def __init__(self, num_classes: int = 8, img_shape: tuple = (1, 64, 64)):
        super().__init__()
        self.img_shape = img_shape
        self.num_classes = num_classes
        
        # Label embedding maps label to image size channel
        self.label_emb = nn.Embedding(num_classes, img_shape[1] * img_shape[2])

        self.model = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=4, stride=2, padding=1),  # input: (img_channels + 1) -> 32
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Flatten(),
            nn.Linear(128 * (img_shape[1] // 8) * (img_shape[2] // 8), 1),
            nn.Sigmoid()
        )

    def forward(self, img: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Embed label and reshape to form a channel layer
        label_embedding = self.label_emb(labels).view(img.size(0), 1, self.img_shape[1], self.img_shape[2])
        # Concatenate label channel to image channels
        d_in = torch.cat((img, label_embedding), dim=1)
        validity = self.model(d_in)
        return validity
