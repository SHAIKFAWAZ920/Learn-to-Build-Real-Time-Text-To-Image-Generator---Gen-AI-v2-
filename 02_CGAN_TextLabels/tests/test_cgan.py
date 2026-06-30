import unittest
import torch
from src.generator import CGANGenerator
from src.discriminator import CGANDiscriminator

class TestCGAN(unittest.TestCase):
    def setUp(self):
        self.latent_dim = 100
        self.num_classes = 8
        self.embedding_dim = 50
        self.img_shape = (1, 64, 64)
        
        self.generator = CGANGenerator(
            latent_dim=self.latent_dim,
            num_classes=self.num_classes,
            embedding_dim=self.embedding_dim,
            img_shape=self.img_shape
        )
        
        self.discriminator = CGANDiscriminator(
            num_classes=self.num_classes,
            img_shape=self.img_shape
        )

    def test_generator_forward(self):
        batch_size = 4
        z = torch.randn(batch_size, self.latent_dim)
        labels = torch.randint(0, self.num_classes, (batch_size,))
        
        img = self.generator(z, labels)
        self.assertEqual(img.shape, (batch_size, 1, 64, 64))
        self.assertTrue(torch.all(img >= -1.0) and torch.all(img <= 1.0))

    def test_discriminator_forward(self):
        batch_size = 4
        img = torch.randn(batch_size, 1, 64, 64)
        labels = torch.randint(0, self.num_classes, (batch_size,))
        
        validity = self.discriminator(img, labels)
        self.assertEqual(validity.shape, (batch_size, 1))
        self.assertTrue(torch.all(validity >= 0.0) and torch.all(validity <= 1.0))

if __name__ == "__main__":
    unittest.main()
