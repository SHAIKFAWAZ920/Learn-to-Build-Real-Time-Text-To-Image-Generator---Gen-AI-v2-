import unittest
import torch
from src.attention_gan import SelfAttention, CrossAttention, AttentionGenerator, AttentionDiscriminator

class TestAttentionGAN(unittest.TestCase):
    def setUp(self):
        self.latent_dim = 100
        self.num_classes = 8
        self.embedding_dim = 50
        self.img_shape = (1, 64, 64)

    def test_self_attention_shape(self):
        # 128 channels, 16x16 grid
        x = torch.randn(2, 128, 16, 16)
        layer = SelfAttention(in_dim=128)
        out, att_map = layer(x)
        
        self.assertEqual(out.shape, (2, 128, 16, 16))
        self.assertEqual(att_map.shape, (2, 256, 256))  # 16*16 = 256

    def test_cross_attention_shape(self):
        x = torch.randn(2, 128, 16, 16)
        embeddings = torch.randn(2, self.embedding_dim)
        layer = CrossAttention(in_dim=128, emb_dim=self.embedding_dim)
        out, att_map = layer(x, embeddings)
        
        self.assertEqual(out.shape, (2, 128, 16, 16))
        self.assertEqual(att_map.shape, (2, 256, 1))

    def test_attention_generator_output(self):
        generator = AttentionGenerator(
            latent_dim=self.latent_dim,
            num_classes=self.num_classes,
            embedding_dim=self.embedding_dim,
            img_shape=self.img_shape
        )
        
        z = torch.randn(4, self.latent_dim)
        labels = torch.randint(0, self.num_classes, (4,))
        img, cross_maps, self_maps = generator(z, labels)
        
        self.assertEqual(img.shape, (4, 1, 64, 64))
        self.assertEqual(cross_maps.shape, (4, 256, 1))
        self.assertEqual(self_maps.shape, (4, 1024, 1024))  # 32*32 = 1024

if __name__ == "__main__":
    unittest.main()
