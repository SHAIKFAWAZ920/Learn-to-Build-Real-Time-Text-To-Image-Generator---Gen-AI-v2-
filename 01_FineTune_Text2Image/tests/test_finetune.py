import os
import unittest
import torch
from transformers import AutoTokenizer
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import LoRAAttnProcessor

from src.dataset import DreamBoothDataset

class TestFinetune(unittest.TestCase):
    def setUp(self):
        # Local mock setup
        self.dataset_dir = "test_dataset"
        self.instance_prompt = "a photo of a shape"
        self.model_path = "hf-internal-testing/tiny-stable-diffusion-torch"
        os.makedirs(self.dataset_dir, exist_ok=True)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, subfolder="tokenizer")

    def test_dreambooth_dataset(self):
        # Create dataset
        dataset = DreamBoothDataset(
            instance_data_dir=self.dataset_dir,
            instance_prompt=self.instance_prompt,
            tokenizer=self.tokenizer,
            size=64,
        )
        
        # Check size
        self.assertGreater(len(dataset), 0)
        
        # Retrieve item
        item = dataset[0]
        self.assertIn("instance_images", item)
        self.assertIn("instance_prompt_ids", item)
        
        # Check shapes
        self.assertEqual(item["instance_images"].shape, (3, 64, 64))
        self.assertEqual(item["instance_prompt_ids"].dtype, torch.int64)

    def test_lora_injection(self):
        # Load small unet
        unet = UNet2DConditionModel.from_pretrained(self.model_path, subfolder="unet")
        
        # Inject LoRA
        lora_attn_procs = {}
        for name, attn_processor in unet.attn_processors.items():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name.split(".")[1])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name.split(".")[1])
                hidden_size = unet.config.block_out_channels[block_id]
            
            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=4,
            )
        unet.set_attn_processor(lora_attn_procs)
        
        # Verify unet has LoRA layer in the parameters
        lora_params = [p for p in unet.parameters() if p.requires_grad]
        self.assertGreater(len(lora_params), 0)

    def tearDown(self):
        # Cleanup test folder
        if os.path.exists(self.dataset_dir):
            for file in os.listdir(self.dataset_dir):
                os.remove(os.path.join(self.dataset_dir, file))
            os.rmdir(self.dataset_dir)

if __name__ == "__main__":
    unittest.main()
