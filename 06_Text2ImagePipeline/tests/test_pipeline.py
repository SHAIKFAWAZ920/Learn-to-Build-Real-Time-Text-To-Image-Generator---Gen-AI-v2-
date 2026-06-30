import os
import sys
import unittest
from fastapi.testclient import TestClient
from PIL import Image

# Add paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.pipeline import TextToImagePipeline
from api.app import app

class TestPipeline(unittest.TestCase):
    def setUp(self):
        # Configuration parameters
        self.config = {
            "gan_weights": None,
            "sd_model_name": "hf-internal-testing/tiny-stable-diffusion-torch"
        }
        self.pipeline = TextToImagePipeline(mode="attention-gan", config=self.config)
        self.client = TestClient(app)

    def test_pipeline_generation(self):
        prompt = "A shiny red heart symbol"
        img, metadata = self.pipeline.generate(prompt)
        
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (64, 64))
        self.assertEqual(metadata["matched_label"], "heart")
        self.assertGreater(metadata["similarity"], 0.0)

    def test_api_generation_endpoint(self):
        # Request generation
        response = self.client.post(
            "/generate",
            json={
                "prompt": "A blue triangle shape",
                "mode": "attention-gan"
            }
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["media-type"], "image/png")
        self.assertEqual(response.headers["x-model-mode"], "attention-gan")
        self.assertEqual(response.headers["x-matched-label"], "triangle")

    def test_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "running")

if __name__ == "__main__":
    unittest.main()
