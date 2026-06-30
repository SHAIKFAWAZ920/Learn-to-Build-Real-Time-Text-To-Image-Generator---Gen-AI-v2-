import os
import sys
import unittest
from fastapi.testclient import TestClient

# Add paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.embedder import EmbeddingGenerator
from api.app import app

class TestEmbeddingSoftware(unittest.TestCase):
    def setUp(self):
        # Configuration mock
        self.config = {
            "bert_model_name": "prajjwal1/bert-tiny",
            "clip_model_name": "hf-internal-testing/tiny-random-clip",
            "sentence_transformer_name": "all-MiniLM-L6-v2"
        }
        self.generator = EmbeddingGenerator(model_type="bert", config=self.config)
        self.client = TestClient(app)

    def test_embedding_output(self):
        texts = ["A simple circle shape", "A yellow star"]
        embeddings = self.generator.embed(texts)
        self.assertEqual(embeddings.shape[0], 2)
        # bert-tiny outputs 128-dimensional embeddings
        self.assertEqual(embeddings.shape[1], 128)

    def test_api_endpoint(self):
        response = self.client.post(
            "/embed",
            json={
                "texts": ["Hello world"],
                "model_type": "bert"
            }
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("embeddings", data)
        self.assertEqual(len(data["embeddings"]), 1)
        self.assertEqual(data["embeddings"][0]["text"], "Hello world")

    def test_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "healthy")

if __name__ == "__main__":
    unittest.main()
