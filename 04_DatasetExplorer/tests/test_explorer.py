import os
import unittest
import shutil

from src.analyzer import DatasetAnalyzer

class TestDatasetExplorer(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_explore_dataset"
        os.makedirs(self.test_dir, exist_ok=True)
        self.analyzer = DatasetAnalyzer(dataset_path=self.test_dir, dataset_format="custom")

    def test_analyzer_statistics(self):
        # Trigger toy generation inside analyze
        metrics = self.analyzer.analyze()
        
        self.assertEqual(metrics["total_images"], 12)
        self.assertEqual(metrics["vocab_size"], 9)  # Unique tokens in dummy caption
        self.assertEqual(metrics["mean_width"], 64)
        self.assertEqual(metrics["mean_height"], 64)
        self.assertGreater(metrics["mean_caption_length"], 0)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

if __name__ == "__main__":
    unittest.main()
