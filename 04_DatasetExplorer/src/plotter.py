import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def generate_plots(metrics: dict, output_dir: str = "plots"):
    """
    Renders and saves statistical charts (histograms, bar plots)
    based on the analyzed dataset metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # 1. Caption length distribution histogram
    if "caption_lengths" in metrics and len(metrics["caption_lengths"]) > 0:
        plt.figure(figsize=(8, 4))
        sns.histplot(metrics["caption_lengths"], bins=10, kde=True, color="skyblue")
        plt.title("Caption Length Distribution (Words)")
        plt.xlabel("Length (words)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "caption_lengths.png"))
        plt.close()

    # 2. Word Frequency bar chart
    if "most_common_words" in metrics and len(metrics["most_common_words"]) > 0:
        words, counts = zip(*metrics["most_common_words"][:15])
        plt.figure(figsize=(10, 5))
        sns.barplot(x=list(counts), y=list(words), palette="viridis")
        plt.title("Top 15 Most Common Words in Captions")
        plt.xlabel("Frequency")
        plt.ylabel("Word")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "word_frequencies.png"))
        plt.close()

    # 3. Aspect Ratio distribution
    if "aspect_ratios" in metrics and len(metrics["aspect_ratios"]) > 0:
        plt.figure(figsize=(8, 4))
        sns.histplot(metrics["aspect_ratios"], bins=10, kde=True, color="salmon")
        plt.title("Aspect Ratio Distribution")
        plt.xlabel("Aspect Ratio (W/H)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "aspect_ratios.png"))
        plt.close()

    # 4. Class distribution (if applicable)
    if "class_distribution" in metrics and len(metrics["class_distribution"]) > 0:
        classes = list(metrics["class_distribution"].keys())
        counts = list(metrics["class_distribution"].values())
        plt.figure(figsize=(8, 4))
        sns.barplot(x=classes, y=counts, palette="rocket")
        plt.title("Class Distribution")
        plt.xlabel("Classes")
        plt.ylabel("Images")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "class_distribution.png"))
        plt.close()
