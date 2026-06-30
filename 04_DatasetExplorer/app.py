import os
import yaml
import gradio as gr

from src.analyzer import DatasetAnalyzer
from src.plotter import generate_plots
from src.reporter import DatasetReporter

# Default configuration loading
config_path = "configs/config.yaml"
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
else:
    config = {"dataset": {"path": "dataset", "format": "custom"}, "outputs": {"reports_dir": "reports", "plots_dir": "plots"}}

def run_gui_analysis(dataset_dir, dataset_format):
    reports_dir = config["outputs"]["reports_dir"]
    plots_dir = config["outputs"]["plots_dir"]
    
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Run pipeline
    analyzer = DatasetAnalyzer(dataset_path=dataset_dir, dataset_format=dataset_format)
    metrics = analyzer.analyze()
    
    if metrics.get("status") == "empty":
        return "No images found in target directory.", "", "", "", None, None, None, None
        
    generate_plots(metrics, output_dir=plots_dir)
    
    reporter = DatasetReporter(metrics, output_dir=reports_dir)
    reporter.export_csv()
    
    rel_plots_dir = os.path.relpath(plots_dir, start=reports_dir)
    reporter.export_html(plots_dir=rel_plots_dir)
    reporter.export_pdf(plots_dir=plots_dir)
    
    # Text statistics output
    summary_text = (
        f"✅ Dataset Path: {dataset_dir}\n"
        f"📊 Total Images Analysed: {metrics['total_images']}\n"
        f"📏 Mean Resolution: {int(metrics['mean_width'])}x{int(metrics['mean_height'])}\n"
        f"📚 Vocabulary Size: {metrics['vocab_size']}\n"
        f"📝 Mean Caption Length: {metrics['mean_caption_length']:.2f} words"
    )
    
    # Class distribution preview
    class_dist_str = ""
    for cls, count in metrics["class_distribution"].items():
        class_dist_str += f"- {cls}: {count} images\n"
    if not class_dist_str:
        class_dist_str = "No class-conditional directories detected."
        
    # Top word frequencies preview
    freq_str = ""
    for word, count in metrics["most_common_words"][:10]:
        freq_str += f"- {word}: {count} occurrences\n"
        
    # Paths to generated plots
    word_plot = os.path.join(plots_dir, "word_frequencies.png")
    len_plot = os.path.join(plots_dir, "caption_lengths.png")
    aspect_plot = os.path.join(plots_dir, "aspect_ratios.png")
    
    # Download files
    pdf_file = os.path.join(reports_dir, "report.pdf")
    html_file = os.path.join(reports_dir, "report.html")
    csv_file = os.path.join(reports_dir, "word_frequencies.csv")
    
    # Return outputs
    return (
        summary_text,
        class_dist_str,
        freq_str,
        word_plot if os.path.exists(word_plot) else None,
        len_plot if os.path.exists(len_plot) else None,
        aspect_plot if os.path.exists(aspect_plot) else None,
        pdf_file if os.path.exists(pdf_file) else None,
        html_file if os.path.exists(html_file) else None,
        csv_file if os.path.exists(csv_file) else None,
    )

# Interface
with gr.Blocks(title="Dataset Explorer & Analyzer") as demo:
    gr.Markdown("# 🔍 Dataset Explorer & Analyzer")
    gr.Markdown("Profile your image-caption datasets, visualize resolutions and vocabulary distributions, and generate printable PDF summaries.")

    with gr.Row():
        with gr.Column(scale=1):
            dataset_path_in = gr.Textbox(label="Dataset Directory Path", value=config["dataset"]["path"])
            format_in = gr.Dropdown(
                label="Dataset Format Schema",
                choices=["custom", "coco", "oxford"],
                value=config["dataset"]["format"]
            )
            btn = gr.Button("Analyze Dataset", variant="primary")
            
            gr.Markdown("### 📥 Download Reports")
            download_pdf = gr.File(label="Download PDF Report")
            download_html = gr.File(label="Download HTML Report")
            download_csv = gr.File(label="Download CSV Word Frequency List")
            
        with gr.Column(scale=2):
            with gr.Tab("Summary Statistics"):
                stat_output = gr.Textbox(label="General Statistics Summary", max_lines=6)
                class_output = gr.Textbox(label="Class Distribution Breakdown")
                freq_output = gr.Textbox(label="Top 10 High Frequency Words")
                
            with gr.Tab("Resolution & Words Plots"):
                with gr.Row():
                    plot_words = gr.Image(label="Word Frequency Chart")
                    plot_lens = gr.Image(label="Caption Lengths Histogram")
                with gr.Row():
                    plot_aspect = gr.Image(label="Aspect Ratio Distribution")

    btn.click(
        run_gui_analysis,
        inputs=[dataset_path_in, format_in],
        outputs=[
            stat_output,
            class_output,
            freq_output,
            plot_words,
            plot_lens,
            plot_aspect,
            download_pdf,
            download_html,
            download_csv
        ]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861)
