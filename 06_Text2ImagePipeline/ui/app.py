import os
import sys
import yaml
import gradio as gr

# Add path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.pipeline import TextToImagePipeline

# Load config
config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Pipeline cache
pipelines = {}

def get_pipeline(mode):
    if mode not in pipelines:
        pipelines[mode] = TextToImagePipeline(mode=mode, config=config["pipeline"])
    return pipelines[mode]

def run_pipeline(prompt, mode):
    if not prompt.strip():
        return None, "Please enter a valid text prompt."
        
    try:
        pipe = get_pipeline(mode)
        img, metadata = pipe.generate(prompt)
        
        # Format response
        info_str = (
            f"✅ Generation complete using '{mode}' mode.\n"
            f"📝 Raw Prompt: '{metadata['prompt']}'\n"
        )
        if "matched_label" in metadata:
            info_str += (
                f"🏷️ Matched Category: {metadata['matched_label']}\n"
                f"📏 Cosine Similarity: {metadata['similarity']:.4f}"
            )
            
        return img, info_str
    except Exception as e:
        return None, f"Error occurred: {str(e)}"

# Gradio Interface
with gr.Blocks(title="Unified Text-to-Image Pipeline") as demo:
    gr.Markdown("# 🎨 Unified Text-to-Image Generation Workspace")
    gr.Markdown("An end-to-end interface wrapping custom embedding classifiers, attention-guided GAN shapes, and Stable Diffusion models.")

    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="Prompt Text Input", 
                value="A brilliant yellow star in the night sky"
            )
            mode_input = gr.Radio(
                choices=["attention-gan", "stable-diffusion"], 
                label="Pipeline Generator Model", 
                value="attention-gan"
            )
            btn = gr.Button("Generate Image", variant="primary")
        with gr.Column():
            output_image = gr.Image(label="Output Image")
            output_info = gr.Textbox(label="Metadata Details", max_lines=6)

    btn.click(run_pipeline, inputs=[prompt_input, mode_input], outputs=[output_image, output_info])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7862)
