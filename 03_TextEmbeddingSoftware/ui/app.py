import os
import sys
import yaml
import numpy as np
import gradio as gr

# Add src path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.embedder import EmbeddingGenerator

# Load config
config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Cache generators
generators = {}

def get_generator(model_type):
    if model_type not in generators:
        generators[model_type] = EmbeddingGenerator(model_type=model_type, config=config["model"])
    return generators[model_type]

def handle_embed(text, model_type):
    if not text.strip():
        return "Please enter some text.", "", ""
    try:
        gen = get_generator(model_type)
        embeds = gen.embed([text])
        embedding = embeds[0]
        
        # Format display
        dim_str = f"Embedding Dimension: {embedding.shape[0]}"
        preview_str = f"Vector Preview (first 10 components):\n{embedding[:10].tolist()}..."
        
        # Save temporary formats
        os.makedirs("temp_exports", exist_ok=True)
        npy_path = "temp_exports/embedding.npy"
        np.save(npy_path, embedding)
        
        return dim_str, preview_str, npy_path
    except Exception as e:
        return f"Error: {str(e)}", "", None

def calculate_similarity(text1, text2, model_type):
    if not text1.strip() or not text2.strip():
        return "Please enter text in both fields."
    try:
        gen = get_generator(model_type)
        embeds = gen.embed([text1, text2])
        v1, v2 = embeds[0], embeds[1]
        
        # Cosine similarity
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        similarity = dot_product / (norm_v1 * norm_v2)
        
        return f"Cosine Similarity: {similarity:.4f}"
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio Interface
with gr.Blocks(title="Text Embedding Suite") as demo:
    gr.Markdown("# 🚀 Text Embedding Suite")
    gr.Markdown("Convert text descriptions into high-dimensional embeddings for downstream Text-to-Image generation.")
    
    with gr.Tab("Single Text Embedding"):
        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(label="Input Text", value="A majestic red dragon flying over mountains")
                model_sel = gr.Dropdown(
                    label="Embedding Model", 
                    choices=["sentence-transformer", "bert", "clip"], 
                    value="sentence-transformer"
                )
                btn = gr.Button("Generate Embedding", variant="primary")
            with gr.Column():
                out_dim = gr.Textbox(label="Dimension Info")
                out_preview = gr.Textbox(label="Vector Preview", max_lines=5)
                out_file = gr.File(label="Download NumPy Vector (.npy)")
                
        btn.click(handle_embed, inputs=[input_text, model_sel], outputs=[out_dim, out_preview, out_file])

    with gr.Tab("Semantic Similarity Analysis"):
        with gr.Row():
            with gr.Column():
                t1 = gr.Textbox(label="Text Sentence A", value="A photo of a circle")
                t2 = gr.Textbox(label="Text Sentence B", value="A drawing of a round shape")
                model_sel_sim = gr.Dropdown(
                    label="Model", 
                    choices=["sentence-transformer", "bert", "clip"], 
                    value="sentence-transformer"
                )
                sim_btn = gr.Button("Calculate Match Score", variant="secondary")
            with gr.Column():
                sim_out = gr.Label(label="Similarity Score")
                
        sim_btn.click(calculate_similarity, inputs=[t1, t2, model_sel_sim], outputs=[sim_out])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
