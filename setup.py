from setuptools import setup, find_packages

setup(
    name="internship_text_to_image",
    version="1.0.0",
    description="Advanced AI/ML Engineering Internship Workspace for Text-to-Image Generation",
    author="Intern",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "diffusers>=0.20.0",
        "accelerate>=0.20.0",
        "sentence-transformers>=2.2.0",
        "fastapi>=0.95.0",
        "uvicorn[standard]>=0.22.0",
        "gradio>=3.35.0",
        "reportlab>=3.6.0",
        "jinja2>=3.1.0",
        "pyyaml>=6.0",
        "requests>=2.28.0",
        "tqdm>=4.65.0",
        "pandas>=1.4.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
    ],
)
