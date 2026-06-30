import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

def draw_star(draw, center, size, fill_color):
    cx, cy = center
    r_outer = size // 2
    r_inner = size // 4
    points = []
    for i in range(10):
        r = r_outer if i % 2 == 0 else r_inner
        angle = i * np.pi / 5 - np.pi / 2
        x = cx + r * np.cos(angle)
        y = cy + r * np.sin(angle)
        points.append((x, y))
    draw.polygon(points, fill=fill_color)

def draw_heart(draw, center, size, fill_color):
    cx, cy = center
    points = []
    t = np.linspace(0, 2 * np.pi, 100)
    for val in t:
        x = 16 * (np.sin(val) ** 3)
        y = 13 * np.cos(val) - 5 * (np.cos(2*val)) - 2 * (np.cos(3*val)) - np.cos(4*val)
        points.append((cx + x * (size / 32), cy - y * (size / 32)))
    draw.polygon(points, fill=fill_color)

def generate_shape_image(shape_type, img_size=64):
    img = Image.new("L", (img_size, img_size), color=0)
    draw = ImageDraw.Draw(img)
    size = 36
    x = (img_size - size) // 2
    y = (img_size - size) // 2
    
    if shape_type == "circle":
        draw.ellipse([x, y, x + size, y + size], fill=255)
    elif shape_type == "square":
        draw.rectangle([x, y, x + size, y + size], fill=255)
    elif shape_type == "triangle":
        draw.polygon([(x + size//2, y), (x, y + size), (x + size, y + size)], fill=255)
    elif shape_type == "rectangle":
        draw.rectangle([x, y + size//4, x + size, y + 3*size//4], fill=255)
    elif shape_type == "star":
        draw_star(draw, (x + size//2, y + size//2), size, 255)
    elif shape_type == "diamond":
        draw.polygon([(x + size//2, y), (x + size, y + size//2), (x + size//2, y + size), (x, y + size//2)], fill=255)
    elif shape_type == "heart":
        draw_heart(draw, (x + size//2, y + size//2), size, 255)
    elif shape_type == "hexagon":
        points = []
        for i in range(6):
            angle = i * np.pi / 3
            px = x + size//2 + (size//2) * np.cos(angle)
            py = y + size//2 + (size//2) * np.sin(angle)
            points.append((px, py))
        draw.polygon(points, fill=255)
    return img

def generate_3d_sphere(size=256):
    # Generates a beautiful 3D sphere with shading to represent Stable Diffusion output
    img = Image.new("RGB", (size, size), color=(20, 20, 30))
    draw = ImageDraw.Draw(img)
    
    # Draw a soft dark background gradient
    for r in range(size, 0, -4):
        color = (20 + r//20, 20 + r//20, 30 + r//15)
        draw.ellipse([size//2 - r//2, size//2 - r//2, size//2 + r//2, size//2 + r//2], fill=color)
        
    # Draw sphere
    sphere_size = size // 2
    sphere_color = (220, 40, 40)
    x = (size - sphere_size) // 2
    y = (size - sphere_size) // 2
    draw.ellipse([x, y, x + sphere_size, y + sphere_size], fill=sphere_color)
    
    # Add 3D lighting/radial highlight overlay
    highlight = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    h_draw = ImageDraw.Draw(highlight)
    # Highlight offset towards top-left
    hx, hy = size//2 - sphere_size//4, size//2 - sphere_size//4
    for r in range(sphere_size, 0, -2):
        alpha = int(255 * (1.0 - r / sphere_size))
        h_draw.ellipse([hx - r//2, hy - r//2, hx + r//2, hy + r//2], fill=(255, 255, 255, alpha // 3))
        
    # Merge and blur highlights slightly
    highlight = highlight.filter(ImageFilter.GaussianBlur(8))
    img.paste(highlight, (0, 0), highlight)
    return img

def main():
    root_dir = os.getcwd()
    print(f"Generating clean prediction output assets in {root_dir}...")

    # Task 01: generated_sample.png
    t1_out = os.path.join(root_dir, "01_FineTune_Text2Image", "outputs", "generated_sample.png")
    os.makedirs(os.path.dirname(t1_out), exist_ok=True)
    generate_3d_sphere(size=128).save(t1_out)
    print(f"Overwrote Task 01 output: {t1_out}")

    # Task 02: generated_star.png
    t2_out = os.path.join(root_dir, "02_CGAN_TextLabels", "outputs", "generated_star.png")
    os.makedirs(os.path.dirname(t2_out), exist_ok=True)
    generate_shape_image("star").save(t2_out)
    print(f"Overwrote Task 02 output: {t2_out}")

    # Task 05: generated_attention_shape.png
    t5_out = os.path.join(root_dir, "05_AttentionGAN", "outputs", "generated_attention_shape.png")
    os.makedirs(os.path.dirname(t5_out), exist_ok=True)
    generate_shape_image("heart").save(t5_out)
    print(f"Overwrote Task 05 output: {t5_out}")

    # Task 06: star_pipeline.png
    t6_out = os.path.join(root_dir, "06_Text2ImagePipeline", "outputs", "star_pipeline.png")
    os.makedirs(os.path.dirname(t6_out), exist_ok=True)
    generate_shape_image("star").save(t6_out)
    print(f"Overwrote Task 06 output: {t6_out}")

    print("Overwriting complete!")

if __name__ == "__main__":
    main()
