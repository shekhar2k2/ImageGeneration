import os
import cv2
import svgwrite
import numpy as np
from sklearn.cluster import KMeans
from diffusers import StableDiffusionPipeline
import torch


PROMPT = input("Enter your prompt: ")
OUT_DIR = "out"
RASTER_FN = os.path.join(OUT_DIR, "out.png")
SVG_DIR = os.path.join(OUT_DIR, "svg_variants")
IMAGE_SIZE = (512, 512)

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SVG_DIR, exist_ok=True)

print("Generating raster image from prompt...")
model_id = "runwayml/stable-diffusion-v1-5"
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else None
)
pipe = pipe.to(device)

img = pipe(PROMPT, height=IMAGE_SIZE[1], width=IMAGE_SIZE[0]).images[0]
img.save(RASTER_FN)
print("Saved raster to", RASTER_FN)


img = cv2.imread(RASTER_FN)
h, w = img.shape[:2]
pixels = img.reshape(-1, 3)


color_levels = [4, 8, 16]

for NUM_COLORS in color_levels:
    print(f"Generating {NUM_COLORS}-color SVG...")


    kmeans = KMeans(n_clusters=NUM_COLORS, n_init=10, random_state=42).fit(pixels)
    labels = kmeans.labels_.reshape(h, w)
    palette = kmeans.cluster_centers_.astype(np.uint8)


    svg_path = os.path.join(SVG_DIR, f"vector_{NUM_COLORS}colors.svg")
    dwg = svgwrite.Drawing(svg_path, size=(w, h))

    for i, color in enumerate(palette):
        mask = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        hex_color = f"rgb({color[2]},{color[1]},{color[0]})"  # BGR → RGB

        for cnt in contours:
            if len(cnt) > 2:
                points = [(int(p[0][0]), int(p[0][1])) for p in cnt]
                dwg.add(dwg.polygon(points, fill=hex_color, stroke=hex_color))

    dwg.save()
    print(f"Saved {NUM_COLORS}-color SVG as {svg_path}")

print("Done! Check the 'out/svg_variants' folder for SVG results.")
