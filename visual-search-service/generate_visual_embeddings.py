#!/usr/bin/env python3
import os
from dotenv import load_dotenv
import psycopg2
import requests
from io import BytesIO
from PIL import Image
import pillow_avif
import numpy as np
import pickle
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights
from colorthief import ColorThief

# ──────────────────────────────────────────────────────────────────────────────
# Load environment
load_dotenv()
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set in environment")
print(f"DATABASE_URL = {DATABASE_URL}")
# ──────────────────────────────────────────────────────────────────────────────

def fetch_product_images():
    """Return dict of {product_id: [image_url, …]}"""
    conn = psycopg2.connect(DATABASE_URL)
    cur  = conn.cursor()
    cur.execute("""
        SELECT "Product".id, "Image".url
          FROM "Product"
          JOIN "Image" ON "Image"."productId" = "Product".id
         ORDER BY "Product".id;
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    images = {}
    for pid, url in rows:
        images.setdefault(pid, []).append(url)
    return images

def load_model():
    """Load and configure the ResNet50 model"""
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    model.eval()
    
    transform = ResNet50_Weights.IMAGENET1K_V1.transforms()
    return model, transform

def get_image_embedding(image_bytes: bytes, model, transform) -> np.ndarray:
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        feat = model(tensor)
    return feat.squeeze(0).cpu().numpy()

def get_dominant_color(image_bytes: bytes) -> tuple[int,int,int]:
    try:
        return ColorThief(BytesIO(image_bytes)).get_color(quality=1)
    except Exception as e:
        print(f"⚠️ Color extraction failed: {repr(e)}")
        return (0, 0, 0)

def generate_embeddings(product_images: dict[str, list[str]]) -> dict:
    model, transform = load_model()
    data = {"embeddings": {}, "colors": {}}
    skipped = 0

    for pid, urls in product_images.items():
        url = urls[0]
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            img_bytes = resp.content

            emb   = get_image_embedding(img_bytes, model, transform)
            color = get_dominant_color(img_bytes)

            data["embeddings"][pid] = emb
            data["colors"][pid]     = color

        except Exception as e:
            skipped += 1
            print(f"❌ Skipped {pid} ({url}): {repr(e)}")

    print(f"✅ Processed {len(data['embeddings'])} products")
    print(f"❌ Skipped   {skipped} products")
    return data

if __name__ == "__main__":
    print("Fetching product images…")
    imgs = fetch_product_images()
    print(f"Found {len(imgs)} products")

    print("Generating embeddings & colors…")
    pd = generate_embeddings(imgs)

    print(f"About to save {len(pd['embeddings'])} embeddings...")
    with open("product_data.pkl", "wb") as f:
        pickle.dump(pd, f)
    print("✅ Saved embeddings and colors to product_data.pkl")
    
    # Verify file was written
    import os
    size = os.path.getsize("product_data.pkl")
    print(f"📁 File size: {size} bytes")
