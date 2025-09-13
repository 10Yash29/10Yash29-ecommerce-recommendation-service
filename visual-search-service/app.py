import os
import pickle
import tempfile
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights
from PIL import Image
from io import BytesIO
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables
model = None
transform = None
product_embeddings = {}
product_colors = {}

def load_visual_model():
    """Load and configure the ResNet50 model"""
    global model, transform
    try:
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Identity()
        model.eval()
        transform = ResNet50_Weights.IMAGENET1K_V1.transforms()
        logger.info("✅ Visual model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return False

def load_product_data():
    """Load product embeddings and colors"""
    global product_embeddings, product_colors
    try:
        # Try to load from current directory first
        data_file = "product_data.pkl"
        if not os.path.exists(data_file):
            logger.warning(f"⚠️ {data_file} not found in current directory")
            return False
            
        with open(data_file, "rb") as f:
            data = pickle.load(f)
            
        product_embeddings = data.get("embeddings", {})
        product_colors = data.get("colors", {})
        
        logger.info(f"✅ Loaded {len(product_embeddings)} product embeddings")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load product data: {e}")
        return False

def get_image_embedding(image_bytes: bytes) -> np.ndarray:
    """Extract image embedding using ResNet50"""
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            feat = model(tensor)
        return feat.squeeze(0).cpu().numpy()
    except Exception as e:
        logger.error(f"Error extracting embedding: {e}")
        return None

def find_similar_products(query_embedding, top_k=5):
    """Find most similar products based on embedding"""
    if not product_embeddings or query_embedding is None:
        return []
    
    try:
        # Calculate similarities
        similarities = []
        for product_id, embedding in product_embeddings.items():
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                embedding.reshape(1, -1)
            )[0][0]
            similarities.append((product_id, float(similarity)))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
        
    except Exception as e:
        logger.error(f"Error finding similar products: {e}")
        return []

@app.route("/", methods=["GET"])
def home():
    """Health check endpoint"""
    return jsonify({
        "message": "Visual Search Service",
        "status": "healthy",
        "model_loaded": model is not None,
        "products_loaded": len(product_embeddings),
        "version": "1.0.0"
    }), 200

@app.route("/health", methods=["GET"])
def health_check():
    """Detailed health check"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "transform_loaded": transform is not None,
        "num_products": len(product_embeddings),
        "num_colors": len(product_colors),
        "torch_version": torch.__version__
    }), 200

@app.route("/search", methods=["POST"])
def visual_search():
    """Visual search endpoint"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    if not product_embeddings:
        return jsonify({"error": "No product data available"}), 503
    
    try:
        # Check if image is provided
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({"error": "No image selected"}), 400
        
        # Get number of results
        top_k = int(request.form.get('top_k', 5))
        top_k = min(max(top_k, 1), 20)  # Limit between 1-20
        
        # Process image
        image_bytes = image_file.read()
        query_embedding = get_image_embedding(image_bytes)
        
        if query_embedding is None:
            return jsonify({"error": "Failed to process image"}), 400
        
        # Find similar products
        similar_products = find_similar_products(query_embedding, top_k)
        
        # Format response
        results = []
        for product_id, similarity in similar_products:
            result = {
                "productId": product_id,
                "similarity": round(similarity, 4)
            }
            
            # Add color info if available
            if product_id in product_colors:
                result["dominantColor"] = product_colors[product_id]
            
            results.append(result)
        
        return jsonify({
            "results": results,
            "query_processed": True,
            "total_results": len(results)
        }), 200
        
    except Exception as e:
        logger.error(f"Visual search error: {e}")
        return jsonify({"error": "Search failed"}), 500

@app.route("/search-url", methods=["POST"])
def visual_search_url():
    """Visual search by image URL"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    try:
        data = request.get_json()
        if not data or 'image_url' not in data:
            return jsonify({"error": "No image URL provided"}), 400
        
        image_url = data['image_url']
        top_k = min(max(data.get('top_k', 5), 1), 20)
        
        # Download image
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        
        # Process image
        query_embedding = get_image_embedding(response.content)
        
        if query_embedding is None:
            return jsonify({"error": "Failed to process image"}), 400
        
        # Find similar products
        similar_products = find_similar_products(query_embedding, top_k)
        
        # Format response
        results = []
        for product_id, similarity in similar_products:
            result = {
                "productId": product_id,
                "similarity": round(similarity, 4)
            }
            
            if product_id in product_colors:
                result["dominantColor"] = product_colors[product_id]
            
            results.append(result)
        
        return jsonify({
            "results": results,
            "query_processed": True,
            "total_results": len(results)
        }), 200
        
    except Exception as e:
        logger.error(f"URL search error: {e}")
        return jsonify({"error": "Search failed"}), 500

@app.route("/visual-search", methods=["POST"])
def visual_search_frontend():
    """Visual search endpoint for frontend (matches expected endpoint name)"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    if not product_embeddings:
        return jsonify({"error": "No product data available"}), 503
    
    try:
        # Check if image is provided
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({"error": "No image selected"}), 400
        
        # Get number of results
        top_k = int(request.form.get('top_k', 5))
        top_k = min(max(top_k, 1), 20)  # Limit between 1-20
        
        # Process image
        image_bytes = image_file.read()
        query_embedding = get_image_embedding(image_bytes)
        
        if query_embedding is None:
            return jsonify({"error": "Failed to process image"}), 400
        
        # Find similar products
        similar_products = find_similar_products(query_embedding, top_k)
        
        # Format response for frontend compatibility
        results = []
        for product_id, similarity in similar_products:
            result = {
                "productId": product_id,
                "similarity": round(similarity, 4),
                "visualScore": round(similarity, 4),  # Frontend expects this field
                "colorScore": round(similarity * 0.8, 4)  # Derived from similarity for now
            }
            
            # Add color info if available
            if product_id in product_colors:
                result["dominantColor"] = product_colors[product_id]
                # Adjust color score based on color data availability
                result["colorScore"] = round(similarity * 0.9, 4)
            
            results.append(result)
        
        return jsonify({
            "results": results,
            "query_processed": True,
            "total_results": len(results)
        }), 200
        
    except Exception as e:
        logger.error(f"Visual search error: {e}")
        return jsonify({"error": "Search failed"}), 500

# Initialize on startup
if __name__ == "__main__":
    logger.info("🚀 Starting Visual Search Service...")
    
    # Load model
    model_loaded = load_visual_model()
    
    # Load product data
    data_loaded = load_product_data()
    
    if not model_loaded:
        logger.error("❌ Failed to load model - service may not work properly")
    
    if not data_loaded:
        logger.warning("⚠️ No product data loaded - search will not work")
    
    # Start Flask app
    port = int(os.environ.get("PORT", 7860))  # HF Spaces uses port 7860
    app.run(host="0.0.0.0", port=port, debug=False)
