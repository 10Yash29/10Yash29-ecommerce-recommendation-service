import os
import pickle
import pandas as pd
from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from scipy.sparse import csr_matrix
from config import Config  # New import
from model_utils import ModelUtils  # New import

# Load environment variables
load_dotenv()
app = Flask(__name__)
CORS(app)

# Global variables for model and utilities
model = None
user_to_idx = None
idx_to_product = None
model_utils = None

try:
    with open(Config.MODEL_PATH, "rb") as f:  # Updated to use Config.MODEL_PATH
        saved_data = pickle.load(f)
        model = saved_data["model"]
        user_to_idx = saved_data["user_to_idx"]
        idx_to_product = saved_data["idx_to_product"]
        
        # Load data for model utilities if available
        if os.path.exists(Config.DATA_PATH):  # New logic to load data
            df = pd.read_csv(Config.DATA_PATH)
            idx_to_user = {v: k for k, v in user_to_idx.items()}
            product_to_idx = {v: k for k, v in idx_to_product.items()}
            model_utils = ModelUtils(df, user_to_idx, idx_to_user, product_to_idx, idx_to_product)
            print("✅ Model utilities loaded.")
            
    print("✅ ALS model loaded.")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit(1)

@app.route("/recommend/<user_id>", methods=["GET"])
def recommend_products(user_id):
    if user_id not in user_to_idx:
        # Handle cold start with popular products
        if model_utils:  # New cold start handling
            cold_start_recommendations = model_utils.get_cold_start_recommendations(Config.DEFAULT_RECOMMENDATIONS)
            return jsonify({
                "userId": user_id,
                "recommendedProducts": cold_start_recommendations,
                "message": "User not found in training data. Showing popular products."
            }), 200
        else:
            return jsonify({
                "userId": user_id,
                "recommendedProducts": [],
                "message": "User not found in training data (cold start)."
            }), 404

    user_idx = user_to_idx[user_id]
    try:
        # Fix: Use actual user interactions instead of empty matrix
        user_interactions = model_utils.get_user_interactions(user_idx)
        recommended_items, _ = model.recommend(
            user_idx,
            user_items=user_interactions,  # Use actual user data
            N=Config.DEFAULT_RECOMMENDATIONS
        )
        recommended_product_ids = [idx_to_product[i] for i in recommended_items]
        return jsonify({
            "userId": user_id,
            "recommendedProducts": recommended_product_ids
        }), 200
    except Exception as e:
        print(f"❌ Recommendation error: {e}")
        return jsonify({"error": "Recommendation failed"}), 500

@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": f"Welcome to Recommendation Service {Config.API_VERSION}"}), 200  # Updated message

@app.route("/popular", methods=["GET"])  # New endpoint
def get_popular_products():
    """Get popular products for cold start scenarios"""
    if model_utils:
        popular_products = model_utils.get_cold_start_recommendations(Config.POPULAR_PRODUCTS_COUNT)
        return jsonify({
            "popularProducts": popular_products,
            "count": len(popular_products)
        }), 200
    else:
        return jsonify({"error": "Popular products not available"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_utils_loaded": model_utils is not None,
        "num_users": len(user_to_idx) if user_to_idx else 0,
        "num_products": len(idx_to_product) if idx_to_product else 0,
        "version": Config.API_VERSION
    }), 200

if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", 5050))
    
    print(f"🚀 Running on {host}:{port} (debug={Config.DEBUG})")  # Updated debug message
    app.run(host=host, port=port, debug=Config.DEBUG)  # Updated to use Config.DEBUG

