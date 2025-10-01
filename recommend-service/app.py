from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)


model_data = None

def load_model():
    global model_data
    try:
        model_data = joblib.load('trained_model.pkl')
        print("✅ Model loaded successfully!")
    except FileNotFoundError:
        print("❌ Model file not found. Please train the model first.")
        model_data = None

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "model_loaded": model_data is not None,
        "version": "1.0.0"
    })

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    if model_data is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()
        user_id = data.get('userId')
        num_recommendations = data.get('numRecommendations', 5)
        
        if not user_id:
            return jsonify({"error": "userId is required"}), 400
        
        recommendations = generate_recommendations(user_id, num_recommendations)
        
        return jsonify({
            "userId": user_id,
            "recommendations": recommendations
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def generate_recommendations(user_id, num_recommendations=5):
    user_item_matrix = model_data['user_item_matrix']
    item_similarity = model_data['item_similarity']
    

    if user_id not in user_item_matrix.index:

        user_interactions = user_item_matrix.sum(axis=0)
        popular_items = user_interactions.nlargest(num_recommendations).index.tolist()
        return [{"productId": item, "score": float(user_interactions[item])} for item in popular_items]
    
  
    user_vector = user_item_matrix.loc[user_id]
    user_interacted_items = user_vector[user_vector > 0].index
    

    scores = {}
    for item in user_item_matrix.columns:
        if item not in user_interacted_items:
          
            item_idx = list(user_item_matrix.columns).index(item)
            score = 0
            for interacted_item in user_interacted_items:
                interacted_idx = list(user_item_matrix.columns).index(interacted_item)
                similarity = item_similarity[item_idx][interacted_idx]
                interaction_strength = user_vector[interacted_item]
                score += similarity * interaction_strength
            scores[item] = score
    

    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    recommendations = [
        {"productId": item, "score": float(score)} 
        for item, score in sorted_items[:num_recommendations]
    ]
    
    return recommendations

if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5001)
