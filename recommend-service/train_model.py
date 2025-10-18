import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

def train_recommendation_model():
    try:
        print("Loading data from aggregated_interactions.csv...")
        df = pd.read_csv('aggregated_interactions.csv')
        print(f"Loaded {len(df)} interaction records")
        
        print("Creating user-item matrix...")
        user_item_matrix = df.pivot_table(
            index='userId', 
            columns='productId', 
            values='strength', 
            fill_value=0
        )
        
        print(f"Matrix shape: {user_item_matrix.shape}")

        print("Training SVD model...")
        svd = TruncatedSVD(n_components=min(50, min(user_item_matrix.shape) - 1), random_state=42)
        user_features = svd.fit_transform(user_item_matrix)
        
        print("Calculating item similarities...")
        item_features = svd.components_.T
        item_similarity = cosine_similarity(item_features)

        model_data = {
            'svd': svd,
            'user_item_matrix': user_item_matrix,
            'item_similarity': item_similarity,
            'user_features': user_features,
            'item_features': item_features
        }
        
        joblib.dump(model_data, 'trained_model.pkl')
        print("✅ Model trained and saved successfully!")
        
        return model_data
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    train_recommendation_model()
