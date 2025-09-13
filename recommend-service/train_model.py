import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
import pickle
from config import Config
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_preprocess_data():
    """Load and preprocess the interaction data"""
    try:
        logger.info(f"Loading data from {Config.DATA_PATH}")
        df = pd.read_csv(Config.DATA_PATH)
        
        # Filter data based on minimum interactions
        user_counts = df['userId'].value_counts()
        product_counts = df['productId'].value_counts()
        
        valid_users = user_counts[user_counts >= Config.MIN_USER_INTERACTIONS].index
        valid_products = product_counts[product_counts >= Config.MIN_PRODUCT_INTERACTIONS].index
        
        df_filtered = df[df['userId'].isin(valid_users) & df['productId'].isin(valid_products)]
        
        logger.info(f"Filtered data: {len(df_filtered)} interactions from {df_filtered['userId'].nunique()} users and {df_filtered['productId'].nunique()} products")
        
        return df_filtered
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def create_mappings(df):
    """Create user and product mappings"""
    users = df['userId'].unique()
    products = df['productId'].unique()
    
    user_to_idx = {user: idx for idx, user in enumerate(users)}
    idx_to_user = {idx: user for user, idx in user_to_idx.items()}
    product_to_idx = {product: idx for idx, product in enumerate(products)}
    idx_to_product = {idx: product for product, idx in product_to_idx.items()}
    
    return user_to_idx, idx_to_user, product_to_idx, idx_to_product

def create_interaction_matrix(df, user_to_idx, product_to_idx):
    """Create the user-item interaction matrix"""
    rows = [user_to_idx[user] for user in df['userId']]
    cols = [product_to_idx[product] for product in df['productId']]
    data = df['strength'].values
    
    matrix = csr_matrix(
        (data, (rows, cols)), 
        shape=(len(user_to_idx), len(product_to_idx))
    )
    
    return matrix

def train_model():
    """Train the ALS model"""
    try:
        # Load and preprocess data
        df = load_and_preprocess_data()
        
        # Create mappings
        user_to_idx, idx_to_user, product_to_idx, idx_to_product = create_mappings(df)
        
        # Create interaction matrix
        user_item_matrix = create_interaction_matrix(df, user_to_idx, product_to_idx)
        
        logger.info("Training ALS model...")
        
        # Initialize and train model
        model = AlternatingLeastSquares(
            factors=Config.MODEL_FACTORS,
            iterations=Config.MODEL_ITERATIONS,
            regularization=Config.MODEL_REGULARIZATION,
            random_state=42
        )
        
        # Train on transposed matrix (items x users)
        model.fit(user_item_matrix.T)
        
        # Save model and mappings
        model_data = {
            "model": model,
            "user_to_idx": user_to_idx,
            "idx_to_user": idx_to_user,
            "product_to_idx": product_to_idx,
            "idx_to_product": idx_to_product,
            "user_item_matrix": user_item_matrix,
            "config": {
                "factors": Config.MODEL_FACTORS,
                "iterations": Config.MODEL_ITERATIONS,
                "regularization": Config.MODEL_REGULARIZATION
            }
        }
        
        with open(Config.MODEL_PATH, "wb") as f:
            pickle.dump(model_data, f)
        
        logger.info(f"✅ Model trained and saved to {Config.MODEL_PATH}")
        logger.info(f"Model details: {Config.MODEL_FACTORS} factors, {Config.MODEL_ITERATIONS} iterations, {Config.MODEL_REGULARIZATION} regularization")
        
        return model, user_to_idx, idx_to_product
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    train_model()
