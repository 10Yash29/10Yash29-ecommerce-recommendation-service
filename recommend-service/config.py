import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Model parameters
    MODEL_FACTORS = int(os.getenv('MODEL_FACTORS', 50))
    MODEL_ITERATIONS = int(os.getenv('MODEL_ITERATIONS', 20))
    MODEL_REGULARIZATION = float(os.getenv('MODEL_REGULARIZATION', 0.01))
    
    # Recommendation parameters
    DEFAULT_RECOMMENDATIONS = int(os.getenv('DEFAULT_RECOMMENDATIONS', 5))
    POPULAR_PRODUCTS_COUNT = int(os.getenv('POPULAR_PRODUCTS_COUNT', 10))
    
    # Cache settings
    CACHE_TIMEOUT = int(os.getenv('CACHE_TIMEOUT', 3600))
    
    # API settings
    API_VERSION = os.getenv('API_VERSION', 'v1')
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Data paths
    DATA_PATH = os.getenv('DATA_PATH', 'aggregated_interactions.csv')
    MODEL_PATH = os.getenv('MODEL_PATH', 'trained_model.pkl')
    
    # Thresholds
    MIN_USER_INTERACTIONS = int(os.getenv('MIN_USER_INTERACTIONS', 5))
    MIN_PRODUCT_INTERACTIONS = int(os.getenv('MIN_PRODUCT_INTERACTIONS', 3))
