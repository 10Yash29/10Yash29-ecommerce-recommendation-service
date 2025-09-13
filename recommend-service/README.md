# E-commerce Recommendation Service

A machine learning-based recommendation system for e-commerce platforms.

## Features

- Product recommendation based on user interactions
- Collaborative filtering algorithm
- RESTful API for integration
- Model training and evaluation utilities

## Files

- `app.py` - Flask API server
- `train_model.py` - Model training script
- `model_utils.py` - Utility functions for model operations
- `config.py` - Configuration settings
- `requirements.txt` - Python dependencies
- `Procfile` - Deployment configuration

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train the model:

```bash
python train_model.py
```

3. Run the API server:

```bash
python app.py
```

## API Endpoints

- `GET /health` - Health check
- `POST /recommend` - Get product recommendations

## Deployment

This service can be deployed on platforms like Heroku using the included Procfile.
