import os
import pickle
import faiss
import numpy as np
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from functools import lru_cache
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_ENDPOINT = "https://api-inference.huggingface.co/models/google/flan-t5-large"

# Load retriever components
retriever_dir = os.path.join(os.path.dirname(__file__), "retriever")
index = None
chunks = None
embedder = None

def initialize_components():
    """Initialize all components on startup"""
    global index, chunks, embedder
    try:
        logger.info("Loading FAISS index...")
        index = faiss.read_index(os.path.join(retriever_dir, "vector_index.faiss"))
        
        logger.info("Loading chunks...")
        with open(os.path.join(retriever_dir, "chunks.pkl"), "rb") as f:
            chunks = pickle.load(f)
        
        logger.info("Loading sentence transformer...")
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        
        logger.info("✅ All components loaded successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize components: {e}")
        return False

app = Flask(__name__)

# CORS configuration - allow all Vercel preview URLs and production
CORS(app, resources={
    r"/*": {
        "origins": "*",  # Allow all origins for now, restrict in production
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": False
    }
})

@lru_cache(maxsize=100)
def get_cached_embedding(message: str):
    """Cache embeddings for frequently asked questions"""
    return embedder.encode([message]).astype("float32")

def retrieve_context(user_msg: str, k: int = 3) -> str:
    """Retrieve relevant context from vector store"""
    try:
        query_vec = get_cached_embedding(user_msg)
        D, I = index.search(query_vec, k=k)
        
        # Filter by distance threshold
        relevant_indices = [I[0][i] for i in range(len(D[0])) if D[0][i] < 1.5]
        
        if not relevant_indices:
            return "No specific information found. I'll provide general assistance."
        
        context = "\n".join([chunks[i] for i in relevant_indices])
        return context
    except Exception as e:
        logger.error(f"Error retrieving context: {e}")
        return ""

def generate_fallback_response(user_msg: str, context: str) -> str:
    """Generate a simple response using the retrieved context when API is unavailable"""
    user_msg_lower = user_msg.lower()
    
    # Simple pattern matching for common queries
    if any(word in user_msg_lower for word in ["product", "sell", "buy", "available", "stock", "item"]):
        return f"We sell a variety of football merchandise including jerseys, training gear, and accessories. Here's what I found in our catalog:\n\n{context[:300]}...\n\nVisit our store to browse all products!"
    
    elif any(word in user_msg_lower for word in ["return", "refund", "exchange"]):
        return "Our return policy allows you to return items within 30 days of purchase. Items must be unworn with tags attached. Visit your account dashboard to initiate a return or contact our support team."
    
    elif any(word in user_msg_lower for word in ["ship", "delivery", "shipping", "track"]):
        return "We offer standard and express shipping options. Standard delivery takes 5-7 business days, while express is 2-3 days. You can track your order from your account dashboard. International shipping is available to select countries."
    
    elif any(word in user_msg_lower for word in ["size", "sizing", "fit"]):
        return "We have a comprehensive size guide available on each product page. For jerseys, we recommend checking the size chart as football kits often run true to size. If you're between sizes, we suggest sizing up for a more comfortable fit."
    
    elif any(word in user_msg_lower for word in ["payment", "pay", "credit", "debit"]):
        return "We accept all major credit cards (Visa, Mastercard, American Express), PayPal, and Apple Pay. All payments are securely processed and your information is encrypted."
    
    elif any(word in user_msg_lower for word in ["contact", "support", "help", "email"]):
        return "You can reach our support team at support@footytrends.com or through the contact form on our website. We typically respond within 24 hours during business days."
    
    elif context:
        # Use the retrieved context directly
        return f"Based on our store information: {context[:400]}...\n\nFor more details, please browse our store or contact support@footytrends.com"
    
    else:
        return "I'm here to help you with information about our products, shipping, returns, and more! What would you like to know?"

def generate_response(user_msg: str, context: str, retry_count: int = 2) -> str:
    """Generate response using Hugging Face API with retry logic"""
    prompt = f"""You are FootyBot, a helpful assistant for FootyTrends - a football merchandise store.

Context from store knowledge:
{context}

Guidelines:
- Be friendly and concise
- Use football terminology naturally
- If asked about products not in context, suggest browsing the store
- For orders/tracking, guide users to their account dashboard
- Always provide actionable next steps

User Question: {user_msg}

Assistant Response:"""

    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        },
        "options": {"wait_for_model": True}
    }
    
    for attempt in range(retry_count):
        try:
            response = requests.post(HF_ENDPOINT, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            
            if isinstance(result, list) and len(result) > 0:
                answer = result[0].get("generated_text", "").strip()
                # Clean up the response - remove the prompt if it's included
                if "Assistant Response:" in answer:
                    answer = answer.split("Assistant Response:")[-1].strip()
                return answer or "I'm here to help! Could you please rephrase your question?"
            
            return "I'm having trouble generating a response. Please try again."
            
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout on attempt {attempt + 1}")
            if attempt == retry_count - 1:
                # Use fallback response
                logger.info("Using fallback response due to timeout")
                return generate_fallback_response(user_msg, context)
        except requests.exceptions.RequestException as e:
            logger.error(f"API error on attempt {attempt + 1}: {e}")
            if attempt == retry_count - 1:
                # Use fallback response instead of generic error
                logger.info("Using fallback response due to API error")
                return generate_fallback_response(user_msg, context)
    
    return generate_fallback_response(user_msg, context)

@app.route("/", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "FootyBot Chatbot",
        "version": "2.0.0",
        "components": {
            "index_loaded": index is not None,
            "chunks_loaded": chunks is not None,
            "embedder_loaded": embedder is not None
        }
    })

@app.route("/chatbot", methods=["POST"])
def chatbot():
    """Main chatbot endpoint"""
    try:
        if not all([index, chunks, embedder]):
            return jsonify({"error": "Service not fully initialized"}), 503
        
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        user_msg = data.get("message", "").strip()
        
        if not user_msg:
            return jsonify({"error": "No message provided"}), 400
        
        if len(user_msg) > 500:
            return jsonify({"error": "Message too long. Please keep it under 500 characters."}), 400
        
        logger.info(f"Processing message: {user_msg[:50]}...")
        
        # Retrieve relevant context
        retrieved_context = retrieve_context(user_msg, k=3)
        
        # Generate response
        answer = generate_response(user_msg, retrieved_context)
        
        return jsonify({
            "response": answer,
            "confidence": "high" if retrieved_context else "low"
        })
        
    except Exception as e:
        logger.error(f"Chatbot error: {e}", exc_info=True)
        return jsonify({
            "error": "An unexpected error occurred",
            "response": "I apologize, but I'm experiencing technical difficulties. Please try again or contact our support team."
        }), 500

@app.route("/feedback", methods=["POST"])
def feedback():
    """Endpoint to collect user feedback"""
    try:
        data = request.get_json()
        message = data.get("message", "")
        response = data.get("response", "")
        rating = data.get("rating", 0)
        
        # Log feedback (in production, save to database)
        logger.info(f"Feedback received - Rating: {rating}, Message: {message[:50]}")
        
        return jsonify({"status": "Feedback received. Thank you!"})
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        return jsonify({"error": "Failed to submit feedback"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    
    # Initialize components
    if initialize_components():
        logger.info(f"🚀 Starting FootyBot Chatbot on port {port}...")
        app.run(host="0.0.0.0", port=port, debug=False)
    else:
        logger.error("Failed to start - components not initialized")
        exit(1)
