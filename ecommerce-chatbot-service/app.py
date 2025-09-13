import os
import pickle
import faiss
import numpy as np
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_ENDPOINT = "https://api-inference.huggingface.co/models/google/flan-t5-large"

# Load FAISS index and chunks from retriever/
retriever_dir = os.path.join(os.path.dirname(__file__), "retriever")
index = faiss.read_index(os.path.join(retriever_dir, "vector_index.faiss"))
with open(os.path.join(retriever_dir, "chunks.pkl"), "rb") as f:
    chunks = pickle.load(f)

# Load embedder
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

@app.route("/chatbot", methods=["POST"])
def chatbot():
    try:
        data = request.get_json()
        user_msg = data.get("message", "")

        if not user_msg:
            return jsonify({"error": "No message provided"}), 400

        # Embed the query
        query_vec = embedder.encode([user_msg]).astype("float32")
        D, I = index.search(query_vec, k=3)
        retrieved_context = "\n".join([chunks[i] for i in I[0]])

        # Create prompt
        prompt = f"""You are FootyBot, a smart assistant for the football store FootyTrends.
Use the following store context to answer the user question:

{retrieved_context}

User: {user_msg}
Bot:"""

        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        payload = {"inputs": prompt, "options": {"wait_for_model": True}}
        response = requests.post(HF_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()

        result = response.json()
        answer = result[0].get("generated_text", "Sorry, I couldn't generate a response.")

        return jsonify({"response": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def health():
    return jsonify({"message": "RAG chatbot is running ✅"})

# ✅ Main entry point for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Use Render's injected PORT
    print(f"🚀 Starting FootyBot Chatbot on port {port}...")
    app.run(host="0.0.0.0", port=port)
