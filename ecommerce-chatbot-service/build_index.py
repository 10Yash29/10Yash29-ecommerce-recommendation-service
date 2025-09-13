import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from utils.chunker import chunk_text
from retriever.vector_store import save_faiss_index

# 📂 Load the markdown file
STORE_INFO_PATH = "store_knowledge/store_info.md"

# 💾 Output paths
INDEX_SAVE_PATH = "retriever/vector_index.faiss"
CHUNKS_SAVE_PATH = "retriever/chunks.pkl"

# 🔍 Load store_info.md
def load_store_info():
    with open(STORE_INFO_PATH, "r", encoding="utf-8") as f:
        return f.read()

def main():
    print("📚 Loading store info...")
    text = load_store_info()

    print("🔪 Chunking store text...")
    chunks = chunk_text(text, max_tokens=100)

    print("🧠 Embedding chunks...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, show_progress_bar=True)

    print("💾 Saving index and chunks...")
    save_faiss_index(embeddings, chunks, INDEX_SAVE_PATH, CHUNKS_SAVE_PATH)

    print("✅ Index build complete!")

if __name__ == "__main__":
    main()
