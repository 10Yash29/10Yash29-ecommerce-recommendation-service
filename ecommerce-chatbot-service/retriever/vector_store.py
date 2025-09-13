import faiss
import pickle
import numpy as np

def save_faiss_index(embeddings, chunks, index_path, chunk_path):
    # Convert to numpy array
    embeddings = np.array(embeddings).astype("float32")

    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Save index and chunks
    faiss.write_index(index, index_path)
    with open(chunk_path, "wb") as f:
        pickle.dump(chunks, f)
