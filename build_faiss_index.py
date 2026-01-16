import faiss
import numpy as np
import os

def build_faiss():
    embeddings = np.load("saved_data/embeddings.npy")
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings)
    os.makedirs("saved_data", exist_ok=True)
    faiss.write_index(index, "saved_data/faiss.index")
    print(f"Saved FAISS index to saved_data/faiss.index")

if __name__ == "__main__":
    build_faiss()
