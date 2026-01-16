import faiss
import numpy as np
import os

def build_faiss():
    embeddings = np.load("saved_data/embeddings.npy")
    embedding_dim = embeddings.shape[1]   #requires dimension
    index = faiss.IndexFlatL2(embedding_dim)  #dlatl2 is used cos its fast and uses exact nearest neighbour search (no approximation)
    index.add(embeddings)   #our vector database
    os.makedirs("saved_data", exist_ok=True)
    faiss.write_index(index, "saved_data/faiss.index")
    print(f"Saved FAISS index to saved_data/faiss.index")   # saved

if __name__ == "__main__":
    build_faiss()
