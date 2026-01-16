import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os

def build_embeddings():
    df = pd.read_csv("saved_data/quotes.csv")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(df['context'].tolist(), show_progress_bar=True)
    os.makedirs("saved_data", exist_ok=True)
    np.save("saved_data/embeddings.npy", embeddings)
    print(f"Saved embeddings shape {embeddings.shape} to saved_data/embeddings.npy")

if __name__ == "__main__":
    build_embeddings()
