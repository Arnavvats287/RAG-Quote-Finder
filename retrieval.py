import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self):
        self.df = pd.read_csv("saved_data/quotes.csv")
        self.embedder = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device="cpu"   # ensures cosine distance is maningful(embedding space consistency)
        )
        self.index = faiss.read_index("saved_data/faiss.index")

    def retrieve(self, query, top_k=3):    # returns top k most relevant ones
        query_vec = self.embedder.encode([query])
        scores, indices = self.index.search(query_vec, top_k)         # retrieval part

        results = []
        for rank, idx in enumerate(indices[0]):
            results.append({
                "quote": self.df.iloc[idx]["quote"],
                "author": self.df.iloc[idx]["author"],
                "tags": self.df.iloc[idx]["tags"],
                "similarity_score": float(scores[0][rank])     # index to actual quotes , gets feeded to LLM
            })
        return results
