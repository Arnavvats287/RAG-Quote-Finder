from retrieval import Retriever
from generate import Generator

class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.generator = Generator()

    def run(self, query):
        retrieved = self.retriever.retrieve(query)
        answer = self.generator.generate_answer(query, retrieved)
        return {
            "query": query,
            "answer": answer,
            "summary": f"Retrieved {len(retrieved)} semantically similar quotes using FAISS + MiniLM embeddings.",
            "sources": retrieved
        }

if __name__ == "__main__":
    rag = RAGPipeline()
    query = "Quotes about insanity by Einstein"
    result = rag.run(query)
    print(result)
