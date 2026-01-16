from retrieval import Retriever
from generate import Generator
                                          # True RAG ~
class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever()   # Finds relevant quotes using embeddings + FAISS
        self.generator = Generator()   # Uses gemini to generate answer

    def run(self, query):    # entry point , this gets called everywhere
        retrieved = self.retriever.retrieve(query)
        answer = self.generator.generate_answer(query, retrieved)    # answers are dataset backed
        return {
            "query": query,
            "answer": answer,                                         # JSON Output
            "summary": f"Retrieved {len(retrieved)} semantically similar quotes using FAISS + MiniLM embeddings.",
            "sources": retrieved
        }

if __name__ == "__main__":
    rag = RAGPipeline()
    query = "Quotes about insanity by Einstein"   # testing if pipeline works end to end , evidence part
    result = rag.run(query)
    print(result)
