from ragas import evaluate
from ragas.metrics import answer_relevancy
from datasets import Dataset
from rag_pipeline import RAGPipeline

rag = RAGPipeline()

# Small evaluation set
questions = [
    "Quotes about insanity by Einstein",
    "Motivational quotes about accomplishment",
    "Humorous quotes by Oscar Wilde"
]

records = []

for q in questions:
    result = rag.run(q)
    records.append({
        "question": q,
        "answer": result["answer"],
        "contexts": [r["quote"] for r in result["sources"]]
    })

dataset = Dataset.from_list(records)

scores = evaluate(
    dataset,
    metrics=[answer_relevancy]
)

print(scores)
