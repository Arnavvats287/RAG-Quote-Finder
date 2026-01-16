<div align="center">

# 📚 RAG Quote Finder  
### Retrieval-Augmented Generation over quotes dataset

</div>

---

##  Demo Video

▶️ **Walkthrough**  
https://youtu.be/zVL5Hq1mDPI

 The demo shows:
- Full code walkthrough
- Streamlit UI demo
- Structured JSON output + download
- RAG evaluation
- Dataset visualizations

---

## Project Overview

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline using the  
**Abirate/english_quotes** dataset from HuggingFace.


## Architecture Overview

### RAG Flow

```text
User Query (Streamlit UI)
        ↓
Sentence Transformer (Query Embedding)
        ↓
FAISS Vector Index (Similarity Search)
        ↓
Top-K Relevant Quotes (Context)
        ↓
Gemini LLM (Context + Query)
        ↓
Answer + Sources + Structured JSON Output
