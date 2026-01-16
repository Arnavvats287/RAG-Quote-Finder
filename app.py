import streamlit as st
import json
import pandas as pd
from rag_pipeline import RAGPipeline

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="RAG Quote finder",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG Quote finder")
st.caption("Retrieval-augmented generation over the quotes dataset")

# Initialize RAG pipeline
rag = RAGPipeline()

# ---------------------------
# Query input
# ---------------------------
query = st.text_input(
    "Enter a natural language query",
    placeholder="Show me quotes about courage by women authors"
)

# ---------------------------
# Search
# ---------------------------
if st.button("🔍 Search") and query.strip():
    with st.spinner("running RAG pipeline..."):
        result = rag.run(query)

    # ---------------------------
    # Generated Answer
    # ---------------------------
    st.subheader("🧾 generated answers")
    st.success(result["answer"])

    # ---------------------------
    # Structured JSON Output
    # ---------------------------
    st.subheader("📂 Structured json response")

    json_result = json.dumps(result, indent=2)
    st.json(result)

    # Download button
    st.download_button(
        label="⬇️ Download JSON result",
        data=json_result,
        file_name="rag_result.json",
        mime="application/json"
    )

    # ---------------------------
    # Retrieved Quotes
    # ---------------------------
    st.subheader("😎 Retrieved quotes~~")

    rows = []
    for i, src in enumerate(result["sources"], 1):
        st.markdown(f"### Quote {i} — {src['author']}")
        st.markdown(f"> {src['quote']}")
        st.markdown(f"**Tags:** {src['tags']}")
        st.markdown(f"**Similarity Score:** `{src['similarity_score']:.4f}`")
        st.markdown("---")

        rows.append({
            "author": src["author"],
            "tags": src["tags"]
        })

    # ---------------------------
    # Visualizations
    # ---------------------------
    st.subheader("📊 Dataset insights")

    df_vis = pd.DataFrame(rows)

    # ---- Author distribution
    st.markdown("#### Quotes by Author")
    author_counts = df_vis["author"].value_counts()
    st.bar_chart(author_counts)

    # ---- Tag distribution
    st.markdown("#### Tag Distribution")

    all_tags = []
    for tag_list in df_vis["tags"]:
        if isinstance(tag_list, str):
            tag_list = eval(tag_list)
        all_tags.extend(tag_list)

    tag_df = pd.DataFrame(all_tags, columns=["tag"])
    tag_counts = tag_df["tag"].value_counts()

    st.bar_chart(tag_counts)

else:
    st.info("⬆️ Enter a query and click Search to begin.")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown(
    "Made with ❤️ by **Arnav** ~ "
)
