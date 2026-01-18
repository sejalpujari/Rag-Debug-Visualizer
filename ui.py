import streamlit as st
import numpy as np
from rag_pipeline import run_rag_debug
from llm import generate_answer   # ← new import

st.set_page_config(layout="wide")
st.title("🧠 Explainable RAG System (From Scratch)")

query = st.text_input("Ask a question")

if query:
    data = run_rag_debug(query)
    print(data)

    # -------------------------------
    st.header("1️⃣ Raw Documents")
    for name, text in data["documents"].items():
        with st.expander(name):
            st.write(text)

    # -------------------------------
    st.header("2️⃣ Chunked Data")
    for c in data["chunks"]:
        with st.expander(f"{c['file']} | Chunk {c['chunk_id']}"):
            st.write(c["text"])

    # -------------------------------
    st.header("3️⃣ Chunk Embedding Vectors (Preview)")
    st.info("Showing first 10 dimensions only (vectors are large)")

    for i, emb in enumerate(data["chunk_embeddings"]):
        with st.expander(f"Chunk {i} Embedding"):
            st.write(np.round(emb[:10], 4))

    # -------------------------------
    st.header("4️⃣ Query Embedding Vector")
    st.write(np.round(data["query_embedding"][:10], 4))

    # -------------------------------
    st.header("5️⃣ Similarity Scores (All Chunks)")
    for item in data["similarity_data"]:
        with st.expander(
            f"{item['file']} | Chunk {item['chunk_id']} | Score: {round(item['score'], 4)}"
        ):
            st.write(item["text"])

    # -------------------------------
    st.header("6️⃣ Top-K Retrieved Chunks")
    k = st.slider("Top-K Chunks", 1, 10, 5)
    top_k = data["similarity_data"][:k]
    for item in top_k:
        st.success(f"Score: {round(item['score'], 4)}")
        st.write(item["text"])

    # -------------------------------
    st.header("7️⃣ Final Context Sent to LLM")
    final_context = "\n\n".join([item["text"] for item in top_k])
    st.code(final_context)
    st.header("8️⃣ Generated Answer (Groq RAG)")   # ← new section
if st.button("Generate Answer with Groq", type="primary"):
    with st.spinner("Calling Groq model... (usually < 2 seconds)"):
        try:
            answer = generate_answer(query, final_context)
            st.markdown("**Final Answer:**")
            st.write(answer)
        except Exception as e:
            st.error(f"Error calling Groq API: {e}")
            st.info("Check your GROQ_API_KEY in .env and internet connection.")