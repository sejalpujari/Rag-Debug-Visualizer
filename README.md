# Explainable RAG From Scratch – Interactive Retrieval-Augmented Generation Demo

🧠 A **fully transparent, from-scratch** Retrieval-Augmented Generation (RAG) application built with **Python**, **Streamlit**, **Sentence Transformers**, and **Groq** for fast LLM inference.

This project demonstrates core RAG concepts without relying on high-level frameworks like LangChain or LlamaIndex — perfect for learning how embeddings, chunking, vector similarity, and context-augmented generation actually work under the hood.

https://github.com/**yourusername**/explainable-rag-from-scratch

## ✨ Features

- **End-to-end from-scratch RAG pipeline**:
  - Document loading
  - Custom character-based chunking with configurable size & overlap
  - Embeddings using `all-MiniLM-L6-v2` (384 dimensions)
  - Pure NumPy cosine similarity retrieval (no vector DB)
  - Top-k context selection
  - Prompt augmentation + generation via Groq API
- **Fully interactive & explainable Streamlit UI** showing every step:
  - Raw documents
  - All created chunks
  - Chunk & query embedding previews
  - Similarity scores for every chunk
  - Top retrieved chunks
  - Final context sent to LLM
  - Generated answer
- Real-time parameter tuning via sidebar sliders:
  - Chunk size (50–800 characters)
  - Chunk overlap (0–300 characters)
  - Top-k retrieved chunks (1–12)
- Fast inference using **Groq** (supports Llama-3.1, Mixtral, Gemma2, etc.)
- Strict "use only provided context" prompting → minimizes hallucinations

## Demo Screenshots

(Add 3–5 screenshots here later – e.g. full UI, similarity scores expander, final answer section)

1. Main interface with sliders and query  
2. Chunk & embedding visualization  
3. Top-k retrieved chunks + final context  
4. Generated answer example

## Project Structure

```text
.
├── data/                    # Put your .txt documents here
│   ├── rag.txt
│   ├── llm.txt
│   └── ai.txt
├── .env                     # GROQ_API_KEY=...
├── requirements.txt
├── ui.py                    # ← Main Streamlit application
├── rag_pipeline.py          # Core RAG logic (chunk → embed → retrieve)
├── chunker.py               # Simple overlapping chunker
├── embed_store.py           # SentenceTransformer wrapper
├── retrieve.py              # Cosine similarity + top-k
├── llm.py                   # Groq client + RAG prompt
└── README.md

2. Installation
Bash# Clone the repo
git clone https://github.com/yourusername/explainable-rag-from-scratch.git
cd explainable-rag-from-scratch

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file and add your key
USE THIS WEBSITE AND CREATE A GROQ API KEY https://groq.com/
echo "GROQ_API_KEY=gsk_..." > .env
3. Run the app
Bashstreamlit run ui.py
Open http://localhost:8501 in your browser.