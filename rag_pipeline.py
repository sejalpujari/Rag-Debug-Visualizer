from embed_store import model, create_embeddings
from retrieve import cosine_similarity
from chunker import chunk_text
import os

def load_documents(folder="data"):
    docs = {}
    for file in os.listdir(folder):
        with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
            docs[file] = f.read()
    return docs


def run_rag_debug(query):
    documents = load_documents()

    all_chunks = []
    chunk_metadata = []

    # Chunking
    for filename, text in documents.items():
        chunks = chunk_text(text)
        for idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            chunk_metadata.append({
                "file": filename,
                "chunk_id": idx,
                "text": chunk
            })

    # Embeddings
    chunk_embeddings = create_embeddings(all_chunks)
    query_embedding = model.encode(query)

    # Similarity calculation
    similarity_data = []
    for i, emb in enumerate(chunk_embeddings):
        score = cosine_similarity(query_embedding, emb)
        similarity_data.append({
            "chunk_id": chunk_metadata[i]["chunk_id"],
            "file": chunk_metadata[i]["file"],
            "text": chunk_metadata[i]["text"],
            "score": float(score),
            "embedding": emb
        })

    # Sort by similarity
    similarity_data.sort(key=lambda x: x["score"], reverse=True)

    return {
        "documents": documents,
        "chunks": chunk_metadata,
        "chunk_embeddings": chunk_embeddings,
        "query_embedding": query_embedding,
        "similarity_data": similarity_data
    }
