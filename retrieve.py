import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_top_k(query_embedding, doc_embeddings, chunks, k=2):
    scores = []

    for idx, emb in enumerate(doc_embeddings):
        score = cosine_similarity(query_embedding, emb)
        scores.append((chunks[idx], score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]
