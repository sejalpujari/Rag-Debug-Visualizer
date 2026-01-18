# chunker.py
def chunk_text(text, chunk_size=100, overlap=20):
    """
    Split text into overlapping chunks
    """
    if chunk_size < 10:
        chunk_size = 10
    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        overlap = chunk_size - 1  # prevent useless or dangerous overlap

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks