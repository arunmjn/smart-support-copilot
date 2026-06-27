import io
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL = None

def _get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL

def _extract_text(file) -> str:
    name = getattr(file, "name", "")
    raw  = file.read()
    if name.endswith(".pdf"):
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(raw)) as pdf:
                return "\n".join(p.extract_text() or "" for p in pdf.pages)
        except Exception:
            pass
    return raw.decode("utf-8", errors="ignore")

def _chunk_text(text, chunk_size=400, overlap=80):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i: i + chunk_size]))
        i += chunk_size - overlap
    return chunks

def build_vectorstore(uploaded_files):
    all_chunks, all_sources = [], []
    for f in uploaded_files:
        chunks = _chunk_text(_extract_text(f))
        all_chunks.extend(chunks)
        all_sources.extend([getattr(f, "name", "document")] * len(chunks))
    if not all_chunks:
        return None
    embeddings = _get_model().encode(all_chunks, show_progress_bar=False).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return {"index": index, "chunks": all_chunks, "sources": all_sources}

def retrieve_context(vectorstore, query, top_k=3):
    q_emb  = _get_model().encode([query]).astype("float32")
    _, ids = vectorstore["index"].search(q_emb, top_k)
    chunks  = [vectorstore["chunks"][i]  for i in ids[0] if i < len(vectorstore["chunks"])]
    sources = [vectorstore["sources"][i] for i in ids[0] if i < len(vectorstore["sources"])]
    return "\n\n---\n\n".join(chunks), "Retrieved from: " + ", ".join(dict.fromkeys(sources))
