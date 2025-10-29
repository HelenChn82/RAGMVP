import os
from typing import Iterable, List, Optional
import chromadb
from chromadb.utils import embedding_functions
from app.settings import settings

# --- Embeddings ---
def get_embedding_fn():
    # Use OpenAI embeddings via chroma's helper
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=settings.openai_api_key,
        model_name=settings.openai_embedding
    )

# --- Vector DB (Chroma) ---
def get_chroma_client():
    os.makedirs(settings.chroma_dir, exist_ok=True)
    return chromadb.PersistentClient(path=settings.chroma_dir)

def get_collection():
    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name="corpus",
        embedding_function=get_embedding_fn()
    )
    return collection

# --- Simple text chunker ---
def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:                     # ← keep only non-empty chunks
            chunks.append(chunk)
        start = end - overlap
        if start <= 0:                # guard
            start = end
    return chunks


def ingest_text(doc_id_prefix: str, text: str, metadata: Optional[dict] = None):
    col = get_collection()
    parts = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
    ids = [f"{doc_id_prefix}-{i}" for i in range(len(parts))]
    metadatas = [metadata or {} for _ in parts]
    col.add(documents=parts, ids=ids, metadatas=metadatas)
    return {"added": len(parts)}

def ingest_lines(doc_id_prefix: str, lines: Iterable[str], metadata: Optional[dict] = None):
    col = get_collection()
    parts = []
    ids = []
    metas = []
    i = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts.append(line)
        ids.append(f"{doc_id_prefix}-{i}")
        metas.append(metadata or {})
        i += 1
    if parts:
        col.add(documents=parts, ids=ids, metadatas=metas)
    return {"added": len(parts)}

def search(query: str, top_k: int = 5):
    col = get_collection()
    res = col.query(query_texts=[query], n_results=top_k)
    # res contains: ids, distances, documents, metadatas
    hits = []
    for i in range(len(res["ids"][0])):
        hits.append({
            "id": res["ids"][0][i],
            "distance": res["distances"][0][i],
            "text": res["documents"][0][i],
            "metadata": res["metadatas"][0][i] if res["metadatas"] else {}
        })
    return hits
