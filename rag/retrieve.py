"""Dense-only Chroma retrieval."""
import chromadb
from config.settings import CHROMA_DIR, TOP_K
from rag.embed import embed

def collection():
    return chromadb.PersistentClient(path=str(CHROMA_DIR)).get_or_create_collection("astrorag", metadata={"hnsw:space":"cosine"})

def retrieve(question: str) -> list[dict]:
    col = collection()
    if col.count() == 0: return []
    result = col.query(query_embeddings=embed([question]), n_results=min(TOP_K, col.count()), include=["documents","metadatas","distances"])
    rows = []
    for i, cid in enumerate(result["ids"][0]):
        rows.append({"chunk_id":cid,"text":result["documents"][0][i],**result["metadatas"][0][i],"distance":result["distances"][0][i]})
    return rows
