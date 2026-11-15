"""Dense and exact-term retrieval from indexed NASA passages."""
import chromadb
from config.settings import CHROMA_DIR, TOP_K
from rag.embed import embed

def collection():
    return chromadb.PersistentClient(path=str(CHROMA_DIR)).get_or_create_collection("astrorag", metadata={"hnsw:space":"cosine"})

def retrieve(question: str) -> list[dict]:
    col=collection()
    if col.count()==0: return []
    result=col.query(query_embeddings=embed([question]),n_results=min(TOP_K,col.count()),include=["documents","metadatas","distances"])
    return [{"chunk_id":cid,"text":result["documents"][0][i],**(result["metadatas"][0][i] or {}),"distance":result["distances"][0][i]} for i,cid in enumerate(result["ids"][0])]

def retrieve_matching(terms: list[str]) -> list[dict]:
    """Return exact NASA chunks containing symptom terms without another model call."""
    col=collection()
    rows={}
    for term in terms:
        if not term: continue
        result=col.get(where_document={"$contains":term},limit=8,include=["documents","metadatas"])
        for cid,text,metadata in zip(result["ids"],result["documents"],result["metadatas"]):
            rows[cid]={"chunk_id":cid,"text":text,**(metadata or {}),"distance":1.0}
    return list(rows.values())
