"""Idempotently ingest locally supplied documents into Chroma and SQLite."""
import argparse,hashlib
from pathlib import Path
from api.db import connect,init_db
from config.settings import RAW_DIR
from ingest.parse import parse
from ingest.chunk import chunk_sections
from rag.embed import embed
from rag.retrieve import collection

def ingest_file(path:Path)->int:
    digest=hashlib.sha256(path.read_bytes()).hexdigest(); title=path.stem
    init_db()
    with connect() as con:
        if con.execute("SELECT 1 FROM documents WHERE filename=? AND content_hash=?",(path.name,digest)).fetchone(): return 0
        cur=con.execute("INSERT INTO documents(title,source_url,doc_type,filename,content_hash) VALUES(?,?,?,?,?)",(title,None,path.suffix.lstrip("."),path.name,digest)); doc_id=cur.lastrowid
    parts=chunk_sections(parse(path),title); col=collection()
    for start in range(0,len(parts),16):
        batch=parts[start:start+16]; ids=[f"{doc_id}-{start+i}" for i in range(len(batch))]
        vecs=embed([x["text"] for x in batch]); metas=[{"document_id":doc_id,"title":title,"section":x["section"],"page":x["page"] or 0,"filename":path.name} for x in batch]
        col.add(ids=ids,documents=[x["text"] for x in batch],embeddings=vecs,metadatas=metas)
        with connect() as con:
            con.executemany("INSERT INTO chunks(id,document_id,section,page,text) VALUES(?,?,?,?,?)",[(ids[i],doc_id,batch[i]["section"],batch[i]["page"],batch[i]["text"]) for i in range(len(batch))])
    return len(parts)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--all",action="store_true"); p.add_argument("--file"); a=p.parse_args()
    files=list(RAW_DIR.glob("**/*")) if a.all else ([Path(a.file)] if a.file else [])
    files=[f for f in files if f.is_file() and f.suffix.lower() in (".txt",".pdf",".html",".htm")]
    total=sum(ingest_file(f) for f in files); print(f"Ingested {len(files)} candidate documents; added {total} chunks.")

if __name__=="__main__": main()
