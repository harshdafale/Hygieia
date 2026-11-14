"""AstroRAG FastAPI service."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import ollama
from config.settings import WEB_ORIGIN, LLM_MODEL, EMBED_MODEL
from api.db import init_db, connect, health
from api.models import SessionIn, AskIn, FeedbackIn
from rag.pipeline import ask as run_ask
from rag.retrieve import collection

app=FastAPI(title="AstroRAG",version="1.0")
app.add_middleware(CORSMiddleware,allow_origins=[WEB_ORIGIN],allow_methods=["*"],allow_headers=["*"])

@app.on_event("startup")
def startup(): init_db()

@app.get("/health")
def get_health():
    try:
        tags=ollama.list(); names=[m.get("name","") for m in tags.get("models",[])]
        return {"api":"ok","sqlite":health(),"chroma":True,"ollama":True,"llm_model":LLM_MODEL,"llm_available":any(LLM_MODEL in n for n in names),"embedding_model":EMBED_MODEL,"embedding_available":any(EMBED_MODEL in n for n in names)}
    except Exception as exc: return {"api":"ok","sqlite":health(),"chroma":True,"ollama":False,"detail":str(exc)}

@app.post("/sessions")
def create_session(data:SessionIn):
    with connect() as con:
        if not con.execute("SELECT 1 FROM crew WHERE id=?",(data.crew_id,)).fetchone(): raise HTTPException(404,"Crew member not found")
        sid=con.execute("INSERT INTO sessions(crew_id) VALUES(?)",(data.crew_id,)).lastrowid
    return {"session_id":sid}

@app.get("/sessions/{sid}/messages")
def messages(sid:int):
    with connect() as con: return [dict(x) for x in con.execute("SELECT id,role,content,created_at,refused FROM messages WHERE session_id=? ORDER BY id",(sid,))]

@app.post("/ask")
def ask(data:AskIn):
    try: return run_ask(data.question,data.session_id)
    except Exception as exc: raise HTTPException(503,f"Local retrieval/generation service unavailable: {exc}")

@app.post("/feedback")
def feedback(data:FeedbackIn):
    with connect() as con: con.execute("INSERT INTO feedback(message_id,rating,comment) VALUES(?,?,?)",(data.message_id,data.rating,data.comment))
    return {"saved":True}

@app.get("/crew")
def crew():
    with connect() as con: return [dict(x) for x in con.execute("SELECT id,name,role FROM crew ORDER BY id")]

@app.get("/documents")
def documents():
    with connect() as con: return [dict(x) for x in con.execute("SELECT d.id,d.title,d.filename,COUNT(c.id) chunks FROM documents d LEFT JOIN chunks c ON c.document_id=d.id GROUP BY d.id ORDER BY d.title")]

@app.get("/chunks/{chunk_id}")
def chunk(chunk_id:str):
    with connect() as con:
        row=con.execute("SELECT c.*,d.title FROM chunks c JOIN documents d ON d.id=c.document_id WHERE c.id=?",(chunk_id,)).fetchone()
    if not row: raise HTTPException(404,"Chunk not found")
    return dict(row)

@app.get("/analytics")
def analytics():
    with connect() as con:
        return {"questions":con.execute("SELECT COUNT(*) FROM messages WHERE role='user'").fetchone()[0],"refusals":con.execute("SELECT COUNT(*) FROM messages WHERE role='assistant' AND refused=1").fetchone()[0],"avg_feedback":con.execute("SELECT AVG(rating) FROM feedback").fetchone()[0],"top_refused_questions":[x[0] for x in con.execute("SELECT u.content FROM messages u JOIN messages a ON a.session_id=u.session_id AND a.id=u.id+1 WHERE a.refused=1 ORDER BY a.id DESC LIMIT 5")],"evaluations":[]}
