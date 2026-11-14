"""End-to-end retrieval, evidence gates, and audit persistence."""
from api.db import connect, init_db
from rag.retrieve import retrieve
from rag.guardrails import gate_one, validate, severity, REFUSAL
from rag.prompt import build_prompt
from rag.generate import generate

def ask(question: str, session_id: int) -> dict:
    init_db()
    sources = retrieve(question)
    refused = not gate_one(sources)
    fallback = False
    answer = REFUSAL if refused else generate(build_prompt(question,sources))
    if not refused: answer,fallback = validate(answer,sources)
    with connect() as con:
        user_id = con.execute("INSERT INTO messages(session_id,role,content) VALUES(?,?,?)",(session_id,"user",question)).lastrowid
        mid = con.execute("INSERT INTO messages(session_id,role,content,refused) VALUES(?,?,?,?)",(session_id,"assistant",answer,int(refused))).lastrowid
        for rank,s in enumerate(sources,1): con.execute("INSERT INTO retrievals VALUES(?,?,?,?)",(mid,s["chunk_id"],rank,s["distance"]))
    return {"message_id":mid,"answer":answer,"refused":refused,"fallback":fallback,"severity":severity(question),"sources":[{"n":i,**s} for i,s in enumerate(sources,1)]}
