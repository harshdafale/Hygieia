"""AstroRAG FastAPI service."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import ollama
from config.settings import WEB_ORIGIN, LLM_MODEL, EMBED_MODEL
from api.db import init_db, connect, health
from api.models import SessionIn, AskIn, CopilotIn, DiagnoseIn, FeedbackIn
from rag.pipeline import ask as run_ask
from rag.retrieve import collection
from rag.generate import generate
from rag.guardrails import severity
from rag.retrieve import retrieve
import json, re

app=FastAPI(title="AstroRAG",version="1.0")
app.add_middleware(CORSMiddleware,allow_origins=[WEB_ORIGIN,"http://127.0.0.1:5173"],allow_methods=["*"],allow_headers=["*"])

@app.on_event("startup")
def startup(): init_db()

@app.get("/health")
def get_health():
    try:
        tags=ollama.list(); names=[str(m.get("model") or m.get("name") or "") for m in tags.get("models",[])]
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

@app.post("/copilot")
def copilot(data:CopilotIn):
    """Handle general crew tasks locally; route health questions through NASA RAG."""
    medical_terms=("symptom","pain","injury","sick","medical","health","medicine","dose","treatment","bleeding","fever","headache","sleep","heart rate","blood pressure","oxygen","radiation","nausea","dizzy","breath","hydration")
    if any(term in data.question.lower() for term in medical_terms):
        try: return run_ask(data.question,data.session_id)
        except Exception as exc: raise HTTPException(503,f"Local evidence assistant unavailable: {exc}")
    with connect() as con:
        history=con.execute("SELECT role,content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT 8",(data.session_id,)).fetchall()
    prior="\n".join(f"{row['role'].upper()}: {str(row['content'])[-900:]}" for row in reversed(history))
    prompt=("You are AstroMed, a helpful private local copilot for astronauts. Respond directly to the current request. "
            "You can explain general topics, brainstorm, draft communications, make schedules, create checklists, summarize information, and organize work. "
            "Give a practical first draft even when details are missing; state a reasonable assumption and ask one focused follow-up only if needed. "
            "Do not refuse ordinary writing, planning, or general knowledge requests or answer only 'No'. Do not claim to know live mission status, invent mission rules, or issue spacecraft control commands. "
            "For diagnosis, treatment, medication, or NASA medical guidance, use the NASA-source health assistant. "
            "Keep simple answers direct. Use headings and numbered steps for plans or handovers. Avoid unnecessary disclaimers.\nRecent conversation:\n" + prior + "\nCurrent request:\n" + data.question + "\nHelpful response:")
    try: answer=generate(prompt)
    except Exception as exc: raise HTTPException(503,f"Local Ollama assistant unavailable: {exc}")
    with connect() as con:
        con.execute("INSERT INTO messages(session_id,role,content) VALUES(?,?,?)",(data.session_id,"user",data.question))
        mid=con.execute("INSERT INTO messages(session_id,role,content,refused) VALUES(?,?,?,0)",(data.session_id,"assistant",answer)).lastrowid
    return {"message_id":mid,"answer":answer,"refused":False,"fallback":False,"severity":"info","sources":[],"mode":"crew_copilot"}

@app.post("/diagnose")
def diagnose(data:DiagnoseIn):
    """Return a rule-prioritized crew health assessment grounded in up to three NASA passages."""
    question=f"Crew report: {data.symptoms}. Severity {data.severity}/10; duration: {data.duration}; worsening: {data.worsening}. Vitals: {json.dumps(data.vitals, ensure_ascii=False)}"
    sources=retrieve(question)[:3]
    red_flags=("chest","breath","bleed","faint","loss of consciousness","unconscious","seizure","confus")
    low=data.symptoms.lower()
    critical=any(flag in low for flag in red_flags)
    level="critical" if critical else "elevated" if data.severity>=7 or data.worsening else "routine"
    answer_data={"summary":"No supporting NASA source found for this report.","possible_causes":[],"actions":[],"contact_ground":"Consult the mission medical protocol and crew medical officer.","confidence":"Low: no supporting passage was retrieved."}
    if sources:
        context="\n\n".join(f"[{i}] {s['text']}" for i,s in enumerate(sources,1))
        prompt=("You are a cautious onboard crew-health information assistant, not a clinician. Use only NASA context. "
                "Return valid JSON with keys summary, possible_causes (array), actions (array), contact_ground, confidence. "
                "Do not diagnose. Do not recommend medication, doses, or treatment. If unsupported, use empty arrays and say so. "
                "Cite any supported statement with [n]. Keep concise.\nNASA context:\n"+context+"\nReport:\n"+question+"\nJSON:")
        try:
            raw=generate(prompt)
            match=re.search(r"\{.*\}",raw,re.S)
            if match:
                parsed=json.loads(match.group(0))
                answer_data={k:parsed.get(k,answer_data[k]) for k in answer_data}
                for key in ("summary","contact_ground","confidence"):
                    if not isinstance(answer_data[key],str): answer_data[key]=str(answer_data[key])
                for key in ("possible_causes","actions"):
                    if not isinstance(answer_data[key],list): answer_data[key]=[]
        except Exception:
            answer_data["confidence"]="Low: the local model response could not be validated."
    if critical:
        answer_data["summary"]="Critical warning sign reported. This is not a diagnosis; follow the onboard emergency protocol now."
        answer_data["possible_causes"]=[]
        answer_data["actions"]=["Stop strenuous activity and alert another crew member.","Follow the onboard emergency medical checklist.","Contact the crew medical officer and ground support immediately."]
        answer_data["contact_ground"]="Immediately, following mission communication procedures."
        answer_data["confidence"]="Urgency set by a fixed safety rule, not by the model."
    elif level=="elevated":
        answer_data["summary"]="Elevated concern based on the reported severity or worsening symptoms. This is not a diagnosis."
        answer_data["actions"]=(answer_data["actions"] or ["Notify the crew medical officer.","Record and recheck the reported vitals."])
    with connect() as con:
        con.execute("INSERT INTO messages(session_id,role,content) VALUES(?,?,?)",(data.session_id,"user",question))
        mid=con.execute("INSERT INTO messages(session_id,role,content,refused) VALUES(?,?,?,?)",(data.session_id,"assistant",json.dumps(answer_data,ensure_ascii=False),int(not sources))).lastrowid
        for rank,source in enumerate(sources,1): con.execute("INSERT INTO retrievals(message_id,chunk_id,rank,distance) VALUES(?,?,?,?)",(mid,source["chunk_id"],rank,source["distance"]))
    return {"message_id":mid,"level":level,**answer_data,"sources":[{"n":i,**s} for i,s in enumerate(sources,1)],"disclaimer":"Guidance only, not a diagnosis. Simulated crew data only."}

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
