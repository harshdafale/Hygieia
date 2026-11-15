"""AstroRAG FastAPI service."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import ollama
from config.settings import WEB_ORIGIN, LLM_MODEL, DIAGNOSIS_MODEL, EMBED_MODEL
from api.db import init_db, connect, health
from api.models import SessionIn, AskIn, CopilotIn, DiagnoseIn, FeedbackIn
from rag.pipeline import ask as run_ask
from rag.retrieve import collection
from rag.generate import generate
from rag.guardrails import severity
from rag.retrieve import retrieve
from rag.evidence import validate_nasa_actions, parse_json_object
import json, re

app=FastAPI(title="AstroRAG",version="1.0")
app.add_middleware(CORSMiddleware,allow_origins=[WEB_ORIGIN,"http://127.0.0.1:5173"],allow_methods=["*"],allow_headers=["*"])

@app.on_event("startup")
def startup(): init_db()

@app.get("/health")
def get_health():
    try:
        tags=ollama.list(); names=[str(m.get("model") or m.get("name") or "") for m in tags.get("models",[])]
        return {"api":"ok","sqlite":health(),"chroma":True,"ollama":True,"llm_model":LLM_MODEL,"llm_available":any(LLM_MODEL in n for n in names),"diagnosis_model":DIAGNOSIS_MODEL,"diagnosis_model_available":any(DIAGNOSIS_MODEL in n for n in names),"embedding_model":EMBED_MODEL,"embedding_available":any(EMBED_MODEL in n for n in names)}
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
    """Generate a Qwen preliminary assessment; verify follow-up steps against NASA quotes."""
    question=f"Crew report: {data.symptoms}. Severity {data.severity}/10; duration: {data.duration}; worsening: {data.worsening}. Vitals: {json.dumps(data.vitals, ensure_ascii=False)}"
    try: sources=retrieve(question)[:3]; retrieval_error=False
    except Exception: sources=[]; retrieval_error=True
    red_flags=("chest","breath","bleed","faint","loss of consciousness","unconscious","seizure","confus")
    low=data.symptoms.lower()
    critical=any(flag in low for flag in red_flags)
    level="critical" if critical else "elevated" if data.severity>=7 or data.worsening else "routine"
    answer_data={"likely_explanation":"The local model could not produce a preliminary assessment.","preliminary_assessment":"The local model could not produce a preliminary assessment.","summary":"The local model could not produce a preliminary assessment.","possible_causes":[],"actions":[],"action_evidence":[],"contact_ground":"No NASA-sourced contact threshold was found in the retrieved passages.","contact_ground_evidence":None,"confidence":"The model response was unavailable or could not be parsed.","safety_override":""}
    prompt=("You are AstroMed, a local crew-health assistant. Provide a useful, cautious MODEL-GENERATED preliminary assessment from symptoms, severity, duration, worsening, and vitals. Use general medical pattern recognition to explain the most plausible cause and a short differential. Do not give a final diagnosis, medication, or treatment plan. Do not refuse or use NASA source availability to determine the assessment. State uncertainty plainly. Return JSON only: likely_explanation (1-3 sentences), preliminary_assessment (a clear explanation with relevant context), possible_causes (2-4 concise alternatives), confidence (brief reason).\n"+question+"\nJSON:")
    try:
        raw=generate(prompt,model=LLM_MODEL)
        parsed=parse_json_object(raw)
        if parsed:
            for key in ("likely_explanation","preliminary_assessment","confidence"):
                if isinstance(parsed.get(key),str) and parsed[key].strip(): answer_data[key]=parsed[key].strip()
            alternatives=parsed.get("possible_causes")
            answer_data["possible_causes"]=[str(x).strip() for x in alternatives if str(x).strip()] if isinstance(alternatives,list) else []
        elif raw.strip():
            answer_data["likely_explanation"]=raw.strip()[:1200]
            answer_data["preliminary_assessment"]=raw.strip()[:3000]
        answer_data["summary"]=answer_data["preliminary_assessment"]
    except Exception as exc:
        answer_data["confidence"]=f"Local Qwen assessment unavailable: {exc}"
    if sources:
        chunks="\n\n".join(f"[{i}] NASA document: {s.get('title') or s.get('filename') or 'NASA source'}; section: {s.get('section') or 'not listed'}; page: {s.get('page') or 'not listed'}\n{s['text']}" for i,s in enumerate(sources,1))
        action_prompt=("Extract only follow-up actions explicitly supported by the numbered NASA passages. Do not diagnose. Return JSON with actions, each {step, source, evidence_quote}; quote must be an exact excerpt of at least 20 characters. Include contact_ground only if the passages directly support it, using the same object shape. Never invent procedures, thresholds, treatments, or medication. Unsupported items are discarded.\nREPORT:\n"+question+"\nNASA PASSAGES:\n"+chunks+"\nJSON:")
        try:
            action_data=parse_json_object(generate(action_prompt,model=LLM_MODEL))
            answer_data["actions"],answer_data["action_evidence"]=validate_nasa_actions(action_data.get("actions"),sources)
            contact,evidence=validate_nasa_actions([action_data.get("contact_ground")],sources)
            answer_data["contact_ground"]=contact[0] if contact else "No NASA-sourced contact threshold was found in the retrieved passages."
            answer_data["contact_ground_evidence"]=evidence[0] if evidence else None
        except Exception:
            answer_data["confidence"]+=" NASA step extraction failed; no unverified steps are shown."
    if not answer_data["actions"]: answer_data["confidence"]+=" No NASA-supported step was verified in the retrieved passages."
    if retrieval_error: answer_data["confidence"]+=" Local NASA search was unavailable, so no source passages could be checked."
    if critical:
        answer_data["safety_override"]="Urgency was raised by an application safety rule, not by the language model. Follow the approved onboard emergency protocol and contact the crew medical officer/ground support."
    elif level=="elevated":
        answer_data["safety_override"]="The application raised urgency because reported severity is high or symptoms are worsening; that rule does not establish a diagnosis."
    with connect() as con:
        con.execute("INSERT INTO messages(session_id,role,content) VALUES(?,?,?)",(data.session_id,"user",question))
        mid=con.execute("INSERT INTO messages(session_id,role,content,refused) VALUES(?,?,?,0)",(data.session_id,"assistant",json.dumps(answer_data,ensure_ascii=False))).lastrowid
        for rank,source in enumerate(sources,1): con.execute("INSERT INTO retrievals(message_id,chunk_id,rank,distance) VALUES(?,?,?,?)",(mid,source["chunk_id"],rank,source["distance"]))
    return {"message_id":mid,"level":level,**answer_data,"sources":[{"n":i,**s} for i,s in enumerate(sources,1)],"disclaimer":"Preliminary model-generated assessment only. Follow-up steps are shown as NASA-checked only when their supporting excerpt is verified in the retrieved source."}
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
