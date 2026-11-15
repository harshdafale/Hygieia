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
    """Return a preliminary assessment whose medical claims cite NASA passages."""
    question=f"Crew report: {data.symptoms}. Severity {data.severity}/10; duration: {data.duration}; worsening: {data.worsening}. Vitals: {json.dumps(data.vitals, ensure_ascii=False)}"
    try: sources=retrieve(question)[:3]; retrieval_error=False
    except Exception: sources=[]; retrieval_error=True
    red_flags=("chest","breath","bleed","faint","loss of consciousness","unconscious","seizure","confus")
    low=data.symptoms.lower()
    critical=any(flag in low for flag in red_flags)
    level="critical" if critical else "elevated" if data.severity>=7 or data.worsening else "routine"
    answer_data={"likely_explanation":"The retrieved NASA passages do not support a specific preliminary diagnosis.","preliminary_assessment":"The retrieved NASA passages do not establish a symptom-specific preliminary explanation.","summary":"The retrieved NASA passages do not establish a symptom-specific preliminary explanation.","possible_causes":[],"actions":[],"contact_ground":"The retrieved passages do not specify a symptom-specific contact threshold. Follow the approved onboard protocol and consult the crew medical officer for mission-specific direction.","confidence":"Limited: no directly supporting NASA passage was available.","safety_override":""}
    if sources:
        context="\n\n".join(f"[{i}] NASA document: {s.get('title') or s.get('filename') or 'NASA source'}; section: {s.get('section') or 'not listed'}; page: {s.get('page') or 'not listed'}\n{s['text']}" for i,s in enumerate(sources,1))
        prompt=("You are AstroMed's preliminary crew-health assessment assistant. Analyze the reported symptoms, severity, duration, worsening, and reported vitals. "
                "Use the supplied NASA passages as the sole authority for medical explanations and recommended actions. "
                "Give the best-supported preliminary diagnosis or physiological explanation only when a passage supports it; explain matching details and missing information. "
                "Do not default to 'no' when a cautious, cited possibility is supported. Do not give a definitive diagnosis or invent treatments, medication, doses, thresholds, or NASA procedures. "
                "Cite each possible cause, action, and contact-ground recommendation with [n]. If unsupported, leave its array empty and state the evidence gap. "
                "Return one JSON object only with keys: likely_explanation, preliminary_assessment, summary, possible_causes (array), actions (array), contact_ground, confidence, safety_override. "
                "Do not use markdown fences.\nReport and vitals:\n"+question+"\nNASA passages (cite only these):\n"+context+"\nJSON:")
        try:
            raw=generate(prompt)
            match=re.search(r"\{.*\}",raw,re.S)
            parsed=json.loads(match.group(0)) if match else {}
            if not isinstance(parsed,dict): raise ValueError("Model response was not a JSON object")
            valid_citation=lambda value: bool(re.search(r"\[[1-9]\d*\]",value)) and all(int(n)<=len(sources) for n in re.findall(r"\[([1-9]\d*)\]",value))
            for key in ("likely_explanation","preliminary_assessment","summary","contact_ground","confidence"):
                if isinstance(parsed.get(key),str) and parsed[key].strip(): answer_data[key]=parsed[key].strip()
            for key in ("possible_causes","actions"):
                values=parsed.get(key,[])
                answer_data[key]=[str(x).strip() for x in values if str(x).strip() and valid_citation(str(x))] if isinstance(values,list) else []
            if not valid_citation(answer_data["likely_explanation"]): answer_data["likely_explanation"]="The model did not return a preliminary diagnosis supported by a valid NASA citation."
            if not valid_citation(answer_data["preliminary_assessment"]): answer_data["preliminary_assessment"]="NASA passages were retrieved, but the model did not return a source-cited preliminary explanation. Review the passages with the crew medical officer."
            answer_data["summary"]=answer_data["preliminary_assessment"]
            if not valid_citation(answer_data["contact_ground"]): answer_data["contact_ground"]="The retrieved NASA passages do not establish a contact threshold. Follow approved onboard protocol and consult the crew medical officer."
        except Exception:
            answer_data["confidence"]="Limited: the local model response could not be validated against the retrieved NASA passages. No unsupported actions are presented."
    if retrieval_error: answer_data["confidence"]="Limited: local NASA search was unavailable, so no source passages could be checked."
    if critical:
        answer_data["likely_explanation"]="A preliminary cause cannot be established by the red-flag rule alone."
        answer_data["preliminary_assessment"]="A configured red-flag rule detected a potentially urgent symptom. The rule does not identify a cause; NASA passages and the crew medical team must guide further assessment."
        answer_data["summary"]=answer_data["preliminary_assessment"]
        answer_data["safety_override"]="Urgency was raised by an application safety rule, not by the language model. Follow the approved onboard emergency protocol and contact the crew medical officer/ground support."
        answer_data["confidence"]="Urgency set by a fixed safety rule; possible causes and actions remain limited to cited NASA passages."
    elif level=="elevated":
        answer_data["preliminary_assessment"]+=" Application urgency is elevated because the reported severity is high or symptoms are worsening; that rule does not establish a diagnosis."
        answer_data["summary"]=answer_data["preliminary_assessment"]
    with connect() as con:
        con.execute("INSERT INTO messages(session_id,role,content) VALUES(?,?,?)",(data.session_id,"user",question))
        mid=con.execute("INSERT INTO messages(session_id,role,content,refused) VALUES(?,?,?,?)",(data.session_id,"assistant",json.dumps(answer_data,ensure_ascii=False),int(not sources))).lastrowid
        for rank,source in enumerate(sources,1): con.execute("INSERT INTO retrievals(message_id,chunk_id,rank,distance) VALUES(?,?,?,?)",(mid,source["chunk_id"],rank,source["distance"]))
    return {"message_id":mid,"level":level,**answer_data,"sources":[{"n":i,**s} for i,s in enumerate(sources,1)],"disclaimer":"Preliminary, NASA-source-grounded decision support only. Not a definitive diagnosis or a replacement for the approved onboard protocol or crew medical officer."}

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
