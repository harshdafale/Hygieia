"""SQLite-backed, crew-isolated AstroMed web API."""
from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from functools import wraps

from flask import Flask, g, jsonify, request
from werkzeug.security import generate_password_hash, check_password_hash

from api.db import connect
from config.settings import WEB_ORIGIN
from rag.retrieve import retrieve
from rag.generate import generate
from rag.evidence import validate_nasa_actions, parse_json_object
from config.settings import LLM_MODEL

app = Flask(__name__, static_folder="../web/dist", static_url_path="")

AUTH_SCHEMA = """
CREATE TABLE IF NOT EXISTS crew_members (
 id INTEGER PRIMARY KEY, name TEXT NOT NULL COLLATE NOCASE UNIQUE,
 role TEXT NOT NULL, pin_hash TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS auth_tokens (
 token_hash TEXT PRIMARY KEY, crew_id INTEGER NOT NULL REFERENCES crew_members(id) ON DELETE CASCADE,
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS vitals_logs (
 id INTEGER PRIMARY KEY, crew_id INTEGER NOT NULL REFERENCES crew_members(id) ON DELETE CASCADE,
 heart_rate REAL, systolic REAL, diastolic REAL, temperature REAL, spo2 REAL,
 sleep_hours REAL, exercise_min REAL, co2 REAL, radiation REAL,
 logged_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS symptom_reports (
 id INTEGER PRIMARY KEY, crew_id INTEGER NOT NULL REFERENCES crew_members(id) ON DELETE CASCADE,
 symptoms TEXT NOT NULL, severity INTEGER NOT NULL, urgency TEXT NOT NULL,
 ai_answer_json TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS advisories (
 id INTEGER PRIMARY KEY, crew_id INTEGER NOT NULL REFERENCES crew_members(id) ON DELETE CASCADE,
 status TEXT NOT NULL, text TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS crew_chat_messages (
 id INTEGER PRIMARY KEY, crew_id INTEGER NOT NULL REFERENCES crew_members(id) ON DELETE CASCADE,
 role TEXT NOT NULL CHECK(role IN ('user','assistant')), content TEXT NOT NULL,
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_vitals_owner_time ON vitals_logs(crew_id, logged_at);
CREATE INDEX IF NOT EXISTS idx_reports_owner_time ON symptom_reports(crew_id, created_at);
CREATE INDEX IF NOT EXISTS idx_advisories_owner_time ON advisories(crew_id, created_at);
"""

VITAL_FIELDS = ("heart_rate", "systolic", "diastolic", "temperature", "spo2", "sleep_hours", "exercise_min", "co2", "radiation")
FIELD_LABELS = {"heart_rate":"Heart rate", "systolic":"Systolic BP", "diastolic":"Diastolic BP", "temperature":"Temperature", "spo2":"Blood oxygen", "sleep_hours":"Sleep", "exercise_min":"Exercise", "co2":"Cabin CO2", "radiation":"Radiation dose"}

def init_auth_db() -> None:
    with connect() as db:
        db.executescript(AUTH_SCHEMA)

@app.before_request
def prepare_request():
    init_auth_db()
    origin=request.headers.get("Origin")
    if origin in (WEB_ORIGIN, "http://127.0.0.1:5173"):
        g.cors_origin=origin

@app.after_request
def cors(response):
    if getattr(g,"cors_origin",None):
        response.headers["Access-Control-Allow-Origin"]=g.cors_origin
        response.headers["Vary"]="Origin"
        response.headers["Access-Control-Allow-Headers"]="Content-Type, Authorization"
        response.headers["Access-Control-Allow-Methods"]="GET, POST, OPTIONS"
    return response

@app.route("/api/<path:_path>", methods=["OPTIONS"])
def options(_path):
    return "", 204

def crew_required(fn):
    @wraps(fn)
    def wrapped(*args,**kwargs):
        token=request.headers.get("Authorization","").removeprefix("Bearer ").strip()
        if not token:
            return jsonify(error="Log in to continue."),401
        digest=hashlib.sha256(token.encode()).hexdigest()
        with connect() as db:
            row=db.execute("SELECT crew_id FROM auth_tokens WHERE token_hash=?",(digest,)).fetchone()
        if not row:
            return jsonify(error="Session expired. Please log in again."),401
        g.crew_id=int(row["crew_id"])
        return fn(*args,**kwargs)
    return wrapped

@app.get("/api/crew-members")
def crew_members():
    with connect() as db:
        return jsonify([dict(row) for row in db.execute("SELECT id,name,role FROM crew_members ORDER BY name")])

@app.get("/api/me")
@crew_required
def me():
    with connect() as db:
        row=db.execute("SELECT id,name,role FROM crew_members WHERE id=?",(g.crew_id,)).fetchone()
    return jsonify(dict(row)) if row else (jsonify(error="Crew member not found."),404)

@app.post("/api/register")
def register():
    data=request.get_json(silent=True) or {}
    name=str(data.get("name","")).strip()[:80]
    role=str(data.get("role","")).strip()[:80]
    pin=str(data.get("pin","")).strip()
    if len(name)<2 or not role or not pin.isdigit() or not 4<=len(pin)<=12:
        return jsonify(error="Enter a name, role, and a 4 to 12 digit PIN."),400
    try:
        with connect() as db:
            db.execute("INSERT INTO crew_members(name,role,pin_hash) VALUES(?,?,?)",(name,role,generate_password_hash(pin)))
        return jsonify(registered=True),201
    except sqlite3.IntegrityError:
        return jsonify(error="That crew member name is already registered."),409

@app.post("/api/login")
def login():
    data=request.get_json(silent=True) or {}
    try: crew_id=int(data.get("crew_id"))
    except (TypeError,ValueError): return jsonify(error="Choose a crew member."),400
    pin=str(data.get("pin","")).strip()
    with connect() as db:
        row=db.execute("SELECT id,pin_hash FROM crew_members WHERE id=?",(crew_id,)).fetchone()
        if not row or not check_password_hash(row["pin_hash"],pin):
            return jsonify(error="Crew member or PIN is incorrect."),401
        token=secrets.token_urlsafe(32)
        db.execute("INSERT INTO auth_tokens(token_hash,crew_id) VALUES(?,?)",(hashlib.sha256(token.encode()).hexdigest(),crew_id))
    return jsonify(token=token,crew_id=crew_id)

@app.post("/api/logout")
@crew_required
def logout():
    token=request.headers.get("Authorization","").removeprefix("Bearer ").strip()
    with connect() as db: db.execute("DELETE FROM auth_tokens WHERE token_hash=?",(hashlib.sha256(token.encode()).hexdigest(),))
    return jsonify(logged_out=True)

def baseline_for(db: sqlite3.Connection, crew_id: int) -> dict:
    since=(datetime.now(timezone.utc)-timedelta(days=14)).isoformat()
    row=db.execute("SELECT "+", ".join(f"AVG({field}) AS {field}" for field in VITAL_FIELDS)+" FROM vitals_logs WHERE crew_id=? AND datetime(logged_at)>=datetime(?)",(crew_id,since)).fetchone()
    return {field:row[field] for field in VITAL_FIELDS if row[field] is not None}

def status_for(value, baseline):
    if value is None or baseline is None or float(baseline)==0: return "Nominal"
    deviation=abs(float(value)-float(baseline))/abs(float(baseline))
    return "Nominal" if deviation<=0.10 else "Watch" if deviation<=0.20 else "Out of range"

def serialized_vitals(row, baseline):
    result=dict(row)
    result["status"]={f:status_for(result.get(f),baseline.get(f)) for f in VITAL_FIELDS if result.get(f) is not None}
    return result

@app.post("/api/vitals")
@crew_required
def save_vitals():
    data=request.get_json(silent=True) or {}
    values=[]
    for field in VITAL_FIELDS:
        raw=data.get(field)
        try: values.append(None if raw in (None,"") else float(raw))
        except (TypeError,ValueError): return jsonify(error=f"{FIELD_LABELS[field]} must be numeric."),400
    if all(v is None for v in values): return jsonify(error="Enter at least one vital measurement."),400
    with connect() as db:
        db.execute("INSERT INTO vitals_logs(crew_id,"+",".join(VITAL_FIELDS)+") VALUES("+",".join("?" for _ in range(1+len(values)))+")",(g.crew_id,*values))
        baseline=baseline_for(db,g.crew_id)
        latest=db.execute("SELECT * FROM vitals_logs WHERE crew_id=? ORDER BY logged_at DESC,id DESC LIMIT 1",(g.crew_id,)).fetchone()
        statuses=serialized_vitals(latest,baseline)["status"]
        for field,status in statuses.items():
            if status!="Nominal": db.execute("INSERT INTO advisories(crew_id,status,text) VALUES(?,?,?)",(g.crew_id,status,f"{FIELD_LABELS[field]} differs from your 14-day personal average."))
    return jsonify(log=serialized_vitals(latest,baseline),baseline=baseline),201

@app.get("/api/vitals")
@crew_required
def get_vitals():
    since=(datetime.now(timezone.utc)-timedelta(days=7)).isoformat()
    with connect() as db:
        baseline=baseline_for(db,g.crew_id)
        rows=db.execute("SELECT * FROM vitals_logs WHERE crew_id=? AND datetime(logged_at)>=datetime(?) ORDER BY logged_at",(g.crew_id,since)).fetchall()
    return jsonify(baseline=baseline,logs=[serialized_vitals(row,baseline) for row in rows])

@app.post("/api/symptom-reports")
@crew_required
def save_report():
    data=request.get_json(silent=True) or {}
    try:
        severity=int(data.get("severity")); answer=data.get("ai_answer_json",{})
        if not 1<=severity<=10: raise ValueError
    except (TypeError,ValueError): return jsonify(error="Severity must be between 1 and 10 and the AI response must be valid JSON."),400
    symptoms=str(data.get("symptoms","")).strip()[:2000]; urgency=str(data.get("urgency","Routine"))
    if not symptoms: return jsonify(error="Symptom report cannot be empty."),400
    if not isinstance(answer,dict): return jsonify(error="AI response must be an object."),400
    with connect() as db:
        cur=db.execute("INSERT INTO symptom_reports(crew_id,symptoms,severity,urgency,ai_answer_json) VALUES(?,?,?,?,?)",(g.crew_id,symptoms,severity,urgency,json.dumps(answer,ensure_ascii=False)))
    return jsonify(id=cur.lastrowid),201

@app.get("/api/symptom-reports")
@crew_required
def list_reports():
    with connect() as db:
        rows=db.execute("SELECT * FROM symptom_reports WHERE crew_id=? ORDER BY created_at DESC,id DESC LIMIT 100",(g.crew_id,)).fetchall()
    return jsonify([{**dict(r),"ai_answer_json":json.loads(r["ai_answer_json"])} for r in rows])

@app.get("/api/advisories")
@crew_required
def list_advisories():
    with connect() as db:
        rows=db.execute("SELECT * FROM advisories WHERE crew_id=? ORDER BY created_at DESC,id DESC LIMIT 100",(g.crew_id,)).fetchall()
    return jsonify([dict(r) for r in rows])

@app.post("/api/copilot")
@crew_required
def copilot():
    data=request.get_json(silent=True) or {}; question=str(data.get("question","")).strip()
    if not question or len(question)>2000: return jsonify(error="Enter a task or question (up to 2000 characters)."),400
    with connect() as db:
        history=db.execute("SELECT role,content FROM crew_chat_messages WHERE crew_id=? ORDER BY id DESC LIMIT 8",(g.crew_id,)).fetchall()
    # Bound prior chat so a long thread cannot push the current question out of
    # the local model's context window. The latest turns are the most useful for follow-ups.
    prior="\n".join(f"{r['role'].upper()}: {str(r['content'])[-900:]}" for r in reversed(history))
    prompt=("You are AstroMed, a helpful local assistant for an astronaut. Respond to the user's current request directly. "
            "You can explain general topics, brainstorm, draft messages, make schedules, create checklists, summarize information, and help organize work. "
            "Give a useful first draft or practical next steps even when some details are missing; state a reasonable assumption and ask one focused follow-up only if needed. "
            "Do not answer only 'No' or refuse a normal writing, planning, or general knowledge request. "
            "Do not claim to know this mission's live status, invent mission rules, or issue spacecraft control commands. If a request depends on unavailable mission-specific facts, say exactly what is missing and still offer a generic template. "
            "For diagnosis, treatment, or medication advice, direct the crew member to Medical Diagnostics and its NASA evidence. "
            "Use a short direct answer for simple questions. For plans and handovers, use clear headings, steps, and a checklist. Avoid unnecessary disclaimers.\n"
            "Recent conversation (use it to understand references such as 'that' or 'make it shorter'):\n"+prior+
            "\nCurrent user request:\n"+question+"\nHelpful response:")
    try: answer=generate(prompt)
    except Exception as exc: return jsonify(error=f"Local Ollama assistant unavailable: {exc}"),503
    with connect() as db:
        db.execute("INSERT INTO crew_chat_messages(crew_id,role,content) VALUES(?,?,?)",(g.crew_id,"user",question))
        db.execute("INSERT INTO crew_chat_messages(crew_id,role,content) VALUES(?,?,?)",(g.crew_id,"assistant",answer))
    return jsonify(answer=answer)

@app.post("/diagnose")
@crew_required
def diagnose():
    data=request.get_json(silent=True) or {}
    symptoms=str(data.get("symptoms","")).strip()[:2000]
    try: severity=int(data.get("severity",1))
    except (TypeError,ValueError): return jsonify(error="Severity must be 1 through 10."),400
    if not symptoms or not 1<=severity<=10: return jsonify(error="Enter symptoms and severity from 1 to 10."),400
    with connect() as db:
        baseline=baseline_for(db,g.crew_id)
        vitals=db.execute("SELECT * FROM vitals_logs WHERE crew_id=? ORDER BY logged_at DESC,id DESC LIMIT 1",(g.crew_id,)).fetchone()
        recent=db.execute("SELECT symptoms,severity,urgency,ai_answer_json,created_at FROM symptom_reports WHERE crew_id=? ORDER BY created_at DESC,id DESC LIMIT 3",(g.crew_id,)).fetchall()
    recent_vitals=serialized_vitals(vitals,baseline) if vitals else {}
    duration=str(data.get("duration") or "not provided")[:160]
    # Retrieval can fail when the local vector index or embedding model is unavailable.
    # Keep the safety-rule assessment usable and explain the missing evidence instead of
    # letting an indexing/model exception crash the entire request.
    recent_context=[]
    for row in recent:
        item=dict(row)
        # Keep symptom history for context, but don't feed old model refusals back
        # into the next assessment as if they were clinical findings.
        recent_context.append({"symptoms":item["symptoms"],"severity":item["severity"],"urgency":item["urgency"],"created_at":item["created_at"]})
    context_query=(f"Current symptoms: {symptoms}. Reported severity: {severity}/10. Duration: {duration}. "
                   f"Worsening: {bool(data.get('worsening'))}. Current vitals and personal-baseline status: {json.dumps(recent_vitals,ensure_ascii=False)}. "
                   f"Recent symptom history: {json.dumps(recent_context,ensure_ascii=False)}")
    # Retrieval can fail when the local vector index or embedding model is unavailable.
    # Keep the safety-rule assessment usable and explain the missing evidence instead of
    # letting an indexing/model exception crash the entire request.
    try:
        sources=retrieve(context_query)[:3]
        retrieval_error=False
    except Exception:
        sources=[]
        retrieval_error=True
    red_flags=("chest","breath","bleed","faint","loss of consciousness","unconscious","seizure","confus")
    critical=any(term in symptoms.lower() for term in red_flags)
    urgency="Critical" if critical else "Elevated" if severity>=7 or data.get("worsening") else "Routine"
    answer={"likely_explanation":"The medical model did not return a preliminary explanation.","preliminary_assessment":"The medical model did not return a preliminary explanation.","summary":"The medical model did not return a preliminary explanation.","possible_causes":[],"actions":[],"action_evidence":[],"contact_ground":"No NASA-sourced contact threshold was found in the retrieved passages.","contact_ground_evidence":None,"confidence":"Model assessment unavailable; NASA action review is separate.","safety_override":""}
    # Keep the diagnosis model's prompt free of NASA retrieval text. NASA material
    # is only supplied to the separate action-step checker below.
    diagnosis_prompt=("You are AstroMed's local medical model. Give a cautious PRELIMINARY assessment based on the crew report, symptoms, severity, duration, worsening, vitals compared with personal baseline, and prior symptom reports. "
                      "Use medical pattern recognition to state the most plausible explanation and a brief differential. This is model-generated, uncertain, and not a final diagnosis. "
                      "Do not refuse because NASA evidence is missing; NASA sources are not used to decide this assessment. Do not prescribe medications or treatment. "
                      "Return a JSON object only with likely_explanation, preliminary_assessment, possible_causes (array), and confidence.\nCREW REPORT AND CONTEXT:\n"+context_query+"\nJSON:")
    try:
        diagnosis_raw=generate(diagnosis_prompt, model=LLM_MODEL)
        diagnosis=parse_json_object(diagnosis_raw)
        if diagnosis:
            for key in ("likely_explanation","preliminary_assessment","confidence"):
                value=diagnosis.get(key)
                if isinstance(value,str) and value.strip(): answer[key]=value.strip()
            alternatives=diagnosis.get("possible_causes")
            answer["possible_causes"]=[str(x).strip() for x in alternatives if str(x).strip()] if isinstance(alternatives,list) else []
        elif diagnosis_raw.strip():
            answer["likely_explanation"]=diagnosis_raw.strip()[:1200]
            answer["preliminary_assessment"]=diagnosis_raw.strip()[:2400]
        answer["summary"]=answer["preliminary_assessment"]
    except Exception as exc:
        answer["confidence"]=f"The local medical model could not produce an assessment: {exc}"
    if sources:
        chunks="\n\n".join(f"[{i}] NASA document: {s.get('title') or s.get('filename') or 'NASA source'}; section: {s.get('section') or 'not listed'}; page: {s.get('page') or 'not listed'}\n{s['text']}" for i,s in enumerate(sources,1))
        action_prompt=("You are extracting follow-up steps from NASA text for a crew-health report. Use only the numbered NASA passages below. Do not diagnose. "
                       "Return only steps directly supported by these passages. Return each step as {step, source, evidence_quote}, where evidence_quote is an exact excerpt of at least 20 characters from that numbered passage. "
                       "Never add general medical advice, treatments, medication, dose, threshold, or procedure absent from the passage. Unverified steps will be discarded. "
                       "Return JSON only with actions (array) and contact_ground (same object or empty string). If no step is supported, use an empty actions array.\nREPORT:\n"+context_query+"\nNASA PASSAGES:\n"+chunks+"\nJSON:")
        try:
            action_data=parse_json_object(generate(action_prompt, model=LLM_MODEL))
            answer["actions"],answer["action_evidence"]=validate_nasa_actions(action_data.get("actions"),sources)
            contact,contact_evidence=validate_nasa_actions([action_data.get("contact_ground")],sources)
            answer["contact_ground"]=contact[0] if contact else "No NASA-sourced contact threshold was found in the retrieved passages."
            answer["contact_ground_evidence"]=contact_evidence[0] if contact_evidence else None
        except Exception:
            answer["confidence"]+=" NASA step extraction failed; no unverified actions are shown."
    if not answer["actions"]: answer["confidence"]+=" No NASA-supported step was verified in the retrieved passages."
    if retrieval_error:
        answer["confidence"]="Limited: the local NASA search service was unavailable, so no source passages could be checked."
    if critical:
        answer.update(safety_override="Urgency was raised by an application safety rule, not by the language model. Follow the approved onboard emergency protocol and contact the crew medical officer/ground support.",confidence="Urgency was raised by a fixed safety rule; the preliminary explanation is model-generated and NASA-supported steps are shown separately.")
    elif urgency=="Elevated":
        answer["safety_override"]="The application raised urgency because the reported severity is high or symptoms are worsening; that rule does not establish a diagnosis."
    payload={**answer,"urgency":urgency,"sources":[{"n":i,**s} for i,s in enumerate(sources,1)],"disclaimer":"The preliminary explanation is model-generated and may be wrong. Steps are shown as NASA-supported only when they cite retrieved passages; this is not a final diagnosis or a replacement for the crew medical officer."}
    with connect() as db:
        db.execute("INSERT INTO symptom_reports(crew_id,symptoms,severity,urgency,ai_answer_json) VALUES(?,?,?,?,?)",(g.crew_id,symptoms,severity,urgency,json.dumps(payload,ensure_ascii=False)))
        for field,status in (recent_vitals.get("status") or {}).items():
            if status!="Nominal": db.execute("INSERT INTO advisories(crew_id,status,text) VALUES(?,?,?)",(g.crew_id,status,f"{FIELD_LABELS[field]} differs from your 14-day personal average."))
    return jsonify(payload)

@app.get("/api/dashboard")
@crew_required
def dashboard():
    since7=(datetime.now(timezone.utc)-timedelta(days=7)).isoformat()
    with connect() as db:
        baseline=baseline_for(db,g.crew_id)
        latest=db.execute("SELECT * FROM vitals_logs WHERE crew_id=? ORDER BY logged_at DESC,id DESC LIMIT 1",(g.crew_id,)).fetchone()
        logs=db.execute("SELECT * FROM vitals_logs WHERE crew_id=? AND datetime(logged_at)>=datetime(?) ORDER BY logged_at",(g.crew_id,since7)).fetchall()
        advisories=db.execute("SELECT * FROM advisories WHERE crew_id=? ORDER BY created_at DESC,id DESC LIMIT 8",(g.crew_id,)).fetchall()
    return jsonify(baseline=baseline,latest=serialized_vitals(latest,baseline) if latest else None,logs=[serialized_vitals(r,baseline) for r in logs],advisories=[dict(x) for x in advisories])

@app.get("/api/health")
def health():
    with connect() as db: db.execute("SELECT 1")
    return jsonify(api="ok",database="ok")

@app.get("/")
def index():
    return app.send_static_file("index.html")
