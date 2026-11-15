# AstroMed

AstroMed is a local crew-support prototype for long-duration spaceflight. It combines a private per-crew health log, Qwen 2.5 7B for the general copilot, MedGemma 1.5 for preliminary medical assessments, and NASA-source retrieval to verify suggested follow-up steps.

The interface follows the supplied “AstroMed | Crew health, beyond Earth” design. The website uses Vite, the local API uses Flask, crew records use SQLite, document search uses ChromaDB, and Ollama runs the language and embedding models locally.

> AstroMed is an informational prototype, not a diagnostic tool or a substitute for an onboard medical protocol or clinician. Use simulated crew data only. Do not enter real personal health information.

## Features

- Crew members register a name, role, and 4–12 digit PIN. Only a Werkzeug PIN hash is stored.
- Login issues a random bearer token; SQLite stores only its SHA-256 hash.
- Vitals, symptom reports, advisories, and copilot conversation history are scoped to the authenticated crew member.
- The dashboard loads each member’s latest readings, seven-day telemetry, and advisories from the API.
- Personal baselines average that member’s previous 14 days. Readings within 10% are Nominal, within 20% are Watch, and beyond that are Out of range. Missing baselines show no data.
- Diagnostics includes that member’s latest vitals and three latest symptom reports and asks local MedGemma for a clearly labeled preliminary assessment. The possible explanation is model-generated. Suggested follow-up steps and contact guidance are shown only when an exact supporting quote is found in a retrieved NASA passage. Fixed red-flag and severity rules are shown separately from the model assessment.
- General crew tasks use the local copilot. Recent turns are included to support follow-up questions.
- Documents are indexed locally with dense embeddings. SQLite stores document metadata, chunks, retrieval audit entries, and app records.

All relational database access uses Python’s built-in sqlite3 module. No ORM is used.

## Requirements

Python 3.10+, Node.js and npm, Ollama, and NASA PDF, HTML, or TXT files that you have permission to use.

Install the local models once:

    ollama pull qwen2.5:7b
    ollama pull medgemma1.5:latest
    ollama pull nomic-embed-text

Install Python and frontend dependencies from the project root:

    python -m pip install -r requirements.txt
    npm --prefix web install

## Add NASA documents

Place NASA source files in data/raw/. Supported formats are PDF, HTML, and TXT. Then run:

    python -m ingest.ingest --all

To ingest one file:

    python -m ingest.ingest --file "data/raw/your-document.pdf"

The ingestion command parses files, chunks the text, embeds it with nomic-embed-text, and writes matching chunk IDs to ChromaDB and SQLite. If no useful passages are indexed, health questions are refused. config/sources.yaml is optional and starts empty; add only URLs you have verified. Runtime answering does not call NASA or other external APIs.

## Run locally

Start Ollama if it is not already running. Open two terminals in the project root.

Terminal 1, Flask API:

    python -m flask --app api.flask_app run --host 127.0.0.1 --port 5000

Terminal 2, website:

    npm --prefix web run dev -- --host 127.0.0.1

Open http://127.0.0.1:5173. Register the first crew member from the login screen, choose a PIN, and sign in. Each crew member registers separately and gets a separate history.

The standalone FastAPI RAG service is also available:

    uvicorn api.main:app --host 127.0.0.1 --port 8000

## Database and account isolation

The SQLite database is db/astrorag.sqlite. The Flask service creates these tables automatically:

- crew_members: id, name, role, pin_hash, created_at
- vitals_logs: crew_id and heart rate, blood pressure, temperature, oxygen, sleep, exercise, CO2, and radiation readings
- symptom_reports: crew_id, symptoms, severity, urgency, AI answer JSON, and timestamp
- advisories: crew_id, status, text, and timestamp
- auth_tokens: hash of bearer token and owning crew ID
- crew_chat_messages: private recent copilot history per crew member

Protected endpoints derive the owner ID from the bearer token. They do not accept an owner ID from the browser for reading or writing that member’s history.

## Flask routes

- GET /api/crew-members: list names and roles for the login picker
- POST /api/register: create an account and hash its PIN
- POST /api/login: verify a PIN and issue a token
- GET /api/me: current authenticated crew member
- POST and GET /api/vitals: save readings and retrieve the last seven days
- GET /api/dashboard: latest vitals, baseline, seven-day logs, advisories
- POST /diagnose: NASA-grounded assessment with recent personal context
- GET and POST /api/symptom-reports: private symptom history
- GET /api/advisories: private advisories
- POST /api/copilot: local task copilot with recent conversation context
- GET /api/health: Flask and SQLite health

## Evaluation and configuration

Change model names, chunk sizes, overlap, top-k, and paths in config/settings.py. Set `ASTROMED_LLM_MODEL` for the copilot, `ASTROMED_DIAGNOSIS_MODEL` for preliminary medical assessments, and `ASTRORAG_MAX_DIST` to override the retrieval threshold. Both model overrides must name locally installed Ollama models.

    python -m eval.calibrate
    python -m eval.run_eval

## Limitations

- Qwen 2.5 7B powers general crew tasks; MedGemma 1.5 generates preliminary medical possibilities. Its output can still be wrong and requires independent clinical review. The app validates each recommended step against an exact excerpt from retrieved NASA data; if no step can be verified, it leaves the action list empty. Code-based urgency rules remain separate. NASA assigns in-mission care to its Flight Medicine team, including Crew Medical Officers ([NASA standard](https://www.nasa.gov/reference/3-0-health-and-medical-care/)); Google says MedGemma output is preliminary and requires independent verification ([model card](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card)).
- Personal baselines and percentage bands are prototype logic, not clinical limits.
- Crew accounts are for a local prototype. Tokens are hashed but do not expire and there is no account recovery.
- Do not expose Flask’s development server publicly. No production security review has been done.
- The demo includes sample labels. Only measurements saved after sign-in belong to the crew member’s database history.
