# AstroRAG

AstroRAG is an offline-first space medicine retrieval assistant prototype. It answers from locally ingested NASA documents and reports when evidence is missing. It is an information-retrieval aid, not a diagnostic tool. The severity keyword list is a demo and is not clinical guidance. Use simulated crew data only; never store real personal health data.

## Architecture

```text
NASA PDF / HTML / TXT -> parse -> heading-aware chunks -> Ollama embeddings -> ChromaDB
                                                        -> SQLite metadata/audit
React UI -> FastAPI -> dense retrieval -> evidence gate -> Ollama gemma:2b
```

## Run locally

Requirements: Python 3.10+, Node.js, and Ollama. No cloud APIs or telemetry are used at runtime.

1. `ollama pull gemma:2b`
2. `ollama pull nomic-embed-text`
3. `pip install -r requirements.txt`
4. Put NASA source documents in `data/raw/` (PDF, HTML, or TXT).
5. `python -m ingest.ingest --all`
6. `uvicorn api.main:app --reload`
7. In another terminal: `cd web`, `npm install`, `npm run dev`.

Set `MAX_DIST` in `config/settings.py` for your collection. Run `python -m eval.calibrate` to inspect answerable and unanswerable retrieval distances. `python -m eval.run_eval` runs the sample retrieval/refusal evaluation. `config/sources.yaml` is an optional empty URL list for sources you provide yourself.

## Limitations and future work

The small model can produce weak answers; responses are restricted to ingested documents and citation checks can fall back to source passages. Retrieval thresholds require calibration for each corpus. This is not clinical advice. Future work could evaluate hybrid retrieval, reranking, larger local models, TMR/post-quantum integrity for telemetry, and pgvector.

## Multi-crew local accounts

Run the Flask API with python -m flask --app api.flask_app run --host 127.0.0.1 --port 5000. The first visit shows a crew picker and registration form. Each crew member signs in with a PIN; only a one-way PIN hash is stored. Requests use a random bearer token and derive the owner ID on the server. Vitals, diagnostics, reports, advisories, and copilot context are stored in SQLite and filtered to that crew member. Personal baselines use that crew member's preceding 14 days of records; charts load their preceding 7 days.

For local development, run the Vite site on port 5173 and the Flask API on port 5000. Crew data is simulated only; do not store real personal health data.
