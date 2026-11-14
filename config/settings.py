"""Central configuration for AstroRAG."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
DB_DIR = ROOT / "db"
SQLITE_PATH = DB_DIR / "astrorag.sqlite"
CHROMA_DIR = DB_DIR / "chroma"
SCHEMA_PATH = DB_DIR / "schema.sql"
LLM_MODEL = "gemma:2b"
EMBED_MODEL = "nomic-embed-text"
CHUNK_WORDS = 400
OVERLAP_WORDS = 50
TOP_K = 4
MAX_DIST = float(os.getenv("ASTRORAG_MAX_DIST", "0.45"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
WEB_ORIGIN = "http://localhost:5173"
