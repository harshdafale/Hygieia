"""SQLite helpers."""
import sqlite3
from config.settings import SQLITE_PATH, SCHEMA_PATH, DB_DIR

def connect():
    DB_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(SQLITE_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    return con

def init_db():
    with connect() as con:
        con.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))

def health():
    with connect() as con:
        return con.execute("SELECT 1").fetchone() is not None
