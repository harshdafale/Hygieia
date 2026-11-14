"""Local Gemma generation."""
import ollama
from config.settings import LLM_MODEL

def generate(prompt: str) -> str:
    result = ollama.chat(model=LLM_MODEL, messages=[{"role":"user","content":prompt}], options={"temperature":0.1,"num_ctx":4096})
    return result["message"]["content"].strip()
