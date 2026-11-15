"""Local Ollama language-model generation."""
import ollama
from config.settings import LLM_MODEL

def generate(prompt: str) -> str:
    # Ollama's default generation limit is intentionally short. Give the local
    # assistant enough room to finish organized plans and complete JSON assessments.
    result = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role":"user", "content":prompt}],
        options={"temperature":0.25, "num_ctx":4096, "num_predict":512},
    )
    return result["message"]["content"].strip()
