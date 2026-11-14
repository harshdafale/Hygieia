"""Local Ollama embedding client."""
import ollama
from config.settings import EMBED_MODEL

def embed(texts: list[str]) -> list[list[float]]:
    response = ollama.embed(model=EMBED_MODEL, input=texts)
    return response["embeddings"]
