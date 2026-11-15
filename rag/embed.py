"""Local Ollama embedding client."""
import ollama
from config.settings import EMBED_MODEL

def embed(texts: list[str]) -> list[list[float]]:
    response = ollama.embed(model=EMBED_MODEL, input=texts, options={"num_gpu":0})
    return response["embeddings"]
