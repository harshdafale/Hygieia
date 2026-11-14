"""Compact literal generation prompt."""
def build_prompt(question: str, sources: list[dict]) -> str:
    context = "\n".join(f"[{i}] {s['text']}" for i,s in enumerate(sources,1))
    return ("You are a space medicine assistant. Answer ONLY using the numbered context below. Cite sources like [1]. "
            "If the context does not answer the question, say exactly: Not covered in the provided NASA sources. "
            "Be brief and calm. Do not give doses or treatments unless they appear in the context.\n"
            f"Context: {context}\nQuestion: {question}\nAnswer:")
