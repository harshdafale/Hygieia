"""Evidence gates and demo severity rules."""
import re
from config.settings import MAX_DIST

REFUSAL = "No supporting NASA source found for this question."

def gate_one(sources: list[dict]) -> bool:
    return bool(sources) and float(sources[0]["distance"]) <= MAX_DIST

def validate(answer: str, sources: list[dict]) -> tuple[str,bool]:
    citations = [int(n) for n in re.findall(r"\[(\d+)\]", answer)]
    if not citations or any(n < 1 or n > len(sources) for n in citations):
        return "Showing the most relevant NASA passages instead:\n\n" + "\n\n".join(f"[{i}] {s['text']}" for i,s in enumerate(sources,1)), True
    return answer, False

def severity(question: str) -> str:
    q = question.lower()
    if any(x in q for x in ("chest pain","loss of consciousness","severe bleeding")): return "escalate"
    if any(x in q for x in ("persistent","worsening")): return "monitor"
    return "info"
