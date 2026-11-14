"""Retrieval/refusal evaluation over eval/questions.jsonl."""
import json
from pathlib import Path
from api.db import init_db,connect
from rag.retrieve import retrieve
from rag.guardrails import gate_one

def main():
    init_db(); rows=[json.loads(x) for x in Path("eval/questions.jsonl").read_text().splitlines() if x.strip()]
    hits=refusals=negatives=0
    for q in rows:
        found=retrieve(q["question"]); hit=any(q["expected_doc"] and q["expected_doc"].lower() in s.get("filename","").lower() for s in found)
        refuse=not gate_one(found)
        hits+=hit; refusals+=refuse== (not q["answerable"]); negatives+=not q["answerable"]
        print(f"{q['question'][:48]:48} hit={str(hit):5} refused={refuse}")
    with connect() as con: con.execute("INSERT INTO eval_runs(hit_rate,refusal_accuracy,notes) VALUES(?,?,?)",(hits/max(len(rows),1),refusals/max(len(rows),1),"sample questions"))
    print(f"hit rate {hits/max(len(rows),1):.0%} | refusal accuracy {refusals/max(len(rows),1):.0%} | citation rate: run via UI | fallback rate: run via UI")

if __name__=="__main__": main()
