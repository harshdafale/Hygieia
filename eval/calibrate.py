"""Print retrieval distances for the example calibration set."""
import json
from pathlib import Path
from rag.retrieve import retrieve

def main():
    rows=[json.loads(x) for x in Path("eval/questions.jsonl").read_text().splitlines() if x.strip()]
    for row in rows:
        hits=retrieve(row["question"]); print(f"answerable={row['answerable']} best_distance={hits[0]['distance'] if hits else 'no results'} question={row['question']}")
    print("Choose MAX_DIST by comparing the answerable and unanswerable distance ranges.")

if __name__=="__main__": main()
