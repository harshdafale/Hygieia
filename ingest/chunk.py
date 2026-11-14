"""Sentence-preserving approximate token chunker."""
import re
from config.settings import CHUNK_WORDS, OVERLAP_WORDS

def chunk_sections(sections:list[dict],title:str)->list[dict]:
    result=[]
    for sec in sections:
        sentences=re.split(r"(?<=[.!?])\s+",sec["text"].strip())
        batch=[]; count=0
        for sentence in sentences:
            words=sentence.split()
            if not words: continue
            if count+len(words)>CHUNK_WORDS and batch:
                result.append({"section":sec.get("section","Document"),"page":sec.get("page"),"text":f"{title} > {sec.get('section','Document')}\n"+" ".join(batch)})
                tail=" ".join(batch).split()[-OVERLAP_WORDS:]; batch=[" ".join(tail)] if tail else []; count=len(tail)
            batch.append(sentence); count+=len(words)
        if batch: result.append({"section":sec.get("section","Document"),"page":sec.get("page"),"text":f"{title} > {sec.get('section','Document')}\n"+" ".join(batch)})
    return result
