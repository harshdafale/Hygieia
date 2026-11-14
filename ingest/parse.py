"""Extract text from PDF, HTML, and plain text sources."""
from pathlib import Path
from bs4 import BeautifulSoup

def parse(path:Path)->list[dict]:
    suffix=path.suffix.lower()
    if suffix==".pdf":
        import fitz
        doc=fitz.open(path); out=[]
        for n,page in enumerate(doc,1):
            text=page.get_text("text").strip()
            if text: out.append({"text":text,"page":n,"section":f"Page {n}"})
        return out
    raw=path.read_text(encoding="utf-8",errors="ignore")
    if suffix in (".html",".htm"):
        soup=BeautifulSoup(raw,"html.parser")
        title=soup.title.get_text(" ",strip=True) if soup.title else path.stem
        parts=[]
        for node in soup.select("h1,h2,h3,p,li"):
            text=node.get_text(" ",strip=True)
            if text: parts.append({"text":text,"page":None,"section":title if node.name.startswith("h") else "Document"})
        return parts
    return [{"text":raw,"page":None,"section":"Document"}]
