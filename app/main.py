from fastapi import FastAPI, UploadFile, File, Form
from fastapi import HTTPException
from pydantic import BaseModel
from typing import Optional, List
from app.rag import ingest_text, ingest_lines, search
from app.settings import settings
from openai import OpenAI

app = FastAPI(debug=True)

client = OpenAI(api_key=settings.openai_api_key)

class AskRequest(BaseModel):
    question: str
    top_k: int = 5

class IngestTextRequest(BaseModel):
    text: str
    doc_id: Optional[str] = "doc"

@app.get("/health")
def health():
    return {"status": "ok", "model": settings.openai_model}

@app.post("/ingest/text")
def ingest_text_endpoint(req: IngestTextRequest):
    try:
        if not req.text.strip():
            raise HTTPException(status_code=400, detail="text is empty")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{type(e).__name__}: {e}")
    info = ingest_text(doc_id_prefix=req.doc_id, text=req.text)
    return {"ok": True, **info}

@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...), doc_id: Optional[str] = Form("file")):
    content = (await file.read()).decode("utf-8", errors="ignore")
    lines = content.splitlines()
    info = ingest_lines(doc_id_prefix=doc_id, lines=lines, metadata={"filename": file.filename})
    return {"ok": True, **info}

@app.post("/ask")
def ask(req: AskRequest):
    # 1) retrieve
    hits = search(req.question, top_k=req.top_k)

    # 2) build prompt
    context_blocks = "\n\n".join(
        [f"[{i+1}] {h['text']}" for i, h in enumerate(hits)]
    )
    system_msg = (
        "You are a helpful assistant. Answer strictly based on the provided CONTEXT.\n"
        "If the answer is not in the context, say you don't know.\n"
        "Cite snippets like [1], [2] referring to the context block indices."
    )
    user_msg = f"QUESTION: {req.question}\n\nCONTEXT:\n{context_blocks}"

    # 3) call LLM
    resp = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )

    answer = resp.choices[0].message.content
    return {
        "answer": answer,
        "hits": hits
    }
