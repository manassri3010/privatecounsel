"""PrivateCounsel API.

/ask answers a question from one matter's documents only, with citations to document and page.
"""
from functools import lru_cache

from fastapi import Depends, FastAPI
from openai import OpenAI
from pydantic import BaseModel

from ingestion import db
from retrieval.answer import answer_question

from .config import settings

app = FastAPI(title="PrivateCounsel API", version="0.1.0")
llm = OpenAI(base_url=settings.llm_base_url, api_key=settings.vllm_api_key)


def get_db():
    """One database connection per request (a connection pool comes with the Phase 4 deployment)."""
    conn = db.connect(settings.database_url or None)
    try:
        yield conn
    finally:
        conn.close()


@lru_cache
def get_embedder():
    """The embedding model, loaded on the first question rather than at startup."""
    from ingestion.embed import Embedder

    return Embedder()


class AskRequest(BaseModel):
    question: str
    # No permission check yet: login and per-matter access control come in Phase 4.
    matter_id: str


class Citation(BaseModel):
    number: int  # the [n] used in the answer
    document_id: int
    title: str
    heading: str
    page_start: int
    page_end: int


class AskResponse(BaseModel):
    answer: str
    found: bool  # False when the matter's documents don't answer the question
    cited: bool  # False when the answer cites no source, so it can't be checked
    citations: list[Citation]
    model: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": settings.llm_model}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, conn=Depends(get_db), embedder=Depends(get_embedder)) -> AskResponse:
    result = answer_question(conn, embedder, llm, settings.llm_model, req.question, req.matter_id)
    return AskResponse(
        answer=result.text,
        found=result.found,
        cited=result.cited,
        citations=[Citation(**vars(c)) for c in result.citations],
        model=settings.llm_model,
    )
