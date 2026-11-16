"""PrivateCounsel API.

Phase 0: a health check and a plain chat endpoint that talks to the LLM server.
Retrieval (RAG) and citations are added in Phase 2.
"""
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel

from .config import settings

app = FastAPI(title="PrivateCounsel API", version="0.0.1")
llm = OpenAI(base_url=settings.llm_base_url, api_key=settings.vllm_api_key)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    model: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": settings.llm_model}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    completion = llm.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": "You are a careful legal research assistant for Indian law."},
            {"role": "user", "content": req.question},
        ],
        temperature=0.2,
    )
    return AskResponse(answer=completion.choices[0].message.content or "", model=settings.llm_model)
