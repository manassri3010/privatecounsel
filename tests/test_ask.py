"""/ask end to end: real hybrid search on PostgreSQL (see conftest.py), fake embedder and fake LLM."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app.main
from app.config import settings
from fakes import FakeEmbedder, add_document


class FakeCompletions:
    """Stands in for llm.chat.completions so tests run without a model server."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply))])


@pytest.fixture
def client(conn, monkeypatch):
    embedder = FakeEmbedder()
    app.main.app.dependency_overrides[app.main.get_db] = lambda: conn
    app.main.app.dependency_overrides[app.main.get_embedder] = lambda: embedder
    yield TestClient(app.main.app), embedder
    app.main.app.dependency_overrides.clear()


def fake_llm(monkeypatch, reply):
    completions = FakeCompletions(reply)
    monkeypatch.setattr(app.main, "llm", SimpleNamespace(chat=SimpleNamespace(completions=completions)))
    return completions


def test_answer_cites_the_matters_documents_by_page(client, conn, monkeypatch):
    test_client, embedder = client
    question = "Can a minor enter into a contract?"
    document_id = add_document(conn, "client-a", "The Indian Contract Act, 1872", [
        ("11. Who are competent to contract.—Every person of the age of majority.", 13, embedder.embed_query(question)),
    ])
    add_document(conn, "client-b", "Another client's contract", [("Secret terms.", 1, embedder.embed_query(question))])
    completions = fake_llm(monkeypatch, "Only a person of the age of majority can contract [1].")

    r = test_client.post("/ask", json={"question": question, "matter_id": "client-a"})

    assert r.status_code == 200
    assert r.json() == {
        "answer": "Only a person of the age of majority can contract [1].",
        "found": True,
        "cited": True,
        "citations": [{"number": 1, "document_id": document_id, "title": "The Indian Contract Act, 1872",
                       "heading": "HEADING", "page_start": 13, "page_end": 13}],
        "model": settings.llm_model,
    }
    [call] = completions.calls
    prompt = call["messages"][1]["content"]
    assert "age of majority" in prompt and "Secret terms" not in prompt  # only this matter's chunks
    assert call["temperature"] == 0


def test_not_found_when_the_model_says_so(client, conn, monkeypatch):
    test_client, embedder = client
    add_document(conn, "client-a", "Act", [("Unrelated text.", 1, embedder.vector("x"))])
    fake_llm(monkeypatch, "NOT_FOUND")

    body = test_client.post("/ask", json={"question": "Murder under the BNS?", "matter_id": "client-a"}).json()

    assert (body["found"], body["cited"], body["citations"]) == (False, False, [])


def test_matter_id_is_required(client):
    test_client, _ = client
    assert test_client.post("/ask", json={"question": "Anything?"}).status_code == 422


def test_health_reports_the_model():
    assert TestClient(app.main.app).get("/health").json() == {"status": "ok", "model": settings.llm_model}
