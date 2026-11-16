from types import SimpleNamespace

from fastapi.testclient import TestClient

import app.main
from app.config import settings


class FakeCompletions:
    """Stands in for llm.chat.completions so tests run without a model server."""

    def __init__(self, answer: str | None):
        self.answer = answer
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.answer)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def fake_llm(monkeypatch, answer: str | None) -> FakeCompletions:
    completions = FakeCompletions(answer)
    monkeypatch.setattr(app.main, "llm", SimpleNamespace(chat=SimpleNamespace(completions=completions)))
    return completions


def test_ask_returns_answer_and_model(monkeypatch):
    fake_llm(monkeypatch, "Section 103 BNS covers murder.")
    client = TestClient(app.main.app)
    r = client.post("/ask", json={"question": "What covers murder?"})
    assert r.status_code == 200
    assert r.json() == {"answer": "Section 103 BNS covers murder.", "model": settings.llm_model}


def test_ask_sends_system_prompt_and_question(monkeypatch):
    completions = fake_llm(monkeypatch, "ok")
    TestClient(app.main.app).post("/ask", json={"question": "What covers murder?"})
    [call] = completions.calls
    assert call["model"] == settings.llm_model
    system, user = call["messages"]
    assert system["role"] == "system"
    assert "Indian law" in system["content"]
    assert user == {"role": "user", "content": "What covers murder?"}


def test_ask_handles_empty_model_reply(monkeypatch):
    fake_llm(monkeypatch, None)
    r = TestClient(app.main.app).post("/ask", json={"question": "Anything?"})
    assert r.json()["answer"] == ""


def test_ask_rejects_missing_question():
    r = TestClient(app.main.app).post("/ask", json={})
    assert r.status_code == 422
