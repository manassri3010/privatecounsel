"""The demo page and the document endpoint its citation links open."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.main
from fakes import FakeEmbedder, add_document, make_pdf


@pytest.fixture
def client(conn):
    app.main.app.dependency_overrides[app.main.get_db] = lambda: conn
    yield TestClient(app.main.app)
    app.main.app.dependency_overrides.clear()


def test_page_has_the_question_form():
    r = TestClient(app.main.app).get("/")

    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert '<form id="ask">' in r.text and 'id="matter"' in r.text and 'id="question"' in r.text


def test_page_never_inserts_untrusted_text_as_html():
    page = (Path(app.main.STATIC) / "index.html").read_text()

    assert "innerHTML" not in page and "insertAdjacentHTML" not in page and "document.write" not in page


def test_document_opens_inline_for_its_own_matter(client, conn, tmp_path):
    pdf = make_pdf(tmp_path / "act.pdf", ["10. What agreements are contracts."])
    document_id = add_document(conn, "client-a", "The Indian Contract Act, 1872",
                               [("text", 1, FakeEmbedder().vector("x"))], path=str(pdf))

    r = client.get(f"/matters/client-a/documents/{document_id}/file")

    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.headers["content-disposition"].startswith("inline")
    assert r.content == pdf.read_bytes()


def test_document_is_not_served_under_another_matter(client, conn, tmp_path):
    pdf = make_pdf(tmp_path / "secret.pdf", ["Confidential."])
    document_id = add_document(conn, "client-a", "Secret", [("text", 1, FakeEmbedder().vector("x"))], path=str(pdf))

    assert client.get(f"/matters/client-b/documents/{document_id}/file").status_code == 404


def test_missing_document_or_file_is_404(client, conn, tmp_path):
    document_id = add_document(conn, "client-a", "Gone", [("text", 1, FakeEmbedder().vector("x"))],
                               path=str(tmp_path / "deleted.pdf"))

    assert client.get(f"/matters/client-a/documents/{document_id}/file").status_code == 404
    assert client.get("/matters/client-a/documents/999999/file").status_code == 404
