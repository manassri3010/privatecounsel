import json
import re

import httpx
import pymupdf
import pytest

import fetch_corpus


def make_pdf(path, pages):
    with pymupdf.open() as doc:
        for text in pages:
            doc.new_page().insert_text((72, 72), text)
        doc.save(path)


def pdf_bytes(text="Section 1. Short title."):
    with pymupdf.open() as doc:
        doc.new_page().insert_text((72, 72), text)
        return doc.tobytes()


def mock_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_sources_file_is_complete_and_consistent():
    sources = fetch_corpus.load_sources()
    docs = sources["document"]
    ids = [d["id"] for d in docs] + [s["id"] for s in sources["scan"]]
    assert len(ids) == len(set(ids))
    for doc in docs:
        assert re.fullmatch(r"[a-z0-9-]+", doc["id"]), doc["id"]  # used as the file name
        assert doc["url"].startswith("https://")
        assert doc["licence"] in sources["licences"]
        assert doc["kind"] in {"act", "judgment"}
    for scan in sources["scan"]:
        assert scan["from"] in {d["id"] for d in docs}
        assert 1 <= scan["pages"][0] <= scan["pages"][1]


def test_scan_keeps_the_page_range_and_drops_the_text_layer(tmp_path):
    src, dest = tmp_path / "act.pdf", tmp_path / "scan.pdf"
    make_pdf(src, ["page one", "page two", "page three"])

    fetch_corpus.make_scan(src, dest, 2, 3, dpi=72, skew=1.0)

    with pymupdf.open(dest) as scan:
        assert scan.page_count == 2
        assert all(page.get_text().strip() == "" for page in scan)
        assert all(page.get_images() for page in scan)


def test_download_retries_server_errors(tmp_path):
    responses = iter([httpx.Response(504), httpx.Response(200, content=pdf_bytes())])
    waits = []

    fetch_corpus.download("https://example.test/a.pdf", tmp_path / "a.pdf",
                          mock_client(lambda request: next(responses)), sleep=waits.append)

    assert (tmp_path / "a.pdf").read_bytes().startswith(b"%PDF")
    assert waits == [fetch_corpus.RETRY_WAITS[0]]


def test_download_does_not_retry_client_errors_or_save_non_pdfs(tmp_path):
    calls = []

    def not_found(request):
        calls.append(request)
        return httpx.Response(404)

    with pytest.raises(httpx.HTTPStatusError):
        fetch_corpus.download("https://example.test/a.pdf", tmp_path / "a.pdf", mock_client(not_found),
                              sleep=lambda s: None)
    assert len(calls) == 1

    html = mock_client(lambda request: httpx.Response(200, text="<html>error page</html>"))
    with pytest.raises(ValueError, match="not a PDF"):
        fetch_corpus.download("https://example.test/a.pdf", tmp_path / "a.pdf", html)
    assert not (tmp_path / "a.pdf").exists()


SOURCES = """
[licences]
open = "Test licence text."

[[document]]
id = "act-one"
title = "The Act One, 2000"
kind = "act"
url = "https://example.test/act-one.pdf"
licence = "open"

[[document]]
id = "act-two"
title = "The Act Two, 2001"
kind = "act"
url = "https://example.test/act-two.pdf"
licence = "open"

[[scan]]
id = "scan-act-one"
from = "act-one"
pages = [1, 1]
dpi = 72
skew = 0.0

[[scan]]
id = "scan-act-two"
from = "act-two"
pages = [1, 1]
dpi = 72
skew = 0.0
"""


def test_main_writes_manifest_skips_failures_and_only_fetches_whats_missing(tmp_path):
    sources = tmp_path / "sources.toml"
    sources.write_text(SOURCES)
    out = tmp_path / "corpus"
    requested = []
    act_two_up = False

    def server(request):
        requested.append(request.url.path)
        if request.url.path == "/act-two.pdf" and not act_two_up:
            return httpx.Response(404)
        return httpx.Response(200, content=pdf_bytes())

    def run():
        return fetch_corpus.main(["--out", str(out)], client=mock_client(server), sources_path=sources,
                                 sleep=lambda s: None)

    assert run() == 1  # act-two failed, so its scan is skipped too
    manifest = json.loads((out / "manifest.json").read_text())["documents"]
    assert [e["id"] for e in manifest] == ["act-one", "scan-act-one"]
    act, scan = manifest
    assert act["source_url"] == "https://example.test/act-one.pdf"
    assert act["licence"] == "Test licence text."
    assert act["pages"] == 1 and act["text_layer"] is True and len(act["sha256"]) == 64
    assert scan["derived_from"] == "act-one" and scan["text_layer"] is False

    requested.clear()
    act_two_up = True
    assert run() == 0
    assert requested == ["/act-two.pdf"]  # act-one was already downloaded
    manifest = json.loads((out / "manifest.json").read_text())["documents"]
    assert [e["id"] for e in manifest] == ["act-one", "act-two", "scan-act-one", "scan-act-two"]
