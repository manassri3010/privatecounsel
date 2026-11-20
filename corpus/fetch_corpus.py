"""Download the public test corpus listed in corpus/sources.toml.

Usage: python corpus/fetch_corpus.py [--out data/corpus]

Downloads one file at a time (government servers are slow and shouldn't be hammered), retries
server errors, skips files that are already there, makes the image-only "scanned" copies used to
test OCR, and writes manifest.json recording where every file came from and under what licence.
Re-running it only fetches what is missing.
"""

import argparse
import hashlib
import json
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "corpus" / "sources.toml"
DEFAULT_OUT = ROOT / "data" / "corpus"
RETRY_WAITS = (10, 30)  # seconds between attempts; India Code often returns 504 for a while


def load_sources(path=SOURCES):
    return tomllib.loads(Path(path).read_text())


def download(url, dest, client, waits=RETRY_WAITS, sleep=time.sleep):
    """Save url to dest. Retries 5xx and network errors; a 4xx or a non-PDF fails at once."""
    for attempt in range(len(waits) + 1):
        try:
            response = client.get(url)
            response.raise_for_status()
            break
        except httpx.HTTPError as e:
            client_error = isinstance(e, httpx.HTTPStatusError) and e.response.status_code < 500
            if client_error or attempt == len(waits):
                raise
            print(f"    {str(e).splitlines()[0]}; retrying in {waits[attempt]}s")
            sleep(waits[attempt])
    if not response.content.startswith(b"%PDF"):
        raise ValueError(f"not a PDF (content-type {response.headers.get('content-type')})")
    # Write to a temporary name first so an interrupted run never leaves a half-written PDF behind.
    partial = dest.with_suffix(".part")
    partial.write_bytes(response.content)
    partial.rename(dest)


def make_scan(src, dest, first, last, dpi=200, skew=0.0):
    """Image-only copy of pages first..last (1-based): like a scanner's output, with no text layer."""
    with pymupdf.open(src) as doc, pymupdf.open() as out:
        for page in doc.pages(first - 1, last):
            matrix = pymupdf.Matrix(dpi / 72, dpi / 72).prerotate(skew)
            pix = page.get_pixmap(matrix=matrix, colorspace=pymupdf.csGRAY)
            new_page = out.new_page(width=pix.width * 72 / dpi, height=pix.height * 72 / dpi)
            new_page.insert_image(new_page.rect, pixmap=pix)
        out.save(dest, garbage=3, deflate=True)


def describe(path):
    """Checksum, size, page count and whether any page has extractable text."""
    data = path.read_bytes()
    with pymupdf.open(path) as doc:
        pages = doc.page_count
        text_layer = any(page.get_text().strip() for page in doc)
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "pages": pages,
            "text_layer": text_layer}


def main(argv=None, client=None, sources_path=SOURCES, sleep=time.sleep):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    sources = load_sources(sources_path)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    previous = {}
    if manifest_path.exists():
        previous = {e["id"]: e for e in json.loads(manifest_path.read_text())["documents"]}

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    client = client or httpx.Client(timeout=120, follow_redirects=True)
    entries, failed = [], []

    for doc in sources["document"]:
        dest = out / f"{doc['id']}.pdf"
        if dest.exists():
            print(f"have  {doc['id']}")
        else:
            print(f"fetch {doc['id']}")
            try:
                download(doc["url"], dest, client, sleep=sleep)
            except (httpx.HTTPError, ValueError) as e:
                print(f"    FAILED: {str(e).splitlines()[0]}")
                failed.append(doc["id"])
                continue
        entries.append({
            "id": doc["id"], "title": doc["title"], "kind": doc["kind"], "file": dest.name,
            "source_url": doc["url"], "licence": sources["licences"][doc["licence"]],
            "retrieved_at": previous.get(doc["id"], {}).get("retrieved_at", now),
            **describe(dest),
        })

    titles = {doc["id"]: doc["title"] for doc in sources["document"]}
    by_id = {e["id"]: e for e in entries}
    for scan in sources.get("scan", []):
        original = by_id.get(scan["from"])
        if original is None:
            print(f"skip  {scan['id']} (source {scan['from']} not downloaded)")
            failed.append(scan["id"])
            continue
        dest = out / f"{scan['id']}.pdf"
        if not dest.exists():
            print(f"scan  {scan['id']}")
            make_scan(out / original["file"], dest, *scan["pages"], dpi=scan["dpi"], skew=scan["skew"])
        first, last = scan["pages"]
        entries.append({
            "id": scan["id"], "title": f"{titles[scan['from']]} [scanned copy, pages {first}-{last}]",
            "kind": "scan", "file": dest.name, "derived_from": scan["from"],
            "derivation": f"pages {first}-{last} rendered at {scan['dpi']} dpi, greyscale, "
                          f"rotated {scan['skew']} degrees, text layer removed",
            "source_url": original["source_url"], "licence": original["licence"],
            "retrieved_at": original["retrieved_at"],
            **describe(dest),
        })

    manifest_path.write_text(json.dumps({"documents": entries}, indent=2, ensure_ascii=False) + "\n")
    print(f"\n{len(entries)} files in {out}, manifest: {manifest_path}")
    if failed:
        print(f"{len(failed)} not available ({', '.join(failed)}); re-run later to retry.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
