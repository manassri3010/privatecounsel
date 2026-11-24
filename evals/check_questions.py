"""Check the evaluation questions against the corpus PDFs.

Usage: python -m evals.check_questions

For every source it confirms the quote appears on each listed page, so a wrong page number is caught
here instead of silently lowering the scores. Needs the corpus (python corpus/fetch_corpus.py).
Matching ignores spacing, case and ligatures, which PDF text extraction mangles ("ﬁ led").
"""

import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pymupdf

from evals.dataset import corpus_ids, load_questions, problems

CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus"


def normalise(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).lower()


def page_texts(corpus=CORPUS):
    files = {e["id"]: e["file"] for e in json.loads((corpus / "manifest.json").read_text())["documents"]}
    return {doc_id: [normalise(page.get_text()) for page in pymupdf.open(corpus / name)]
            for doc_id, name in files.items()}


def quote_problems(questions, texts):
    found = []
    for q in questions:
        for s in q.sources + q.also_acceptable:
            pages = texts.get(s.document)
            if pages is None:
                found.append(f"{q.id}: {s.document} is not downloaded")
                continue
            quote = normalise(s.quote)
            missing = [p for p in s.pages if p > len(pages) or quote not in pages[p - 1]]
            if missing:
                actual = [i + 1 for i, t in enumerate(pages) if quote in t]
                found.append(f"{q.id}: quote not on page(s) {missing} of {s.document} (found on {actual}): {s.quote!r}")
    return found


def main():
    questions = load_questions()
    issues = problems(questions, corpus_ids()) + quote_problems(questions, page_texts())
    counts = Counter(q.type for q in questions)
    print(f"{len(questions)} questions: " + ", ".join(f"{counts[t]} {t}" for t in ("single", "multi", "unanswerable")))
    for issue in issues:
        print("  " + issue)
    print("All sources check out." if not issues else f"{len(issues)} problem(s).")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
