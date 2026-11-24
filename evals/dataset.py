"""Load and check the evaluation questions in evals/questions.toml.

Each question is "single" (answered by one source), "multi" (needs every listed source) or
"unanswerable" (nothing in the corpus answers it, so the right reply is "not found").
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path

QUESTIONS = Path(__file__).with_name("questions.toml")
SOURCES = Path(__file__).resolve().parents[1] / "corpus" / "sources.toml"
TYPES = {"single", "multi", "unanswerable"}


@dataclass(frozen=True)
class Source:
    document: str  # corpus id, as in corpus/sources.toml
    pages: tuple[int, ...]  # 1-based PDF pages where it is stated; any of them counts
    quote: str  # words found on every listed page, so the pages can be checked automatically


@dataclass(frozen=True)
class Question:
    id: str
    type: str
    question: str
    answer: str
    key_facts: tuple[tuple[str, ...], ...]  # each fact: alternative phrasings, any one must appear
    sources: tuple[Source, ...]  # all needed for "multi"; the one source for "single"
    also_acceptable: tuple[Source, ...]  # other passages that also answer it, for retrieval scoring


def _sources(entries):
    return tuple(Source(s["document"], tuple(s["pages"]), s["quote"]) for s in entries)


def load_questions(path=QUESTIONS):
    data = tomllib.loads(Path(path).read_text())
    return [Question(q["id"], q["type"], q["question"], q["answer"],
                     tuple(tuple(fact) for fact in q.get("key_facts", [])),
                     _sources(q.get("sources", [])), _sources(q.get("also_acceptable", [])))
            for q in data["question"]]


def problems(questions, corpus_ids):
    """Structural mistakes in the question set; an empty list means it is consistent."""
    found = []
    ids = [q.id for q in questions]
    found += [f"duplicate id {i}" for i in sorted({i for i in ids if ids.count(i) > 1})]
    for q in questions:
        where = f"{q.id}:"
        if q.type not in TYPES:
            found.append(f"{where} unknown type {q.type!r}")
        expected_sources = {"single": (1, 1), "multi": (2, 9), "unanswerable": (0, 0)}.get(q.type, (0, 99))
        if not expected_sources[0] <= len(q.sources) <= expected_sources[1]:
            found.append(f"{where} a {q.type} question has {len(q.sources)} sources")
        if q.type != "unanswerable" and not q.key_facts:
            found.append(f"{where} no key facts")
        if q.type == "unanswerable" and (q.key_facts or q.also_acceptable):
            found.append(f"{where} an unanswerable question has key facts or sources")
        if any(not alternatives or any(not a.strip() for a in alternatives) for alternatives in q.key_facts):
            found.append(f"{where} empty key fact")
        for s in q.sources + q.also_acceptable:
            if s.document not in corpus_ids:
                found.append(f"{where} unknown document {s.document!r}")
            if not s.pages or any(p < 1 for p in s.pages):
                found.append(f"{where} bad pages {s.pages} for {s.document}")
            if len(s.quote.split()) < 4:
                found.append(f"{where} quote too short to locate reliably: {s.quote!r}")
    return found


def corpus_ids(path=SOURCES):
    return {d["id"] for d in tomllib.loads(Path(path).read_text())["document"]}
