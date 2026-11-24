"""Run the evaluation questions through the system and score them.

Usage: python -m evals.run [--matter public-corpus] [--retrieval-only] [--ids id1,id2] [--out evals/results]

Uses the same pipeline as /ask (hybrid search, the answer prompt, citation checks) and the LLM server
configured in .env (LLM_BASE_URL, LLM_MODEL, VLLM_API_KEY): Ollama locally, vLLM on a GPU. Prints the
score table and saves every answer, citation and score with the settings used, as
evals/results/<time>_<model>.json, so runs can be compared.
--retrieval-only skips the model: seconds instead of minutes, for tuning search.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI
from pydantic_settings import BaseSettings, SettingsConfigDict

from evals.dataset import QUESTIONS, load_questions
from evals.score import format_summary, score_question, summarise
from ingestion import db
from ingestion.chunk import CHUNKER_VERSION
from ingestion.embed import CHUNK_TOKENS
from retrieval import answer as answering
from retrieval import search as searching

RESULTS = Path(__file__).with_name("results")


class LLMSettings(BaseSettings):
    # Same .env and variable names as the API (api/app/config.py).
    model_config = SettingsConfigDict(env_file=db.ROOT / ".env", extra="ignore")

    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen2.5:3b"
    vllm_api_key: str = "not-needed-for-ollama"


def git_commit():
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                check=True, cwd=db.ROOT).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                               text=True, cwd=db.ROOT).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run(questions, conn, embedder, llm, model, matter):
    keys = dict(conn.execute("SELECT id, source_key FROM documents WHERE matter_id = %s", (matter,)).fetchall())
    records = []
    for n, q in enumerate(questions, 1):
        started = time.time()
        hits = searching.search(conn, embedder, q.question, matter, k=20)
        retrieved = [(keys[h.document_id], h.page_start, h.page_end) for h in hits]
        answer = None
        if llm is not None:
            a = answering.answer_question(conn, embedder, llm, model, q.question, matter)
            answer = {"found": a.found, "text": a.text, "removed_citations": a.removed_citations,
                      "citations": [(keys[c.document_id], c.page_start, c.page_end) for c in a.citations]}
        scores = score_question(q, retrieved, answer)
        records.append({"question": q.question, "scores": scores, "answer": answer,
                        "retrieved": retrieved, "seconds": round(time.time() - started, 1)})
        print(f"[{n:2}/{len(questions)}] {_one_line(scores)}  {q.id}", flush=True)
    return records


def _one_line(scores):
    if scores["type"] == "unanswerable":
        return "not found ✓" if scores.get("correct_not_found") else ("answered  ✗" if scores["found"] else "-")
    parts = [f"ret@5 {scores['retrieval@5']:.0%}"]
    if "key_facts" in scores:
        parts.append(f"facts {scores['key_facts']:.0%}")
        parts.append(f"cites {scores['citation_precision']:.0%}" if scores["cited"] else "uncited")
    elif scores["found"] is False:
        parts.append("refused")
    return " | ".join(parts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matter", default="public-corpus")
    parser.add_argument("--retrieval-only", action="store_true", help="score retrieval without calling the model")
    parser.add_argument("--ids", help="comma-separated question ids, to run a subset")
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args(argv)

    questions = load_questions()
    if args.ids:
        wanted = set(args.ids.split(","))
        questions = [q for q in questions if q.id in wanted]
    settings = LLMSettings()
    llm = None if args.retrieval_only else OpenAI(base_url=settings.llm_base_url, api_key=settings.vllm_api_key)
    model = "none (retrieval only)" if args.retrieval_only else settings.llm_model

    from ingestion.embed import Embedder  # loads the model

    embedder = Embedder()
    started = datetime.now(timezone.utc)
    records = run(questions, db.connect(), embedder, llm, settings.llm_model, args.matter)
    summary = summarise([r["scores"] for r in records])
    print(f"\n{format_summary(summary)}")

    result = {
        "summary": summary,
        "settings": {
            "started": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "seconds": round((datetime.now(timezone.utc) - started).total_seconds()),
            "git_commit": git_commit(),
            "model": model,
            "llm_base_url": None if args.retrieval_only else settings.llm_base_url,
            "embedding_model": embedder.name,
            "chunking": f"v{CHUNKER_VERSION}/{CHUNK_TOKENS} tokens",
            "top_k": answering.TOP_K, "candidates": searching.CANDIDATES, "rrf_k": searching.RRF_K,
            "matter": args.matter,
            "questions": len(questions),
            "questions_sha256": hashlib.sha256(QUESTIONS.read_bytes()).hexdigest()[:16],
        },
        "questions": records,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    name = f"{started:%Y%m%dT%H%M%SZ}_{re.sub(r'[^A-Za-z0-9.]+', '-', model).strip('-')}.json"
    (args.out / name).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"\nSaved {args.out / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
