# PrivateCounsel

**Ask questions across your firm's case files and get answers with page citations — on a server you control.**

PrivateCounsel is a self-hosted AI document assistant for Indian law firms. It runs an open-weight LLM with [vLLM](https://docs.vllm.ai/) on your own GPU (or a GPU in an Indian data centre), so confidential client documents never go to a third-party AI provider.

> Status: **Phase 0: setup** (Nov 2026). See the [roadmap](#roadmap).

---

## Architecture

```
Web app ──► RAG API (FastAPI) ──► vLLM server (open-weight LLM)
   │              │
   ▼              ▼
Ingestion ──► PostgreSQL + pgvector        Monitoring: Prometheus + Grafana
(parse, OCR, chunk, embed)
```

- **Ingestion:** parse PDFs, OCR scanned pages, split into chunks with page numbers, embed, store.
- **Query:** check permissions → embed question → hybrid search (vector + BM25) → rerank → build prompt → vLLM → answer with citations.
- **Everything runs inside the client's environment.**

## Results

_Filled in as phases complete._

| Metric | Value | Setup |
| --- | --- | --- |
| Answer accuracy (50-question eval) | TBD | |
| Citation accuracy | TBD | |
| Time to first token (p50 / p95) | TBD | |
| Output tokens/s per user | TBD | |
| Cost per million tokens | TBD | |

## Quickstart

> Requires a machine with an NVIDIA GPU (24 GB+ VRAM), Docker, and the NVIDIA Container Toolkit.

```bash
git clone https://github.com/manassri3010/privatecounsel.git
cd privatecounsel
cp .env.example .env        # set API keys and model name
docker compose up -d
curl http://localhost:8080/health
```

### Local development (no GPU)

Uses [Ollama](https://ollama.com/) instead of vLLM. Both expose an OpenAI-compatible API, so the code is the same.

```bash
ollama pull qwen2.5:3b
brew install tesseract      # OCR for scanned PDFs
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env        # switch to the "API run directly with uvicorn" lines
uvicorn app.main:app --app-dir api --port 8080
```

Run the checks CI runs with `ruff check .` and `pytest`.

### Test corpus

Development and evals use only public legal texts: bare acts from [India Code](https://www.indiacode.nic.in/) and Supreme Court judgments from the [Indian Supreme Court Judgments](https://registry.opendata.aws/indian-supreme-court-judgments/) open dataset (CC-BY-4.0). No client documents.

```bash
python corpus/fetch_corpus.py    # downloads into data/corpus/ (gitignored); re-run to fetch anything missing
```

The list of documents and their licences lives in `corpus/sources.toml`. The script also makes image-only "scanned" copies of a few documents for OCR testing, and writes `data/corpus/manifest.json` with each file's source URL, licence, SHA-256 checksum and page count.

Embeddings use `BAAI/bge-base-en-v1.5` (decision 005), downloaded from Hugging Face on first use (about 440 MB) and run locally.

## Repository layout

| Folder | What's inside |
| --- | --- |
| `api/` | FastAPI service: query pipeline, auth, citations |
| `ingestion/` | Parsing, OCR, chunking, embedding |
| `corpus/` | List of public test documents and the script that downloads them |
| `evals/` | Test questions and the scoring script |
| `benchmarks/` | Inference benchmarks (TTFT, ITL, throughput, cost) |
| `docs/` | Architecture, design decisions, security notes |
| `tests/` | Automated tests run by CI |

## Roadmap

- [ ] **Phase 0** (Nov 23 – Dec 6): setup and foundations
- [ ] **Phase 1** (Dec 7 – 20): inference lab: serve, benchmark, quantize
- [ ] **Phase 2** (Dec 21 – Jan 17): RAG core: ingestion, hybrid search, citations → v0.1
- [ ] **Phase 3** (Jan 18 – 31): evals and hardening
- [ ] **Phase 4** (Feb 1 – 21): productise: UI, auth, one-command deploy → v1.0
- [ ] **Phase 5** (Feb 22 – Mar 21): pilot with a firm

## Design decisions

Recorded in [`docs/decisions.md`](docs/decisions.md).

## License

Apache 2.0. See [LICENSE](LICENSE).
