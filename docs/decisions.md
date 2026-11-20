# Design decisions

One entry per decision: what, why, and what was rejected. Recruiters and clients read this.

## 001: Serve models with vLLM
- **Why:** industry-standard open-source engine; PagedAttention, continuous batching, prefix caching; OpenAI-compatible API.
- **Rejected:** Ollama for production (built for one user, not many concurrent users). Ollama is still used for local development.

## 002: PostgreSQL + pgvector for retrieval
- **Why:** one database for chunks, vectors, users and permissions; simpler to run and back up at a client site.
- **Rejected:** a separate vector database (extra service to operate for small firms).

## 003: Qwen3-8B as the default model
Checked on Hugging Face, 18 Nov 2026.

| | Qwen3-8B | Qwen3.5-9B | Llama 3.1 8B Instruct | Qwen2.5-7B-Instruct |
| --- | --- | --- | --- | --- |
| Released | Apr 2025 | Feb 2026 | Jul 2024 | Sep 2024 |
| Licence | Apache 2.0 | Apache 2.0 | Llama 3.1 Community | Apache 2.0 |
| Gated on Hugging Face | No | No | Yes (manual approval) | No |
| Official FP8 | Yes (Qwen) | No (Red Hat AI) | No (Red Hat AI) | No |
| Official AWQ | Yes (Qwen) | No (community only) | No (community) | Yes (Qwen) |
| Type | Text, dense | Text + vision, hybrid linear attention | Text, dense | Text, dense |
| Native context | 32K | 256K | 128K | 32K |

- **Why:**
  - **Licence:** Apache 2.0 lets us sell setup and hosting with no extra conditions. Llama's licence adds a "Built with Llama" notice, an acceptable-use policy and a 700M-monthly-user threshold.
  - **Easy setup:** the model isn't gated, so no `HF_TOKEN` or approval wait at a client site.
  - **Clean comparison:** Qwen publishes BF16, FP8 and AWQ versions of the same model. Phase 1 compares precisions without mixing in a third party's quantization choices.
  - **Simpler to benchmark:** a plain text-only transformer, supported in vLLM for over a year. The KV-cache and throughput numbers are easy to explain.
  - **Quality:** newer than Llama 3.1 8B and Qwen2.5-7B, and the Qwen3 card reports it beats Qwen2.5-Instruct in non-thinking mode. Our own 50-question eval in Phase 3 makes the final call.
- **Models to benchmark (Phase 1):**
  - BF16: `Qwen/Qwen3-8B`
  - FP8: `Qwen/Qwen3-8B-FP8`
  - AWQ (4-bit): `Qwen/Qwen3-8B-AWQ`
- **Caveats:**
  - **Thinking mode:** Qwen3 "thinks" by default. vLLM turns it off server-wide (`--default-chat-template-kwargs '{"enable_thinking": false}'` in `docker-compose.yml`), because long hidden reasoning is slower and isn't what we want for cited answers. A request can still opt in. This doesn't affect the throughput benchmarks, which use raw prompts.
  - **Context:** 32K native context is enough for RAG (about 5 chunks of 800 tokens plus the answer).
- **Rejected:**
  - **Llama 3.1 8B Instruct:** gated, restrictive licence terms, older, and no official quantizations.
  - **Qwen3.5-9B:** newest and probably the strongest, but it carries a vision encoder we don't need, uses a new hybrid attention design that makes benchmark results harder to compare, and has no official FP8 or AWQ. Re-evaluate in Phase 3 as a quality upgrade.
  - **Qwen2.5-7B-Instruct:** superseded by Qwen3-8B, and has no official FP8.

## 004: Pin vLLM to v0.32.0
- **Why:**
  - **Reproducible numbers:** benchmark results are only comparable if the serving engine is the same. `latest` changes every few weeks, and vLLM releases regularly change defaults that move TTFT and throughput (scheduler, CUDA graphs, KV-cache sizing).
  - **Stability:** v0.32.0 (18 Oct 2026) was the newest full release at least a few weeks old, with no open regressions affecting Qwen3 dense models. v0.34.0 was two days old and v0.33 had only release candidates.
  - **Recorded:** each benchmark run also records the vLLM version it ran on.
- **CUDA:** the default image needs NVIDIA driver 580+ (CUDA 13.0). `VLLM_TAG=v0.32.0-cu129` covers older drivers on rented nodes, and the tag is still pinned.
- **Upgrading:** an upgrade is its own PR, with a re-run of the Phase 1 benchmark to show the difference.
- **Rejected:** `latest` (results drift without notice); `v0.34.0` (too new to trust for a baseline).

## 005: BGE base v1.5 for embeddings, with chunks of at most 400 tokens
Measured on a MacBook Air (Apple M-series, 8 GB) with the 9-file test corpus, 20 Nov 2026.

| | bge-small-en-v1.5 | **bge-base-en-v1.5** | bge-m3 |
| --- | --- | --- | --- |
| Size | 33M parameters, ~130 MB | 110M parameters, ~440 MB | 568M parameters, ~2.3 GB |
| Vector size | 384 | 768 | 1024 |
| Max input | 512 tokens | 512 tokens | 8192 tokens |
| Licence | MIT | MIT | MIT |
| Languages | English | English | Multilingual |

- **Model:** `BAAI/bge-base-en-v1.5`, pinned to revision `a5beb1e3` so re-ingesting gives the same vectors. Loaded with sentence-transformers.
- **Why:**
  - **Quality vs. cost:** retrieves noticeably better than the small model on public retrieval benchmarks, while staying small enough for a client's CPU-only server. Our own evals in Phase 3 make the final call.
  - **Speed:** the whole corpus (318 chunks) embeds in 28 s on the Air's GPU (MPS) or 51 s on its CPU, using about 1.2 GB of memory.
  - **Licence:** MIT, no conditions on commercial use.
  - **Private:** runs locally; no document text leaves the machine. Once the model is downloaded, `HF_HUB_OFFLINE=1` stops even the version check against Hugging Face.
- **How it's used:**
  - Passages are embedded as `title + heading + chunk text`, so a chunk from the middle of a section still carries which act or case and which section it belongs to.
  - Questions get BGE's recommended prefix `Represent this sentence for searching relevant passages: `; passages don't.
  - Vectors are normalised to length 1, so cosine similarity is a dot product (pgvector's `<=>` cosine distance works directly).
- **Chunk size (changes the 500 to 800 tokens in the original plan):** the model reads at most 512 tokens and silently drops the rest. Chunks are therefore at most **400 tokens**, counted with the model's own tokenizer, leaving up to about 100 for the title and heading. On the corpus the longest passage is 463 tokens (limit 510), the median 384. A "4 characters per token" estimate undercounted by up to 44% and would have let 4 of 314 chunks be cut short, so the real tokenizer is used, and `embed_passages` raises an error instead of truncating.
- **Rejected:**
  - **bge-small-en-v1.5:** faster, but lower retrieval quality, and speed isn't the bottleneck at this corpus size.
  - **bge-m3:** longer inputs and multilingual (useful later for Hindi documents), but 5 times larger and slow on an 8 GB laptop. Re-evaluate if clients need regional-language search.

