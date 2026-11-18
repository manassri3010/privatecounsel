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
  - **Thinking mode:** Qwen3 "thinks" by default. The API must send `enable_thinking: false` for RAG answers. That is faster, and long hidden reasoning isn't what we want for cited answers. This doesn't affect the throughput benchmarks, which use raw prompts.
  - **Context:** 32K native context is enough for RAG (about 5 chunks of 800 tokens plus the answer).
- **Rejected:**
  - **Llama 3.1 8B Instruct:** gated, restrictive licence terms, older, and no official quantizations.
  - **Qwen3.5-9B:** newest and probably the strongest, but it carries a vision encoder we don't need, uses a new hybrid attention design that makes benchmark results harder to compare, and has no official FP8 or AWQ. Re-evaluate in Phase 3 as a quality upgrade.
  - **Qwen2.5-7B-Instruct:** superseded by Qwen3-8B, and has no official FP8.
