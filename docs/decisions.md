# Design decisions

One entry per decision: what, why, and what was rejected. Recruiters and clients read this.

## 001: Serve models with vLLM
- **Why:** industry-standard open-source engine; PagedAttention, continuous batching, prefix caching; OpenAI-compatible API.
- **Rejected:** Ollama for production (built for one user, not many concurrent users). Ollama is still used for local development.

## 002: PostgreSQL + pgvector for retrieval
- **Why:** one database for chunks, vectors, users and permissions; simpler to run and back up at a client site.
- **Rejected:** a separate vector database (extra service to operate for small firms).
