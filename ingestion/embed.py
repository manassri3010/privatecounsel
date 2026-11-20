"""Turn chunks and questions into vectors with BGE (decision 005).

Documents and questions must use the same model, revision and settings, or their vectors aren't
comparable. Vectors are normalised to length 1, so cosine similarity is a plain dot product.
"""

MODEL_NAME = "BAAI/bge-base-en-v1.5"
MODEL_REVISION = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"  # pinned so re-ingesting gives the same vectors
DIMENSIONS = 768
# BGE v1.5 recommends this prefix on short queries (not on passages) for query-to-passage search.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Token budget per passage: the model reads 512 tokens, two of which are its [CLS] and [SEP] markers.
# Chunk text gets at most CHUNK_TOKENS; the title and heading prepended to it get the rest.
MAX_TOKENS = 510
CHUNK_TOKENS = 400
HEADER_CHARS = 200  # each of title and heading; about 50 tokens at most


def passage_text(title, heading, text):
    """What gets embedded: the chunk with its document title and heading, so a chunk from the middle
    of section 10 still "knows" it is about the Indian Contract Act and section 10."""
    header = [part[:HEADER_CHARS] for part in (title, heading) if part]
    return "\n".join([*header, "", text]) if header else text


class Embedder:
    name = f"{MODEL_NAME}@{MODEL_REVISION}"  # stored with each document, to spot vectors from another model

    def __init__(self, model=None):
        if model is None:
            from sentence_transformers import SentenceTransformer  # slow import; only when really needed

            # device=None picks the Apple GPU (MPS) or CUDA when present, else the CPU.
            model = SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION)
        self.model = model

    def count_tokens(self, text):
        """Tokens as the model counts them (no special tokens); pass to chunk_pages as count_tokens."""
        return len(self.model.tokenizer.encode(text, add_special_tokens=False))

    def embed_passages(self, passages, batch_size=32):
        # Fail loudly rather than let the model silently drop the end of a passage.
        for i, passage in enumerate(passages):
            if (n := self.count_tokens(passage)) > MAX_TOKENS:
                raise ValueError(f"passage {i} has {n} tokens; the model reads at most {MAX_TOKENS}")
        return self.model.encode(passages, batch_size=batch_size, normalize_embeddings=True)

    def embed_query(self, question):
        return self.model.encode(QUERY_INSTRUCTION + question, normalize_embeddings=True)
