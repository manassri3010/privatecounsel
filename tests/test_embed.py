import numpy as np
import pytest

from ingestion import embed
from ingestion.chunk import chunk_pages
from ingestion.embed import CHUNK_TOKENS, HEADER_CHARS, QUERY_INSTRUCTION, Embedder, passage_text
from ingestion.parse import Page


class FakeTokenizer:
    def encode(self, text, add_special_tokens=True):
        return text.split() + (["[CLS]", "[SEP]"] if add_special_tokens else [])


class FakeModel:
    """Stands in for SentenceTransformer: records calls, returns one fixed vector per input."""

    tokenizer = FakeTokenizer()

    def __init__(self):
        self.calls = []

    def encode(self, inputs, **kwargs):
        self.calls.append((inputs, kwargs))
        return np.ones(3) if isinstance(inputs, str) else np.ones((len(inputs), 3))


def test_passage_includes_title_and_heading_before_the_text():
    assert passage_text("The Indian Contract Act, 1872", "CHAPTER II > 10. What agreements are contracts.",
                        "10. What agreements ...") == (
        "The Indian Contract Act, 1872\nCHAPTER II > 10. What agreements are contracts.\n\n10. What agreements ...")
    assert passage_text("Act", "", "text") == "Act\n\ntext"
    assert passage_text("", "", "text") == "text"
    assert passage_text("T" * 500, "", "x").startswith("T" * HEADER_CHARS + "\n")


def test_passages_are_normalised_and_queries_get_the_instruction():
    model = FakeModel()
    embedder = Embedder(model)

    vectors = embedder.embed_passages(["first passage", "second passage"])
    embedder.embed_query("Who is competent to contract?")

    assert vectors.shape == (2, 3)
    (passages, passage_kwargs), (query, query_kwargs) = model.calls
    assert passages == ["first passage", "second passage"]  # no instruction on passages
    assert query == QUERY_INSTRUCTION + "Who is competent to contract?"
    assert passage_kwargs["normalize_embeddings"] and query_kwargs["normalize_embeddings"]


def test_too_long_passage_is_rejected_instead_of_silently_truncated(monkeypatch):
    monkeypatch.setattr(embed, "MAX_TOKENS", 5)
    embedder = Embedder(FakeModel())

    with pytest.raises(ValueError, match="passage 1 has 6 tokens"):
        embedder.embed_passages(["one two", "one two three four five six"])


def test_chunks_sized_with_the_model_tokenizer_fit_with_their_header():
    embedder = Embedder(FakeModel())
    long_page = Page(1, "\n".join(f"Line {i} of the judgment text here." for i in range(400)), ocr=False)

    chunks = chunk_pages([long_page], max_tokens=CHUNK_TOKENS, count_tokens=embedder.count_tokens)

    assert len(chunks) > 1
    for c in chunks:
        assert embedder.count_tokens(passage_text("Title " * 30, "HEADING " * 30, c.text)) <= embed.MAX_TOKENS


def model_is_cached():
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    path = try_to_load_from_cache(embed.MODEL_NAME, "config.json", revision=embed.MODEL_REVISION)
    return isinstance(path, str)


@pytest.mark.skipif(not model_is_cached(), reason="BGE model not downloaded (about 440 MB)")
def test_real_model_ranks_the_relevant_section_first():
    embedder = Embedder()
    passages = [
        passage_text("The Indian Contract Act, 1872", "11. Who are competent to contract.",
                     "Every person is competent to contract who is of the age of majority and of sound mind."),
        passage_text("The Indian Contract Act, 1872", "73. Compensation for loss or damage caused by breach.",
                     "When a contract has been broken, the party who suffers is entitled to compensation."),
        passage_text("Y. P. Lele v. MSEDCL", "JUDGMENT",
                     "The Trial Court proceeded ex parte after counsel withdrew his vakalatnama."),
    ]

    vectors = embedder.embed_passages(passages)
    query = embedder.embed_query("Can a minor enter into a contract?")

    assert vectors.shape == (3, embed.DIMENSIONS)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    assert np.argmax(vectors @ query) == 0
