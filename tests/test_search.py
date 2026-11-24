"""Hybrid search against real PostgreSQL + pgvector (see conftest.py)."""

import pytest

from fakes import FakeEmbedder, add_document, nearby
from retrieval.search import search


def test_returns_top_k_ordered_by_similarity_with_citation_details(conn):
    embedder = FakeEmbedder()
    question = "Can a minor enter into a contract?"
    target = embedder.embed_query(question)
    add_document(conn, "m", "The Indian Contract Act, 1872", [
        ("far", 1, nearby(target, 0.5, 1)),
        ("closest", 13, target),
        ("close", 14, nearby(target, 0.05, 2)),
    ], path="data/corpus/indian-contract-act-1872.pdf")

    hits = search(conn, embedder, question, "m", k=2, mode="vector")

    assert [h.text for h in hits] == ["closest", "close"]
    best = hits[0]
    assert best.similarity == pytest.approx(1.0, abs=1e-5)
    assert (best.vector_rank, best.keyword_rank) == (1, None)
    assert (best.title, best.heading, best.pages, best.ocr) == ("The Indian Contract Act, 1872", "HEADING", "p. 13", False)
    assert best.link == "data/corpus/indian-contract-act-1872.pdf#page=13"
    assert hits[0].score > hits[1].score


def test_never_returns_another_matters_chunks_even_when_closer(conn):
    embedder = FakeEmbedder()
    target = embedder.embed_query("question")
    add_document(conn, "other-client", "Their contract", [("exact match", 1, target)])
    add_document(conn, "our-client", "Our contract", [("weaker match", 1, nearby(target, 0.5, 3))])

    hits = search(conn, embedder, "question", "our-client")

    assert [h.text for h in hits] == ["weaker match"]


def test_filtered_index_search_still_returns_k_results(conn):
    """When Postgres searches through the HNSW index, a plain index scan stops after about 40 candidates
    from all matters and can return fewer than k for a small matter; the iterative scan keeps going.

    (For a small matter Postgres would normally use the matter_id index and an exact sort, so the test
    removes that option. HNSW is chosen in real use when a matter is a large share of the table.)"""
    embedder = FakeEmbedder()
    target = embedder.embed_query("question")
    # 400 chunks of another matter all sit closer to the question than our matter's 5.
    add_document(conn, "big-matter", "Big", [(f"big {i}", 1, nearby(target, 0.02, i)) for i in range(400)])
    add_document(conn, "small-matter", "Small", [(f"small {i}", 1, nearby(target, 0.2, 1000 + i)) for i in range(5)])
    conn.execute("DROP INDEX chunks_matter_id")  # only in this test's temporary schema
    conn.execute("ANALYZE chunks")
    conn.execute("SET enable_seqscan = off")  # leaves the HNSW index as the only way to answer

    hits = search(conn, embedder, "question", "small-matter", k=5)

    assert sorted(h.text for h in hits) == [f"small {i}" for i in range(5)]


def test_empty_matter_returns_nothing(conn):
    assert search(conn, FakeEmbedder(), "anything", "no-such-matter") == []


def test_keyword_search_finds_an_exact_term_the_vector_ranks_low(conn):
    embedder = FakeEmbedder()
    question = "application under Order IX Rule 13"
    target = embedder.embed_query(question)
    chunks = [(f"unrelated paragraph {i}", 1, nearby(target, 0.1, i)) for i in range(25)]  # all closer
    chunks.append(("The defendants filed an application under Order IX Rule 13 CPC.", 7, nearby(target, 2.0, 99)))
    add_document(conn, "m", "Y. P. Lele v. MSEDCL", chunks)

    vector = search(conn, embedder, question, "m", k=5, mode="vector")
    hybrid = search(conn, embedder, question, "m", k=5)

    assert all("Order IX" not in h.text for h in vector)  # not even among the 20 vector candidates
    order_ix = next(h for h in hybrid if "Order IX" in h.text)
    assert (order_ix.vector_rank, order_ix.keyword_rank, order_ix.similarity) == (None, 1, None)


def test_question_words_are_or_ed_so_a_chunk_needs_only_some_of_them(conn):
    embedder = FakeEmbedder()
    add_document(conn, "m", "Act", [("Every person of the age of majority is competent to contract.", 13,
                                     embedder.vector("x"))])

    [hit] = search(conn, embedder, "Can a minor who is not of the age of majority contract?", "m", mode="keyword")

    assert hit.keyword_rank == 1  # "minor" is missing from the chunk, but "age", "majority", "contract" match


def test_fusion_prefers_a_chunk_ranked_well_by_both_searches(conn):
    embedder = FakeEmbedder()
    question = "undue influence"
    target = embedder.embed_query(question)
    filler = [(f"Filler paragraph {i}.", 9, nearby(target, 0.3, 100 + i)) for i in range(25)]
    add_document(conn, "m", "Act", [
        ("Vector favourite with no matching words.", 1, target),                          # vector #1 only
        ("Undue influence undue influence undue influence.", 2, nearby(target, 3.0, 1)),  # keyword #1 only
        ("A contract induced by undue influence is voidable.", 3, nearby(target, 0.1, 2)),  # #2 in both
        *filler,  # closer than the keyword-only chunk, so it falls outside the 20 vector candidates
    ])

    hits = search(conn, embedder, question, "m", k=3)

    assert hits[0].text.startswith("A contract induced")
    assert (hits[0].vector_rank, hits[0].keyword_rank) == (2, 2)


def test_keyword_search_respects_the_matter_too(conn):
    embedder = FakeEmbedder()
    add_document(conn, "other-client", "Theirs", [("Order IX Rule 13 application.", 1, embedder.vector("a"))])
    add_document(conn, "our-client", "Ours", [("Something else entirely.", 1, embedder.vector("b"))])

    hits = search(conn, embedder, "Order IX Rule 13", "our-client", mode="keyword")

    assert hits == []


def test_question_of_only_common_words_still_gets_vector_results(conn):
    embedder = FakeEmbedder()
    add_document(conn, "m", "Act", [("Some text.", 1, embedder.vector("q"))])

    hits = search(conn, embedder, "what is it?", "m")  # every word is an English stop word

    assert [(h.vector_rank, h.keyword_rank) for h in hits] == [(1, None)]
