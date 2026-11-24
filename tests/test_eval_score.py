import pytest

from evals.dataset import Question, Source
from evals.score import (citation_scores, format_summary, key_fact_coverage, retrieval_recall, score_question,
                         summarise)

ACT = Source("act", (13,), "Every person is competent to contract")
CASE = Source("case", (2, 9), "notice to engage another counsel")


def question(type="single", sources=(ACT,), also=(), facts=(("majority",), ("not competent", "incompetent"))):
    return Question("q", type, "Q?", "A.", facts if type != "unanswerable" else (), sources, also)


def test_retrieval_counts_a_chunk_whose_pages_overlap_an_acceptable_page():
    q = question()
    assert retrieval_recall(q, [("act", 12, 14)], 5) == 1  # chunk spans pages 12-14, source on 13
    assert retrieval_recall(q, [("act", 14, 15)], 5) == 0  # right document, wrong page
    assert retrieval_recall(q, [("other", 13, 13)], 5) == 0  # right page, wrong document
    assert retrieval_recall(q, [("x", 1, 1)] * 5 + [("act", 13, 13)], 5) == 0  # 6th place is outside the top 5


def test_multi_question_recall_is_the_share_of_required_sources_found():
    q = question("multi", sources=(ACT, CASE))
    assert retrieval_recall(q, [("case", 9, 9)], 5) == 0.5
    assert retrieval_recall(q, [("case", 9, 9), ("act", 13, 13)], 5) == 1


def test_also_acceptable_passage_counts_for_a_single_source_question():
    q = question(also=(Source("act", (40,), "another passage that also answers"),))
    assert retrieval_recall(q, [("act", 40, 40)], 5) == 1


def test_key_facts_need_any_one_wording_each_ignoring_case_and_spacing():
    q = question()
    assert key_fact_coverage(q, "A minor has not reached MAJORITY, so is\nINCOMPETENT.") == 1
    assert key_fact_coverage(q, "Minors can contract after majority.") == 0.5


def test_citation_precision_and_recall():
    q = question("multi", sources=(ACT, CASE))
    assert citation_scores(q, [("act", 13, 13), ("case", 5, 5)]) == (0.5, 0.5)  # one wrong page
    assert citation_scores(q, [("act", 13, 13), ("case", 2, 2)]) == (1, 1)
    assert citation_scores(q, []) == (None, 0)


def test_scoring_answerable_unanswerable_and_retrieval_only_questions():
    answer = {"found": True, "text": "Not competent before majority [1].", "citations": [("act", 13, 13)]}
    scores = score_question(question(), [("act", 13, 13)], answer)
    assert scores == {"id": "q", "type": "single", "found": True, "retrieval@5": 1, "retrieval@20": 1,
                      "key_facts": 1, "citation_precision": 1, "citation_recall": 1, "cited": True}

    none = question("unanswerable", sources=())
    assert score_question(none, [], {"found": False, "text": "", "citations": []})["correct_not_found"] is True
    assert score_question(none, [], {"found": True, "text": "x", "citations": []})["correct_not_found"] is False

    refused = score_question(question(), [("act", 13, 13)], {"found": False, "text": "", "citations": []})
    assert refused["found"] is False and "key_facts" not in refused

    retrieval_only = score_question(question(), [("act", 13, 13)], None)
    assert retrieval_only["found"] is None and retrieval_only["retrieval@5"] == 1


def test_summary_of_a_run():
    results = [
        {"type": "single", "found": True, "retrieval@5": 1, "retrieval@20": 1, "key_facts": 1,
         "citation_precision": 1, "citation_recall": 1, "cited": True},
        {"type": "single", "found": True, "retrieval@5": 0, "retrieval@20": 1, "key_facts": 0.5,
         "citation_precision": None, "citation_recall": 0, "cited": False},
        {"type": "single", "found": False, "retrieval@5": 1, "retrieval@20": 1},
        {"type": "unanswerable", "found": False, "correct_not_found": True},
        {"type": "unanswerable", "found": True, "correct_not_found": False},
    ]

    s = summarise(results)

    assert s["retrieval@5"] == pytest.approx(2 / 3) and s["retrieval@20"] == 1
    assert s["answer_correct"] == pytest.approx(1 / 3)  # the refused question counts as not correct
    assert s["key_fact_coverage"] == pytest.approx(0.5)
    assert s["citation_precision"] == 1  # the uncited answer has no precision to average
    assert s["citation_recall"] == 0.5 and s["uncited_answers"] == 0.5
    assert s["not_found_when_unanswerable"] == 0.5 and s["wrongly_refused"] == pytest.approx(1 / 3)
    assert "retrieval@5" in format_summary(s) and "67%" in format_summary(s)


def test_retrieval_only_run_reports_only_retrieval():
    s = summarise([{"type": "single", "found": None, "retrieval@5": 1, "retrieval@20": 1},
                   {"type": "unanswerable", "found": None}])
    assert s["retrieval@5"] == 1 and s["answer_correct"] is None and s["wrongly_refused"] is None
