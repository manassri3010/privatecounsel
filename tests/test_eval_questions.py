from collections import Counter

from evals.check_questions import normalise, quote_problems
from evals.dataset import Question, Source, corpus_ids, load_questions, problems


def test_the_question_set_is_consistent_and_has_the_planned_mix():
    questions = load_questions()

    assert problems(questions, corpus_ids()) == []
    assert len(questions) == 50
    assert Counter(q.type for q in questions) == {"single": 30, "multi": 8, "unanswerable": 12}


def question(**overrides):
    fields = dict(id="q1", type="single", question="Q?", answer="A.", key_facts=(("fact",),),
                  sources=(Source("act", (13,), "four words at least"),), also_acceptable=())
    return Question(**{**fields, **overrides})


def test_structural_mistakes_are_reported():
    bad = [
        question(),
        question(),  # duplicate id
        question(id="q2", type="opinion"),
        question(id="q3", type="multi"),  # needs two or more sources
        question(id="q4", key_facts=()),
        question(id="q5", type="unanswerable", sources=()),  # still has key facts
        question(id="q6", sources=(Source("missing-doc", (0,), "too short"),)),
    ]

    found = problems(bad, {"act"})

    assert found == [
        "duplicate id q1",
        "q2: unknown type 'opinion'",
        "q3: a multi question has 1 sources",
        "q4: no key facts",
        "q5: an unanswerable question has key facts or sources",
        "q6: unknown document 'missing-doc'",
        "q6: bad pages (0,) for missing-doc",
        "q6: quote too short to locate reliably: 'too short'",
    ]


def test_a_wrong_page_number_is_caught():
    texts = {"act": [normalise("Preamble."), normalise("11. Who are competent to contract.—Every person is\ncompetent")]}
    right = question(sources=(Source("act", (2,), "Every person is competent"),))
    wrong = question(id="q2", sources=(Source("act", (1, 2), "Every person is competent"),))

    assert quote_problems([right], texts) == []
    assert quote_problems([wrong], texts) == [
        "q2: quote not on page(s) [1] of act (found on [2]): 'Every person is competent'"]


def test_matching_ignores_spacing_case_and_ligatures():
    assert normalise("Notiﬁ cation  dated\n06.09.2007") == normalise("notification dated 06.09.2007")
