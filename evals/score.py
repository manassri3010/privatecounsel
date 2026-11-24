"""Score one system answer against an evaluation question, and summarise a whole run.

Pure functions: no database or model, so every metric is unit-tested.

A retrieved chunk or a citation "matches" a source when it is from the same document and its page
range overlaps the source's acceptable pages. For a single-source question, the also_acceptable
passages count as that source too.
"""

TOP_RETRIEVED = (5, 20)  # retrieval is scored on the top 5 (what the model reads) and the top 20


def normalise(text):
    return " ".join(text.lower().split())


def matches(document, page_start, page_end, source):
    return document == source.document and any(page_start <= p <= page_end for p in source.pages)


def _alternatives(question, required):
    """The sources that satisfy one required source."""
    return [required, *question.also_acceptable] if question.type == "single" else [required]


def retrieval_recall(question, retrieved, k):
    """Share of the required sources found in the top k chunks. retrieved: (document, start, end) in rank order."""
    top = retrieved[:k]
    found = sum(any(matches(*chunk, alt) for chunk in top for alt in _alternatives(question, s))
                for s in question.sources)
    return found / len(question.sources)


def key_fact_coverage(question, answer_text):
    """Share of key facts stated in the answer; a fact counts if any of its wordings appears."""
    text = normalise(answer_text)
    present = [any(normalise(alt) in text for alt in alternatives) for alternatives in question.key_facts]
    return sum(present) / len(present)


def citation_scores(question, citations):
    """(precision, recall): share of citations pointing at an acceptable source, and share of the
    required sources that were cited. citations: (document, start, end)."""
    acceptable = list(question.sources) + list(question.also_acceptable)
    correct = sum(any(matches(*c, s) for s in acceptable) for c in citations)
    cited = sum(any(matches(*c, alt) for c in citations for alt in _alternatives(question, s))
                for s in question.sources)
    precision = correct / len(citations) if citations else None
    return precision, cited / len(question.sources)


def score_question(question, retrieved, answer):
    """Scores for one question. answer: dict with found (bool), text (str), citations [(doc, start, end)],
    or None for a retrieval-only run."""
    result = {"id": question.id, "type": question.type, "found": answer["found"] if answer else None}
    if question.type != "unanswerable":
        result.update({f"retrieval@{k}": retrieval_recall(question, retrieved, k) for k in TOP_RETRIEVED})
    if answer is None:
        return result
    if question.type == "unanswerable":
        result["correct_not_found"] = not answer["found"]
    elif answer["found"]:
        precision, recall = citation_scores(question, answer["citations"])
        result.update(key_facts=key_fact_coverage(question, answer["text"]),
                      citation_precision=precision, citation_recall=recall, cited=bool(answer["citations"]))
    return result


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def summarise(results):
    """The headline metrics of a run."""
    answerable = [r for r in results if r["type"] != "unanswerable"]
    summary = {f"retrieval@{k}": _mean(r[f"retrieval@{k}"] for r in answerable) for k in TOP_RETRIEVED}
    if all(r["found"] is None for r in results):  # retrieval-only run
        return {**summary, **{name: None for name in METRICS if name not in summary}}
    unanswerable = [r for r in results if r["type"] == "unanswerable"]
    answered = [r for r in answerable if r["found"]]
    summary.update({
        "answer_correct": _mean(r.get("key_facts", 0) == 1 for r in answerable),  # all key facts stated
        "key_fact_coverage": _mean(r.get("key_facts", 0) for r in answerable),
        "citation_precision": _mean(r["citation_precision"] for r in answered),
        "citation_recall": _mean(r["citation_recall"] for r in answered),
        "uncited_answers": _mean(not r["cited"] for r in answered),
        "not_found_when_unanswerable": _mean(r["correct_not_found"] for r in unanswerable),
        "wrongly_refused": _mean(not r["found"] for r in answerable),
    })
    return summary


# Shown in this order, with what each number means.
METRICS = {
    "retrieval@5": "required sources in the 5 chunks the model reads",
    "retrieval@20": "required sources in the top 20 chunks",
    "answer_correct": "answerable questions whose answer states every key fact",
    "key_fact_coverage": "key facts stated, on average",
    "citation_precision": "citations pointing at an acceptable source",
    "citation_recall": "required sources that were cited",
    "uncited_answers": "answers citing nothing (lower is better)",
    "not_found_when_unanswerable": "unanswerable questions answered 'not found'",
    "wrongly_refused": "answerable questions answered 'not found' (lower is better)",
}


def format_summary(summary):
    width = max(map(len, METRICS))
    lines = []
    for name, meaning in METRICS.items():
        value = summary.get(name)
        shown = "  n/a" if value is None else f"{value:5.0%}"
        lines.append(f"{name:<{width}}  {shown}   {meaning}")
    return "\n".join(lines)

