from ingestion.chunk import approx_tokens, chunk_pages, split_sections, strip_running_headers
from ingestion.parse import Page


def words(text):
    """Deterministic token count for tests: one token per word."""
    return len(text.split())


def page(number, *lines):
    return Page(number, "\n".join(lines), ocr=False)


def test_chunk_spanning_a_page_break_records_both_pages():
    pages = [
        page(1, "CHAPTER II", "10. What agreements are contracts.—All agreements are contracts"),
        page(2, "if they are made by the free consent of parties.", "11. Who are competent to contract.—Every"),
        page(3, "person of the age of majority."),
    ]

    chunks = chunk_pages(pages, max_tokens=1000, count_tokens=words)

    assert [(c.heading, c.page_start, c.page_end) for c in chunks] == [
        ("CHAPTER II > 10. What agreements are contracts.", 1, 2),
        ("CHAPTER II > 11. Who are competent to contract.", 2, 3),
    ]


def test_long_section_is_split_with_correct_pages_and_no_text_lost():
    lines = [f"Sentence number {i} of the long section." for i in range(40)]  # 7 words each
    pages = [page(1, "HEADNOTES", *lines[:20]), page(2, *lines[20:])]

    chunks = chunk_pages(pages, max_tokens=50, count_tokens=words)

    assert len(chunks) > 1
    assert all(words(c.text) <= 50 for c in chunks)
    assert all(c.heading == "HEADNOTES" for c in chunks)
    assert "\n".join(c.text for c in chunks) == "\n".join(["HEADNOTES", *lines])
    for c in chunks:  # each chunk's pages match where its lines came from
        first = c.text.splitlines()[0]
        last = c.text.splitlines()[-1]
        assert c.page_start == (1 if first == "HEADNOTES" or first in lines[:20] else 2)
        assert c.page_end == (1 if last in lines[:20] else 2)


def test_split_prefers_the_end_of_a_sentence():
    pages = [page(1, "JUDGMENT", "The appeal was filed in time.", "It raises a question of", "limitation under",
                  "section 5.", "Next sentence here.")]

    chunks = chunk_pages(pages, max_tokens=14, count_tokens=words)

    assert chunks[0].text.endswith("filed in time.")  # not mid-sentence after "question of"


def test_short_judgment_parts_are_merged_but_chunks_stay_under_the_limit():
    parts = ["CASE DETAILS", "Party v. Party", "HEADNOTES", "Issue for consideration here.",
             "LIST OF ACTS", "Code of Civil Procedure, 1908.", "LIST OF KEYWORDS", "Ex parte decree."]

    chunks = chunk_pages([page(1, *parts)], max_tokens=10, count_tokens=words)

    assert [c.text.splitlines() for c in chunks] == [parts[:4], parts[4:6], parts[6:]]  # 10, 8 and 6 tokens
    assert [c.heading for c in chunks] == ["CASE DETAILS", "LIST OF ACTS", "LIST OF KEYWORDS"]


def test_each_act_section_gets_its_own_chunk_however_short():
    pages = [page(1, "CHAPTER II", *[f"{n}. Section {n} title.—Short body text." for n in range(10, 13)])]

    chunks = chunk_pages(pages, max_tokens=1000, count_tokens=words)

    assert [c.heading for c in chunks] == [f"CHAPTER II > {n}. Section {n} title." for n in range(10, 13)]
    assert chunks[0].text == "CHAPTER II\n10. Section 10 title.—Short body text."


def test_state_amendment_note_stays_in_its_section():
    lines = [(1, "CHAPTER I"), (1, "7. Acceptance must be absolute.—In order to convert a proposal."),
             (1, "STATE AMENDMENT"), (1, "Andhra Pradesh.—In section 7, add a proviso."),
             (1, "8. Acceptance by performing conditions.—Performance is acceptance.")]

    sections = split_sections(lines)

    assert [s.heading for s in sections] == [
        "CHAPTER I > 7. Acceptance must be absolute.", "CHAPTER I > 8. Acceptance by performing conditions."]
    assert "STATE AMENDMENT" in [line for _, line in sections[0].lines]


def test_act_sections_need_the_dash_so_the_table_of_contents_is_not_split():
    lines = [(1, "ARRANGEMENT OF SECTIONS"), (1, "10. What agreements are contracts."),
             (1, "11. Who are competent to contract."),
             (2, "10. What agreements are contracts.—All agreements are contracts if made freely.")]

    sections = split_sections(lines)

    assert [s.heading for s in sections] == [
        "ARRANGEMENT OF SECTIONS", "ARRANGEMENT OF SECTIONS > 10. What agreements are contracts."]


def test_act_section_formats_found_in_india_code():
    lines = [(1, "73.Compensation for loss or damage caused by breach of contract.—When a contract is broken."),
             (1, "1[16.“Undue influence” defined.—(1) A contract is said to be induced by undue influence."),
             (1, "2[19A. Power to set aside contract induced by undue influence.—When consent is caused."),
             (2, "55. The aforesaid decision also holds as under:—"),  # a judgment paragraph, not a section
             (2, "“The appeal is dismissed.”")]

    sections = split_sections(lines)

    assert [s.heading for s in sections] == [
        "73. Compensation for loss or damage caused by breach of contract.",
        "16. “Undue influence” defined.",
        "19A. Power to set aside contract induced by undue influence.",
    ]
    assert len(sections[-1].lines) == 3  # the judgment paragraph stays in the last section's text


def test_heading_wrapped_over_two_lines_is_joined():
    lines = [(1, "CHAPTER II"), (1, "OF CONTRACTS, VOIDABLE CONTRACTS AND VOID"), (1, "AGREEMENTS"),
             (1, "Body text of the chapter.")]

    [section] = split_sections(lines)

    assert section.heading == "CHAPTER II OF CONTRACTS, VOIDABLE CONTRACTS AND VOID AGREEMENTS"
    assert len(section.lines) == 4 and not section.numbered


def test_running_headers_footers_and_page_numbers_are_removed():
    pages = [
        page(n, str(140 + n), "SUPREME COURT REPORTS", "[2023] 11 S.C.R.", f"Body line on page {n}.",
             "Another body line.", *[f"Body line {i} on page {n}." for i in range(6)])
        for n in range(1, 5)
    ]
    # A shortened header on one page is removed too, because it is the start of a repeated one.
    pages.append(page(5, "145", "Y. P. LELE v. MSEDCL", "Last body line."))
    pages[0:4] = [Page(p.number, "Y. P. LELE v. MSEDCL [VIKRAM NATH, J.]\n" + p.text, False) for p in pages[0:4]]

    kept = [line for _, line in strip_running_headers(pages)]

    assert not any("SUPREME COURT REPORTS" in line or "S.C.R." in line or "LELE" in line for line in kept)
    assert not any(line.isdigit() for line in kept)
    assert kept.count("Another body line.") == 4  # repeated, but not at the top or bottom of the page
    assert "Last body line." in kept


def test_empty_pages_give_no_chunks():
    assert chunk_pages([page(1, ""), page(2, "  ")]) == []


def test_approx_tokens_rounds_up():
    assert approx_tokens("abcde") == 2
    assert approx_tokens("") == 1
