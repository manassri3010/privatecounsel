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

    [chunk] = chunks
    assert (chunk.page_start, chunk.page_end) == (1, 3)
    assert chunk.heading == "CHAPTER II > 10. What agreements are contracts."


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


def test_short_sections_are_merged_but_chunks_stay_under_the_limit():
    pages = [page(1, *[f"{n}. Section {n} title.—Short body text." for n in range(1, 13)])]  # 6 words each

    chunks = chunk_pages(pages, max_tokens=20, count_tokens=words)

    assert [c.text.count("\n") + 1 for c in chunks] == [3, 3, 3, 3]
    assert chunks[1].heading == "4. Section 4 title."


def test_act_sections_need_the_dash_so_the_table_of_contents_is_not_split():
    lines = [(1, "ARRANGEMENT OF SECTIONS"), (1, "10. What agreements are contracts."),
             (1, "11. Who are competent to contract."),
             (2, "10. What agreements are contracts.—All agreements are contracts if made freely.")]

    sections = split_sections(lines)

    assert [heading for heading, _ in sections] == [
        "ARRANGEMENT OF SECTIONS", "ARRANGEMENT OF SECTIONS > 10. What agreements are contracts."]


def test_heading_wrapped_over_two_lines_is_joined():
    lines = [(1, "CHAPTER II"), (1, "OF CONTRACTS, VOIDABLE CONTRACTS AND VOID"), (1, "AGREEMENTS"),
             (1, "Body text of the chapter.")]

    [(heading, body)] = split_sections(lines)

    assert heading == "CHAPTER II OF CONTRACTS, VOIDABLE CONTRACTS AND VOID AGREEMENTS"
    assert len(body) == 4


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
