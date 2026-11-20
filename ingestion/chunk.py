"""Split parsed pages into chunks for embedding, keeping section headings and page numbers.

Usage: python -m ingestion.chunk <file.pdf>   (prints a summary of each chunk)

1. Running headers, footers and page numbers are removed, so they don't pollute every chunk.
2. The text is split into sections at headings: capitalised lines ("CHAPTER II", "HEADNOTES") and
   numbered act sections ("10. What agreements are contracts.—").
3. Long sections are split at line boundaries, preferably at the end of a sentence; consecutive
   short sections are merged, so every chunk is as close to max_tokens as possible without going over.
Every line remembers its page, so each chunk knows the pages it spans.
"""

import re
import sys
from collections import Counter
from dataclasses import dataclass

# The small and base BGE models read at most 512 tokens. Approximate counts can be off, and the
# heading may be added before embedding, so stay well below that. Revisited with decision 005.
MAX_TOKENS = 450
EDGE_LINES = 4  # how far from the top or bottom of a page a running header or footer can be
MIN_REPEATS = 3  # an edge line seen on this many pages is a running header or footer

# "10. What agreements are contracts.—All agreements ..." (the table of contents has no dash)
SECTION = re.compile(r"^(\d+[A-Z]{0,3})\.\s+(.{2,150}?)\s*(?:—|\.--)")


@dataclass(frozen=True)
class Chunk:
    text: str
    heading: str  # e.g. "CHAPTER II ... > 10. What agreements are contracts."; "" before any heading
    page_start: int
    page_end: int


def approx_tokens(text):
    """About four characters per token for English text; replaced by the embedding model's tokenizer.

    Rounds up, so adding up the counts of single lines never underestimates the count of the whole chunk.
    """
    return max(1, -(-len(text) // 4))


def _shape(line):
    return re.sub(r"\d+", "#", " ".join(line.split()))


def strip_running_headers(pages):
    """Return (page number, line) pairs without page numbers and repeated headers and footers."""
    page_lines = [(page.number, [line.strip() for line in page.text.splitlines() if line.strip()])
                  for page in pages]
    repeats = Counter()
    for _, lines in page_lines:
        repeats.update({_shape(line) for line in lines[:EDGE_LINES] + lines[-EDGE_LINES:]})
    headers = [shape for shape, n in repeats.items() if n >= MIN_REPEATS]
    kept = []
    for number, lines in page_lines:
        for i, line in enumerate(lines):
            at_edge = i < EDGE_LINES or i >= len(lines) - EDGE_LINES
            if at_edge and _is_header(_shape(line), headers):
                continue
            kept.append((number, line))
    return kept


def _is_header(shape, headers):
    """A page number, a repeated header, or the start of one (some pages carry a shortened header)."""
    if shape == "#" or shape in headers:
        return True
    return len(shape) >= 10 and any(header.startswith(shape) for header in headers)


def is_caps_heading(line):
    letters = [c for c in line if c.isalpha()]
    return len(letters) >= 4 and len(line) <= 80 and all(c.isupper() for c in letters)


def split_sections(lines):
    """Group (page, line) pairs into (heading, lines) sections."""
    sections, part, section_title, current = [], "", "", []
    has_body = previous_was_caps = False
    for page, line in lines:
        caps = is_caps_heading(line)
        match = None if caps else SECTION.match(line)
        if caps and previous_was_caps:
            part = f"{part} {line}"  # a heading that wraps onto several lines
        elif caps or match:
            # Start a new section, unless the current one holds only headings so far:
            # "CHAPTER II" followed directly by "10. ...—" is one section.
            if has_body:
                sections.append((_heading(part, section_title), current))
                current, has_body = [], False
            if caps:
                part, section_title = line, ""
            else:
                section_title = f"{match.group(1)}. {match.group(2).rstrip('.')}."
                has_body = True  # an act section's first line also carries its text
        else:
            has_body = True
        previous_was_caps = caps
        current.append((page, line))
    if current:
        sections.append((_heading(part, section_title), current))
    return sections


def _heading(part, section_title):
    return " > ".join(h for h in (part, section_title) if h)


def _split_long(lines, max_tokens, count_tokens):
    """Split a section's lines into pieces of at most max_tokens, preferring sentence ends."""
    pieces, current, tokens = [], [], 0
    for page, line in lines:
        line_tokens = count_tokens(line + "\n")
        if current and tokens + line_tokens > max_tokens:
            cut = len(current)
            for i in range(len(current), (len(current) - 1) // 2, -1):  # keep at least half
                if current[i - 1][1].endswith((".", ":", ";", "?", "”", '"')):
                    cut = i
                    break
            pieces.append(current[:cut])
            current = current[cut:]
            tokens = sum(count_tokens(text + "\n") for _, text in current)
        current.append((page, line))
        tokens += line_tokens
    if current:
        pieces.append(current)
    return pieces


def chunk_pages(pages, max_tokens=MAX_TOKENS, count_tokens=approx_tokens):
    """Turn parsed pages into chunks of at most max_tokens (a single over-long line is kept whole)."""
    pieces = []
    for heading, lines in split_sections(strip_running_headers(pages)):
        for piece in _split_long(lines, max_tokens, count_tokens):
            pieces.append((heading, piece, sum(count_tokens(text + "\n") for _, text in piece)))

    chunks, heading, current, tokens = [], "", [], 0
    for piece_heading, piece, piece_tokens in pieces:
        if current and tokens + piece_tokens > max_tokens:
            chunks.append(_make_chunk(heading, current))
            current, tokens = [], 0
        if not current:
            heading = piece_heading
        current.extend(piece)
        tokens += piece_tokens
    if current:
        chunks.append(_make_chunk(heading, current))
    return chunks


def _make_chunk(heading, lines):
    return Chunk("\n".join(text for _, text in lines), heading, lines[0][0], lines[-1][0])


if __name__ == "__main__":
    from ingestion.parse import parse_pdf

    chunks = chunk_pages(parse_pdf(sys.argv[1]))
    for i, c in enumerate(chunks):
        pages = f"p{c.page_start}" if c.page_start == c.page_end else f"p{c.page_start}-{c.page_end}"
        print(f"{i:3}  {pages:8} ~{approx_tokens(c.text):3} tok  {c.heading[:90]}")
