"""Extract text from a PDF page by page, using OCR for pages that have no text layer.

Usage: python -m ingestion.parse <file.pdf>   (prints a per-page summary)
"""

import sys
from dataclasses import dataclass

import pymupdf

# A page with an image and fewer characters than this is treated as scanned. Scans often still
# carry a few characters (a stamped page number, a typed header), so "no text at all" is too strict.
MIN_TEXT_CHARS = 20
OCR_DPI = 300  # Tesseract is most accurate at around 300 dpi


@dataclass(frozen=True)
class Page:
    number: int  # 1-based, as PDF viewers count, so a citation can link to file.pdf#page=N
    text: str
    ocr: bool  # True if the text came from OCR rather than the PDF's own text layer


def needs_ocr(page):
    return len(page.get_text().strip()) < MIN_TEXT_CHARS and bool(page.get_images())


def ocr_text(page, language="eng", dpi=OCR_DPI):
    """OCR one page with Tesseract, through PyMuPDF's built-in support."""
    textpage = page.get_textpage_ocr(language=language, dpi=dpi, full=True)
    return page.get_text(textpage=textpage)


def parse_pdf(path, ocr=ocr_text):
    """Return one Page per PDF page, in order. Pages without a text layer are OCR'd."""
    pages = []
    with pymupdf.open(path) as doc:
        for page in doc:
            if needs_ocr(page):
                pages.append(Page(page.number + 1, ocr(page), ocr=True))
            else:
                pages.append(Page(page.number + 1, page.get_text(), ocr=False))
    return pages


if __name__ == "__main__":
    for p in parse_pdf(sys.argv[1]):
        first_line = next((line.strip() for line in p.text.splitlines() if line.strip()), "")
        print(f"page {p.number:3}  {'OCR ' if p.ocr else 'text'}  {len(p.text):5} chars  {first_line[:60]}")
