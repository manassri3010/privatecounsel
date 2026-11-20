import shutil

import pymupdf
import pytest

from ingestion.parse import parse_pdf


def add_text_page(doc, text):
    doc.new_page().insert_text((72, 100), text, fontsize=14)


def add_scanned_page(doc, text):
    """A page that is only an image of text, like a scanner's output."""
    with pymupdf.open() as tmp:
        add_text_page(tmp, text)
        pix = tmp[0].get_pixmap(dpi=200, colorspace=pymupdf.csGRAY)
    page = doc.new_page()
    page.insert_image(page.rect, pixmap=pix)


def build(tmp_path, *pages):
    path = tmp_path / "doc.pdf"
    with pymupdf.open() as doc:
        for kind, text in pages:
            {"text": add_text_page, "scan": add_scanned_page, "blank": lambda d, t: d.new_page()}[kind](doc, text)
        doc.save(path)
    return path


def test_text_pages_are_numbered_from_one_without_ocr(tmp_path):
    path = build(tmp_path, ("text", "Section 1. Short title and extent."), ("text", "Section 2. Definitions."))

    pages = parse_pdf(path, ocr=lambda page: pytest.fail("OCR should not run on text pages"))

    assert [p.number for p in pages] == [1, 2]
    assert "Short title" in pages[0].text and "Definitions" in pages[1].text
    assert not any(p.ocr for p in pages)


def test_only_scanned_pages_go_to_ocr(tmp_path):
    path = build(tmp_path, ("text", "Section 1. Short title and extent."), ("scan", "Section 2."),
                 ("blank", ""), ("text", "Section 3. Repeal."))
    ocr_calls = []

    def fake_ocr(page):
        ocr_calls.append(page.number + 1)
        return "ocr text"

    pages = parse_pdf(path, ocr=fake_ocr)

    assert ocr_calls == [2]  # the blank page has no image, so there is nothing to OCR
    assert [(p.number, p.ocr) for p in pages] == [(1, False), (2, True), (3, False), (4, False)]
    assert pages[1].text == "ocr text" and pages[2].text.strip() == ""


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="Tesseract not installed")
def test_tesseract_reads_a_scanned_page(tmp_path):
    path = build(tmp_path, ("scan", "Section 10. What agreements are contracts."))

    [page] = parse_pdf(path)

    assert page.ocr
    assert "What agreements are contracts" in page.text
