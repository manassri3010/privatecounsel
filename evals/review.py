"""Write a page for checking the evaluation questions by hand.

Usage: python -m evals.review   then open data/eval_review.html in a browser

Lists every question with its expected answer and key facts, and links each source to the PDF at the
cited page, so checking an entry means reading one page rather than searching a document.
The page stays in data/ (gitignored), since it links to local files.
"""

import html
import json

from evals.check_questions import CORPUS
from evals.dataset import load_questions

OUTPUT = CORPUS.parent / "eval_review.html"


def source_links(sources, files):
    links = []
    for s in sources:
        pages = ", ".join(f'<a href="{(CORPUS / files[s.document]).as_uri()}#page={p}">p. {p}</a>' for p in s.pages)
        links.append(f"<li>{html.escape(s.document)}: {pages}<br><q>{html.escape(s.quote)}</q></li>")
    return "".join(links)


def render(questions, files):
    rows = []
    for n, q in enumerate(questions, 1):
        facts = " · ".join(" / ".join(html.escape(a) for a in alternatives) for alternatives in q.key_facts)
        rows.append(f"""
<section>
  <h2>{n}. <code>{html.escape(q.id)}</code> <span class="type">{q.type}</span></h2>
  <p class="q">{html.escape(q.question)}</p>
  <p><b>Expected:</b> {html.escape(q.answer)}</p>
  {f'<p><b>Key facts:</b> {facts}</p>' if facts else ''}
  {f'<ul>{source_links(q.sources, files)}</ul>' if q.sources else ''}
  {f'<p><b>Also acceptable:</b></p><ul>{source_links(q.also_acceptable, files)}</ul>' if q.also_acceptable else ''}
  <label><input type="checkbox"> Checked against the page</label>
</section>""")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Eval question review</title>
<style>
  body {{ font: 15px/1.5 system-ui, sans-serif; max-width: 820px; margin: 24px auto; padding: 0 16px; }}
  section {{ border-top: 1px solid #ccc; padding: 8px 0 12px; }}
  h2 {{ font-size: 1rem; margin: 8px 0; }}
  .type {{ font-weight: normal; color: #666; }}
  .q {{ font-size: 1.05rem; }}
  q {{ color: #555; }}
</style></head><body>
<h1>Evaluation questions: review</h1>
<p>For each question, open the linked page, check the expected answer and key facts against it, and tick the box.
For unanswerable questions, check the topic really isn't covered. Note anything wrong to fix in
<code>evals/questions.toml</code>. (Ticks aren't saved.)</p>
{''.join(rows)}
</body></html>
"""


def main():
    files = {e["id"]: e["file"] for e in json.loads((CORPUS / "manifest.json").read_text())["documents"]}
    OUTPUT.write_text(render(load_questions(), files))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
