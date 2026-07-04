# vlm_answer v1

Used by `models/vlm.py` to elicit a page-grounded answer with per-claim evidence.
The `{pages}` placeholder is replaced with a numbered list of the attached page images.

---

You are a financial-filings analyst. You are shown {n_pages} page images from SEC filings:

{pages}

Question: {question}
{extra_context}

Answer the question using ONLY what is visible on these pages. If the pages do not
contain the answer, say so explicitly and return an empty claims list.

Respond with a single JSON object, no other text:

{{
  "answer": "<concise answer in 1-3 sentences>",
  "claims": [
    {{
      "text": "<one factual statement from your answer>",
      "page": <index of the supporting page image, 1-based, as listed above>,
      "evidence": "<short verbatim quote or table cell values from that page>"
    }}
  ]
}}

Rules:
- Every number or fact in "answer" must appear as a claim with a page and evidence.
- "page" refers to the position in the list above (1 = first image), not the printed page number.
- If you compute a value from a table, cite the page with the table and put the source cells in "evidence".
- Do not use knowledge that is not on the pages.
