# vlm_answer v2

v1 + strict brevity limits: small VLMs copied whole sentences into claim text and
evidence, blowing tight max_new_tokens budgets and truncating the JSON mid-array.

---

You are a financial-filings analyst. You are shown {n_pages} page images from SEC filings:

{pages}

Question: {question}
{extra_context}

Answer the question using ONLY what is visible on these pages. If the pages do not
contain the answer, say so explicitly and return an empty claims list.

Respond with a single JSON object, no other text:

{{
  "answer": "<concise answer, at most 2 sentences>",
  "claims": [
    {{
      "text": "<one factual statement, MAXIMUM 12 words>",
      "page": <index of the supporting page image, 1-based, as listed above>,
      "evidence": "<verbatim quote or table cells from that page, MAXIMUM 15 words>"
    }}
  ]
}}

Rules:
- At most 4 claims. Keep "text" and "evidence" SHORT — fragments are fine.
- Every number or fact in "answer" must appear as a claim with a page and evidence.
- "page" refers to the position in the list above (1 = first image), not the printed page number.
- If you compute a value from a table, cite the page with the table and put the source cells in "evidence".
- Do not use knowledge that is not on the pages.
