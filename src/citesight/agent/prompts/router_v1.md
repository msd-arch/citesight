# router v1

Classifies the query and sets retrieval strategy. Placeholders: {question}.

---

You route questions over a corpus of SEC filings (10-K/10-Q/8-K page images).

Question: {question}

Reply with ONLY a JSON object:

{{
  "query_type": "factoid" | "synthesis" | "table-math" | "multi-doc-comparison",
  "ticker": "<uppercase ticker if the question names a company, else null>",
  "filing_type": "<10-K | 10-Q | 8-K if clearly implied, else null>",
  "top_k": <pages to retrieve: 3 for factoid, 5 for synthesis/table-math, 8 for multi-doc-comparison>,
  "text_heavy": <true if the question is about narrative prose rather than tables/figures>
}}
