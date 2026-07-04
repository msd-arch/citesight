# verifier v1

Checks each claim against its cited evidence (VLM-extracted verbatim snippets).
Placeholders: {question}, {claims_block}.

---

You verify that an answer about SEC filings is grounded in its cited evidence.

Question: {question}

Claims with the verbatim evidence extracted from the cited page:

{claims_block}

For each claim, decide whether the evidence actually supports it. Be strict about
numbers: values, units, magnitudes, and fiscal periods must match. Reply with ONLY:

{{
  "verdicts": [
    {{
      "claim_index": <0-based index>,
      "supported": true | false,
      "reason": "<one short sentence>",
      "region_hint": "<short phrase locating the evidence on the page, e.g. 'income statement table, Total net sales row'>"
    }}
  ],
  "grounded": <true only if EVERY claim is supported>,
  "reformulated_query": "<if not grounded: a better retrieval query to find the missing evidence, else null>"
}}
