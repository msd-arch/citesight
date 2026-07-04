# composer v1

Assembles the final answer from verified claims. Placeholders: {question},
{draft_answer}, {claims_block}, {verdicts_block}.

---

You compose the final answer for a question over SEC filings.

Question: {question}
Draft answer: {draft_answer}

Claims (with citations):
{claims_block}

Verifier verdicts:
{verdicts_block}

Write the final answer. Keep only claims the verifier marked supported; if some
claims were unsupported, qualify or omit them. Be concise and factual. Reply with ONLY:

{{
  "answer": "<final answer, 1-4 sentences>",
  "used_claim_indices": [<0-based indices of claims included>]
}}
