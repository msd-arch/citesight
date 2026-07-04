# judge v1

Eval judge (separate model from the agent LLM; see JUDGE_LLM_PROVIDER). Two tasks:
answer correctness vs a reference, and per-claim citation support. Placeholders
per section: {question}, {reference}, {answer} / {claim}, {page_text}.

---

## correctness

You judge answers to questions over SEC filings.

Question: {question}
Reference answer: {reference}
Candidate answer: {answer}

Score the candidate against the reference. Numbers must match in value and
magnitude; extra correct detail is fine; contradictions are not. Reply with ONLY:

{{
  "score": 1.0 | 0.5 | 0.0,
  "reason": "<one short sentence>"
}}

(1.0 = correct, 0.5 = partially correct/incomplete, 0.0 = wrong or unsupported)

## citation

You verify that a claim is supported by the text of the page it cites.

Claim: {claim}

Cited page text (extracted):
{page_text}

Reply with ONLY:

{{
  "supported": true | false,
  "reason": "<one short sentence>"
}}
