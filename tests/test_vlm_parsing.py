"""VLM output parsing: strict JSON, code fences, malformed claims, prompt build."""
import pytest

from citesight.models.vlm import build_prompt, extract_json, parse_vlm_answer
from citesight.utils.json_parsing import salvage_truncated_json

GOOD = '{"answer": "Net sales were $391B.", "claims": [{"text": "Net sales were $391B", "page": 1, "evidence": "Total net sales $391,035"}]}'


def test_extract_plain_json():
    assert extract_json(GOOD)["answer"] == "Net sales were $391B."


def test_extract_fenced_json():
    fenced = f"Here is the result:\n```json\n{GOOD}\n```"
    assert extract_json(fenced)["claims"][0]["page"] == 1


def test_extract_json_with_prefix_text():
    assert extract_json(f"Sure! {GOOD} hope that helps")["answer"]


def test_extract_no_json_raises():
    with pytest.raises(ValueError):
        extract_json("I cannot determine this from the pages.")


def test_parse_clamps_bad_pages_and_drops_malformed():
    raw = (
        '{"answer": "A.", "claims": ['
        '{"text": "ok", "page": 2, "evidence": "e"},'
        '{"text": "out of range", "page": 9, "evidence": "e"},'
        '{"text": "", "page": 1},'
        '{"page": 1, "evidence": "no text"},'
        '{"text": "bad page", "page": "x"}]}'
    )
    parsed = parse_vlm_answer(raw, n_pages=3)
    assert len(parsed["claims"]) == 1
    assert parsed["claims"][0]["page"] == 2


def test_salvage_truncated_mid_string_keeps_complete_claims():
    truncated = (
        '```json\n{"answer": "A.", "claims": ['
        '{"text": "claim one", "page": 1, "evidence": "ev one"},'
        '{"text": "claim two cut off mid sent'
    )
    data = salvage_truncated_json(truncated)
    assert data["answer"] == "A."
    assert len(data["claims"]) == 1
    assert data["claims"][0]["text"] == "claim one"


def test_salvage_truncated_before_any_claim_keeps_answer():
    truncated = '{"answer": "Apple sells iPhones.", "claims": [{"text": "Apple se'
    data = salvage_truncated_json(truncated)
    assert data["answer"] == "Apple sells iPhones."
    assert data.get("claims", []) == []


def test_salvage_complete_json_passthrough():
    assert salvage_truncated_json(GOOD)["claims"][0]["page"] == 1


def test_parse_vlm_answer_uses_salvage():
    truncated = (
        '{"answer": "A.", "claims": ['
        '{"text": "ok claim", "page": 1, "evidence": "e"},'
        '{"text": "cut'
    )
    parsed = parse_vlm_answer(truncated, n_pages=2)
    assert parsed["answer"] == "A."
    assert len(parsed["claims"]) == 1


def test_build_prompt_substitutes_and_unescapes():
    p = build_prompt("What were net sales?", n_pages=2, extra_context="FY2025")
    assert "You are shown 2 page images" in p
    assert "- Page image 1" in p and "- Page image 2" in p
    assert "What were net sales?" in p
    assert "Additional context: FY2025" in p
    assert "{{" not in p and "}}" not in p
    assert '"claims": [' in p  # JSON schema example survived un-escaping


def test_build_prompt_omits_empty_context():
    p = build_prompt("Q?", n_pages=1)
    assert "Additional context" not in p
