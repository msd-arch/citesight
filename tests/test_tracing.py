import json

from citesight.config.settings import Settings
from citesight.observability.tracing import LocalTracer, Tracer, get_tracer


def test_local_tracer_writes_jsonl(tmp_path):
    t = LocalTracer(tmp_path)
    with t.span("router", input="q", attempt=1) as s:
        s.output = {"query_type": "factoid"}
    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    rec = json.loads(files[0].read_text().strip())
    assert rec["span"] == "router"
    assert rec["output"] == {"query_type": "factoid"}
    assert rec["elapsed_ms"] >= 0


def test_off_mode_is_noop(tmp_path):
    settings = Settings(sec_user_agent="t", data_dir=tmp_path, tracing="off", _env_file=None)
    t = get_tracer(settings)
    assert type(t) is Tracer
    with t.span("x") as s:
        s.output = "y"  # must not raise or write anything
    assert not (tmp_path / "traces").exists()


def test_eval_purpose_forces_local(tmp_path):
    settings = Settings(
        sec_user_agent="t", data_dir=tmp_path, tracing="langfuse", _env_file=None
    )
    t = get_tracer(settings, purpose="eval")
    assert isinstance(t, LocalTracer)


def test_span_survives_unserializable_payload(tmp_path):
    t = LocalTracer(tmp_path)
    with t.span("x", input=object()) as s:
        s.output = {"img": object()}
    assert list(tmp_path.glob("*.jsonl"))
