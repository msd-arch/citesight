"""Eval runner: staged so at most one large model is resident at a time.

Stages: (1) retrieval rankings for visual / hybrid / baseline; (2) generation —
CiteSight VLM answers from page images, baseline LLM answers from text chunks;
(3) LLM-judge scoring (correctness both systems + CiteSight citation accuracy);
(4) markdown + JSON report. Judge/agent LLM responses are disk-cached so reruns
stay under free-tier caps. Eval tracing is forced to the local exporter.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from citesight.config.settings import Settings
from citesight.eval.baseline import NaiveRagBaseline, chunk_pages
from citesight.eval.golden import GoldenItem, load_golden, resolve_gold_pages
from citesight.eval.judge import JUDGE_PROMPT_VERSION, judge_citation, judge_correctness
from citesight.eval.metrics import mrr, ndcg_at_k, percentile, recall_at_k
from citesight.models.llm import make_llm
from citesight.models.registry import get_registry
from citesight.models.text_retrieval import TextEncoder
from citesight.store.manifest import Manifest

logger = logging.getLogger(__name__)

PageKey = tuple[str, int]


@dataclass
class ItemResult:
    item_id: str
    query: str
    gold: list[list] = field(default_factory=list)
    rankings: dict = field(default_factory=dict)  # system -> [PageKey]
    answers: dict = field(default_factory=dict)  # system -> str
    correctness: dict = field(default_factory=dict)  # system -> (score, reason)
    citation_checks: list = field(default_factory=list)  # (claim, supported, reason)
    latency_s: dict = field(default_factory=dict)


def run_eval(
    settings: Settings,
    max_items: int | None = None,
    skip_generation: bool = False,
    top_k: int | None = None,
) -> Path:
    top_k = top_k or settings.eval_top_k
    manifest = Manifest(settings.manifest_path)
    page_texts = manifest.get_page_texts()
    texts_by_key = {(d, p): t for d, p, t in page_texts}
    indexed_tickers = {d.ticker.upper() for d in manifest.list_documents()
                       if d.status == "indexed"}

    items = [
        i for i in load_golden(settings.eval_golden_dir)
        if i.ticker.upper() in indexed_tickers
    ]
    if max_items:
        items = items[:max_items]
    if not items:
        raise RuntimeError(
            f"no golden items match indexed tickers {sorted(indexed_tickers)} — "
            "ingest a matching corpus first (e.g. citesight ingest --ticker AAPL)"
        )
    logger.info("eval: %d items over tickers %s", len(items), sorted(indexed_tickers))

    results = [
        ItemResult(
            item_id=i.id, query=i.query,
            gold=[list(g) for g in sorted(resolve_gold_pages(i, page_texts))],
        )
        for i in items
    ]
    for r in results:
        if not r.gold:
            logger.warning("item %s resolved no gold pages", r.item_id)

    # ---------------- stage 1: retrieval rankings -------------------------
    _stage_retrieval(settings, manifest, items, results, page_texts, top_k)

    # ---------------- stage 2: generation ---------------------------------
    if not skip_generation:
        _stage_generation(settings, items, results, top_k)

    # ---------------- stage 3: judging ------------------------------------
    if not skip_generation:
        _stage_judging(settings, items, results, texts_by_key)

    # ---------------- stage 4: report -------------------------------------
    report_path = _write_report(settings, items, results, top_k, skip_generation)
    manifest.close()
    return report_path


def _stage_retrieval(settings, manifest, items, results, page_texts, top_k):
    from citesight.agent.hybrid_adapter import HybridPageSearch
    from citesight.models.retriever import ColQwenRetriever
    from citesight.store.qdrant_store import QdrantPageStore

    store = QdrantPageStore(settings)
    retriever = ColQwenRetriever(settings)
    for item, r in zip(items, results):
        t0 = time.perf_counter()
        hits = store.search(
            retriever.embed_query(item.query), top_k=max(top_k, 10), ticker=item.ticker
        )
        r.rankings["visual"] = [(h.doc_id, h.page_num) for h in hits]
        r.latency_s["visual_retrieval"] = time.perf_counter() - t0
    if settings.sequential_models:
        retriever.unload()

    hybrid = HybridPageSearch(settings, manifest)
    for item, r in zip(items, results):
        t0 = time.perf_counter()
        refs = hybrid.search(item.query, top_k=max(top_k, 10))
        r.rankings["hybrid"] = [
            (p["doc_id"], p["page_num"]) for p in refs
            if p["doc_id"].upper().startswith(item.ticker.upper() + "_")
        ]
        r.latency_s["hybrid_retrieval"] = time.perf_counter() - t0

    baseline = NaiveRagBaseline(TextEncoder(settings), chunk_pages(page_texts))
    for item, r in zip(items, results):
        t0 = time.perf_counter()
        r.rankings["baseline"] = baseline.page_ranking(item.query, ticker=item.ticker)
        r.latency_s["baseline_retrieval"] = time.perf_counter() - t0
    get_registry().unload_prefixed("text_encoder")
    get_registry().unload_prefixed("reranker")
    store.close()


def _stage_generation(settings, items, results, top_k):
    from citesight.models.retriever import load_page_image
    from citesight.models.vlm import QwenVlAnswerer

    answerer = QwenVlAnswerer(settings)
    for item, r in zip(items, results):
        pages = r.rankings["visual"][: settings.vlm_max_pages]
        if not pages:
            r.answers["citesight"] = ""
            r.answers["_claims"] = []
            continue
        images = [
            load_page_image(settings.pages_dir / d / f"{p:04d}.png") for d, p in pages
        ]
        t0 = time.perf_counter()
        out = answerer.answer(
            item.query, images, max_new_tokens=settings.vlm_max_new_tokens
        )
        r.latency_s["citesight_answer"] = time.perf_counter() - t0
        r.answers["citesight"] = out["answer"]
        # map claim page indices back to (doc_id, page_num)
        r.answers["_claims"] = [
            {**c, "key": list(pages[c["page"] - 1])}
            for c in out["claims"]
            if 0 < c["page"] <= len(pages)
        ]
    if settings.sequential_models:
        answerer.unload()

    from citesight.eval.baseline import NaiveRagBaseline, chunk_pages
    from citesight.store.manifest import Manifest

    manifest = Manifest(settings.manifest_path)
    baseline = NaiveRagBaseline(TextEncoder(settings), chunk_pages(manifest.get_page_texts()))
    manifest.close()
    agent_llm = make_llm("agent", settings, cache=True)
    for item, r in zip(items, results):
        t0 = time.perf_counter()
        try:
            r.answers["baseline"] = baseline.answer(agent_llm, item.query, ticker=item.ticker)
        except Exception as exc:
            logger.warning("baseline answer failed for %s: %s", item.id, exc)
            r.answers["baseline"] = ""
        r.latency_s["baseline_answer"] = time.perf_counter() - t0
    get_registry().unload_prefixed("text_encoder")


def _stage_judging(settings, items, results, texts_by_key):
    judge = make_llm("judge", settings, cache=True)
    for item, r in zip(items, results):
        for system in ("citesight", "baseline"):
            r.correctness[system] = judge_correctness(
                judge, item.query, item.reference_answer, r.answers.get(system, "")
            )
        for c in r.answers.get("_claims", []):
            key = tuple(c["key"])
            supported, reason = judge_citation(
                judge, c["text"], texts_by_key.get(key, "")
            )
            r.citation_checks.append(
                {"claim": c["text"], "page": list(key), "supported": supported,
                 "reason": reason}
            )


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def _write_report(settings, items, results, top_k, skip_generation) -> Path:
    reports_dir = settings.eval_reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    scored = [r for r in results if r.gold]

    retrieval = {}
    for system in ("visual", "hybrid", "baseline"):
        rankings = [
            ([tuple(k) for k in r.rankings.get(system, [])], {tuple(g) for g in r.gold})
            for r in scored
        ]
        retrieval[system] = {
            f"ndcg@{top_k}": round(_mean(ndcg_at_k(rk, g, top_k) for rk, g in rankings), 3),
            f"recall@{top_k}": round(_mean(recall_at_k(rk, g, top_k) for rk, g in rankings), 3),
            "mrr": round(_mean(mrr(rk, g) for rk, g in rankings), 3),
        }

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_items": len(results),
        "n_scored": len(scored),
        "top_k": top_k,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "retrieval": retrieval,
    }

    if not skip_generation:
        checks = [c for r in results for c in r.citation_checks]
        summary["generation"] = {
            "citesight_correctness": round(
                _mean(r.correctness.get("citesight", (0,))[0] for r in results), 3
            ),
            "baseline_correctness": round(
                _mean(r.correctness.get("baseline", (0,))[0] for r in results), 3
            ),
            "citation_accuracy": round(
                _mean(1.0 if c["supported"] else 0.0 for c in checks), 3
            ) if checks else None,
            "n_claims_checked": len(checks),
        }
        lat = [r.latency_s.get("citesight_answer", 0) for r in results
               if "citesight_answer" in r.latency_s]
        summary["latency"] = {
            "citesight_answer_p50_s": round(percentile(lat, 50), 1),
            "citesight_answer_p95_s": round(percentile(lat, 95), 1),
        }

    # ---- markdown ----
    lines = [
        "# CiteSight eval report",
        f"\n_{summary['generated_at']} — {summary['n_scored']}/{summary['n_items']} "
        f"items scored, judge prompt {JUDGE_PROMPT_VERSION}_\n",
        "## Retrieval (CiteSight vs baseline)\n",
        f"| system | nDCG@{top_k} | Recall@{top_k} | MRR |",
        "|---|---|---|---|",
    ]
    label = {"visual": "CiteSight (visual)", "hybrid": "CiteSight (hybrid)",
             "baseline": "Naive text RAG"}
    for system, m in retrieval.items():
        lines.append(
            f"| {label[system]} | {m[f'ndcg@{top_k}']} | {m[f'recall@{top_k}']} "
            f"| {m['mrr']} |"
        )
    if not skip_generation:
        g = summary["generation"]
        lines += [
            "\n## Generation (LLM-judged vs reference answers)\n",
            "| system | correctness (0-1) |",
            "|---|---|",
            f"| CiteSight | {g['citesight_correctness']} |",
            f"| Naive text RAG | {g['baseline_correctness']} |",
            "\n## Citation accuracy (CiteSight)\n",
            f"- **{g['citation_accuracy']}** of {g['n_claims_checked']} claims "
            "verified as supported by their cited page",
            "\n## Latency\n",
            f"- CiteSight answer p50 {summary['latency']['citesight_answer_p50_s']}s, "
            f"p95 {summary['latency']['citesight_answer_p95_s']}s",
        ]
    lines.append("\n## Per-item detail\n")
    for item, r in zip(items, results):
        v = r.rankings.get("visual", [])[:3]
        lines.append(
            f"- **{r.item_id}** ({item.type}) gold={len(r.gold)}p, "
            f"visual top3={[f'{d.split(chr(95))[-1][:10]}:{p}' for d, p in v]}"
            + (
                f", correct={r.correctness.get('citesight', ('-',))[0]}"
                if r.correctness else ""
            )
        )

    md_path = reports_dir / "latest.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_payload = {
        "summary": summary,
        "items": [
            {
                "id": r.item_id, "gold": r.gold,
                "rankings": {k: [list(t) for t in v] for k, v in r.rankings.items()},
                "answers": {k: v for k, v in r.answers.items() if k != "_claims"},
                "correctness": {k: list(v) for k, v in r.correctness.items()},
                "citation_checks": r.citation_checks,
                "latency_s": {k: round(v, 2) for k, v in r.latency_s.items()},
            }
            for r in results
        ],
    }
    (reports_dir / "latest.json").write_text(
        json.dumps(json_payload, indent=2), encoding="utf-8"
    )
    logger.info("eval report written: %s", md_path)
    return md_path
