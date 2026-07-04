"""CiteSight CLI: ingest / search / status."""
from __future__ import annotations

import logging

import typer

from citesight.config.settings import get_settings

app = typer.Typer(help="CiteSight - visual RAG over SEC filings")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)


@app.command()
def ingest(
    ticker: str = typer.Option(..., help="Ticker symbol, e.g. AAPL"),
    types: str = typer.Option("10-K", help="Comma-separated form types, e.g. 10-K,10-Q"),
    limit: int = typer.Option(2, help="Max filings to ingest"),
    max_pages: int | None = typer.Option(
        None, help="Cap pages per filing (dev/CPU-friendly)"
    ),
) -> None:
    """Fetch filings from EDGAR, render pages, embed with ColQwen, index in Qdrant."""
    from citesight.ingest.edgar import EdgarClient
    from citesight.ingest.indexer import Indexer
    from citesight.models.retriever import ColQwenRetriever
    from citesight.store.manifest import Manifest
    from citesight.store.qdrant_store import QdrantPageStore

    settings = get_settings()
    forms = [t.strip() for t in types.split(",") if t.strip()]
    edgar = EdgarClient(settings)
    store = QdrantPageStore(settings)
    manifest = Manifest(settings.manifest_path)
    indexer = Indexer(settings, edgar, ColQwenRetriever(settings), store, manifest)
    try:
        n = indexer.ingest(ticker, forms, limit, max_pages=max_pages)
        typer.echo(f"Indexed {n} new pages. Collection total: {store.count()} pages.")
    finally:
        edgar.close()
        store.close()
        manifest.close()


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural-language query"),
    top_k: int = typer.Option(5),
    ticker: str | None = typer.Option(None),
    filing_type: str | None = typer.Option(None),
) -> None:
    """Raw MaxSim retrieval over the indexed pages (Phase 1 acceptance check)."""
    from citesight.models.retriever import ColQwenRetriever
    from citesight.store.qdrant_store import QdrantPageStore

    settings = get_settings()
    retriever = ColQwenRetriever(settings)
    store = QdrantPageStore(settings)
    try:
        qvec = retriever.embed_query(query)
        hits = store.search(qvec, top_k=top_k, ticker=ticker, filing_type=filing_type)
        if not hits:
            typer.echo("No results (is anything indexed?)")
            raise typer.Exit(1)
        for i, h in enumerate(hits, 1):
            typer.echo(
                f"{i}. score={h.score:.3f}  {h.ticker} {h.filing_type} "
                f"({h.filing_date}) page {h.page_num}  ->  {h.image_path}"
            )
    finally:
        store.close()


@app.command()
def ask(
    question: str = typer.Argument(..., help="Natural-language question"),
    top_k: int = typer.Option(3, help="Pages to retrieve for the VLM"),
    ticker: str | None = typer.Option(None),
    filing_type: str | None = typer.Option(None),
) -> None:
    """End-to-end: retrieve top-k pages, answer with the VLM, cite pages (Phase 2)."""
    from citesight.pipeline import ask as run_ask

    settings = get_settings()
    result = run_ask(
        question, settings, top_k=top_k, ticker=ticker, filing_type=filing_type
    )
    typer.echo(f"\nAnswer ({result.model_id}):\n  {result.answer}\n")
    if result.claims:
        typer.echo("Claims:")
        for c in result.claims:
            typer.echo(f"  - {c.text}")
            typer.echo(f"    [{c.doc_id} p.{c.page_num}] \"{c.evidence}\"")
    typer.echo("\nRetrieved pages:")
    for ct in result.citations:
        typer.echo(
            f"  score={ct.score:.2f}  {ct.ticker} {ct.filing_type} "
            f"({ct.filing_date}) page {ct.page_num}"
        )


@app.command(name="agent-ask")
def agent_ask(
    question: str = typer.Argument(..., help="Natural-language question"),
    thread_id: str = typer.Option("default", help="Checkpoint thread id (resumable)"),
) -> None:
    """Full agent loop: route -> retrieve -> reason -> verify -> compose (Phase 3)."""
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    from citesight.agent.graph import AgentDeps, build_graph
    from citesight.models.llm import make_llm
    from citesight.models.retriever import ColQwenRetriever
    from citesight.models.vlm import QwenVlAnswerer
    from citesight.observability.tracing import get_tracer
    from citesight.store.manifest import Manifest
    from citesight.store.qdrant_store import QdrantPageStore

    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    tracer = get_tracer(settings, purpose="app")
    deps = AgentDeps(
        settings=settings,
        retriever=ColQwenRetriever(settings),
        store=QdrantPageStore(settings),
        answerer=QwenVlAnswerer(settings),
        llm=make_llm("agent", settings),
        tracer=tracer,
    )
    manifest = Manifest(settings.manifest_path)
    if manifest.get_page_texts():
        from citesight.agent.hybrid_adapter import HybridPageSearch

        deps.hybrid = HybridPageSearch(settings, manifest)
    conn = sqlite3.connect(settings.checkpoints_path, check_same_thread=False)
    graph = build_graph(deps, checkpointer=SqliteSaver(conn))
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.invoke({"query": question}, config=config)
    tracer.flush()

    typer.echo(f"\nAnswer (grounded={state.get('grounded')}, attempts={state.get('attempts')}):")
    typer.echo(f"  {state.get('answer')}\n")
    for c in state.get("citations", []):
        typer.echo(f"  - {c['claim']}")
        typer.echo(
            f"    [{c['doc_id']} p.{c['page_num']}] region: {c.get('region_hint') or 'n/a'}"
        )
    typer.echo(f"\ntrace_id: {tracer.trace_id}  (tracing={settings.tracing})")


@app.command()
def canary(
    skip_vlm: bool = typer.Option(False, help="Only check the retriever"),
) -> None:
    """Discriminative model checks — run BEFORE long index/eval jobs (see constraints.md)."""
    from citesight.diagnostics import run_canaries

    settings = get_settings()
    ok = run_canaries(settings, include_vlm=not skip_vlm)
    typer.echo("CANARY: " + ("PASS ✓" if ok else "FAIL ✗ (models load but are not discriminative)"))
    raise typer.Exit(0 if ok else 1)


@app.command()
def eval(
    max_items: int | None = typer.Option(None, help="Cap golden items (fast subset)"),
    skip_generation: bool = typer.Option(
        False, help="Retrieval metrics only (no VLM/judge; CPU-friendly)"
    ),
    top_k: int | None = typer.Option(None, help="k for nDCG/Recall (default from settings)"),
) -> None:
    """Run the eval harness -> eval/reports/latest.md + latest.json."""
    from citesight.eval.runner import run_eval

    settings = get_settings()
    report = run_eval(
        settings, max_items=max_items, skip_generation=skip_generation, top_k=top_k
    )
    typer.echo(f"\nReport: {report}")
    typer.echo(report.read_text(encoding="utf-8"))


@app.command(name="backfill-text")
def backfill_text() -> None:
    """Extract per-page text for already-indexed documents (hybrid path backfill)."""
    from citesight.ingest.render import render_document
    from citesight.store.manifest import Manifest

    settings = get_settings()
    manifest = Manifest(settings.manifest_path)
    try:
        existing = {d for d, _, _ in manifest.get_page_texts()}
        for doc in manifest.list_documents():
            if doc.status != "indexed" or doc.doc_id in existing:
                continue
            raw = next(settings.raw_dir.glob(f"{doc.doc_id}.*"), None)
            if raw is None:
                typer.echo(f"skip {doc.doc_id}: raw document missing")
                continue
            pages = render_document(
                raw,
                settings.pages_dir / doc.doc_id,
                dpi=settings.render_dpi,
                max_edge=settings.max_image_edge,
                max_pages=doc.num_pages,
            )
            manifest.add_page_texts(
                doc.doc_id,
                [p.with_suffix(".txt").read_text(encoding="utf-8") for p in pages],
            )
            typer.echo(f"backfilled {doc.doc_id}: {len(pages)} pages of text")
    finally:
        manifest.close()


@app.command(name="search-text")
def search_text(
    query: str = typer.Argument(...),
    top_k: int = typer.Option(5),
) -> None:
    """Hybrid (BM25 + bge-m3 + reranker) search over page texts (Phase 4 debug)."""
    from citesight.agent.hybrid_adapter import HybridPageSearch
    from citesight.store.manifest import Manifest

    settings = get_settings()
    manifest = Manifest(settings.manifest_path)
    try:
        hybrid = HybridPageSearch(settings, manifest)
        for i, ref in enumerate(hybrid.search(query, top_k=top_k), 1):
            typer.echo(
                f"{i}. rerank={ref['score']:.3f}  {ref['ticker']} {ref['filing_type']} "
                f"({ref['filing_date']}) page {ref['page_num']}"
            )
    finally:
        manifest.close()


@app.command()
def status() -> None:
    """Show indexed documents and page counts."""
    from citesight.store.manifest import Manifest
    from citesight.store.qdrant_store import QdrantPageStore

    settings = get_settings()
    manifest = Manifest(settings.manifest_path)
    store = QdrantPageStore(settings)
    try:
        docs = manifest.list_documents()
        if not docs:
            typer.echo("No documents in manifest.")
            return
        for d in docs:
            typer.echo(
                f"{d.status:8s} {d.doc_id}  ({d.filing_date}, {d.num_pages or '?'} pages)"
            )
        typer.echo(f"Qdrant pages: {store.count()}")
    finally:
        manifest.close()
        store.close()


if __name__ == "__main__":
    app()
