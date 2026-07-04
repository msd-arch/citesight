"""Ingestion pipeline: EDGAR fetch -> page images -> ColQwen embeddings -> Qdrant.

Idempotent and resumable: already-indexed doc_ids are skipped, downloads and
rendered pages are reused, and identical content (sha256) is deduped.
"""
from __future__ import annotations

import hashlib
import logging

from citesight.config.settings import Settings
from citesight.ingest.edgar import EdgarClient, Filing
from citesight.ingest.render import render_document
from citesight.models.retriever import ColQwenRetriever, load_page_image
from citesight.store.manifest import DocumentRecord, Manifest
from citesight.store.qdrant_store import PagePoint, QdrantPageStore

logger = logging.getLogger(__name__)


class Indexer:
    def __init__(
        self,
        settings: Settings,
        edgar: EdgarClient,
        retriever: ColQwenRetriever,
        store: QdrantPageStore,
        manifest: Manifest,
    ) -> None:
        self.settings = settings
        self.edgar = edgar
        self.retriever = retriever
        self.store = store
        self.manifest = manifest

    def ingest(
        self,
        ticker: str,
        forms: list[str],
        limit: int,
        max_pages: int | None = None,
    ) -> int:
        """Returns the number of pages newly indexed."""
        filings = self.edgar.list_filings(ticker, forms, limit)
        indexed_pages = 0
        for filing in filings:
            indexed_pages += self._ingest_filing(filing, max_pages)
        return indexed_pages

    def _ingest_filing(self, filing: Filing, max_pages: int | None) -> int:
        doc_id = filing.doc_id
        if self.manifest.is_indexed(doc_id):
            logger.info("skip (already indexed): %s", doc_id)
            return 0

        raw_path = self.edgar.download_primary(filing, self.settings.raw_dir)
        content_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        if self.manifest.has_content_hash(content_hash):
            logger.info("skip (duplicate content): %s", doc_id)
            return 0

        self.manifest.upsert(
            DocumentRecord(
                doc_id=doc_id,
                ticker=filing.ticker,
                cik=filing.cik,
                filing_type=filing.form,
                filing_date=filing.filing_date,
                accession=filing.accession,
                source_url=filing.primary_url,
                content_hash=content_hash,
                status="pending",
            )
        )

        page_paths = render_document(
            raw_path,
            self.settings.pages_dir / doc_id,
            dpi=self.settings.render_dpi,
            max_edge=self.settings.max_image_edge,
            max_pages=max_pages,
        )
        if not page_paths:
            logger.warning("no pages rendered for %s", doc_id)
            return 0

        # page text (from the source document) feeds the hybrid retrieval path
        self.manifest.add_page_texts(
            doc_id,
            [
                p.with_suffix(".txt").read_text(encoding="utf-8")
                if p.with_suffix(".txt").exists()
                else ""
                for p in page_paths
            ],
        )

        images = [load_page_image(p) for p in page_paths]
        embeddings = self.retriever.embed_pages(images)
        if self.settings.pooling_enabled:
            embeddings = self.retriever.pool_embeddings(embeddings)

        self.store.ensure_collection(dim=embeddings[0].shape[1])
        self.store.upsert_pages(
            [
                PagePoint(
                    doc_id=doc_id,
                    ticker=filing.ticker,
                    filing_type=filing.form,
                    filing_date=filing.filing_date,
                    page_num=i + 1,
                    image_path=str(path),
                    multivector=emb,
                )
                for i, (path, emb) in enumerate(zip(page_paths, embeddings))
            ]
        )
        self.manifest.mark_indexed(doc_id, num_pages=len(page_paths))
        logger.info("indexed %s: %d pages", doc_id, len(page_paths))
        return len(page_paths)
