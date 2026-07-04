"""SQLite document manifest: metadata + idempotent/resumable ingestion state."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    ticker       TEXT NOT NULL,
    cik          TEXT NOT NULL,
    filing_type  TEXT NOT NULL,
    filing_date  TEXT NOT NULL,
    accession    TEXT NOT NULL,
    source_url   TEXT NOT NULL,
    content_hash TEXT,
    num_pages    INTEGER,
    status       TEXT NOT NULL DEFAULT 'pending',  -- pending|rendered|indexed
    indexed_at   TEXT
);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash);
CREATE TABLE IF NOT EXISTS page_texts (
    doc_id   TEXT NOT NULL,
    page_num INTEGER NOT NULL,
    text     TEXT NOT NULL,
    PRIMARY KEY (doc_id, page_num)
);
"""


@dataclass
class DocumentRecord:
    doc_id: str
    ticker: str
    cik: str
    filing_type: str
    filing_date: str
    accession: str
    source_url: str
    content_hash: str | None = None
    num_pages: int | None = None
    status: str = "pending"
    indexed_at: str | None = None


class Manifest:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def upsert(self, rec: DocumentRecord) -> None:
        self.conn.execute(
            """INSERT INTO documents
               (doc_id, ticker, cik, filing_type, filing_date, accession,
                source_url, content_hash, num_pages, status, indexed_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(doc_id) DO UPDATE SET
                 content_hash=excluded.content_hash,
                 num_pages=excluded.num_pages,
                 status=excluded.status,
                 indexed_at=excluded.indexed_at""",
            (
                rec.doc_id, rec.ticker, rec.cik, rec.filing_type,
                rec.filing_date, rec.accession, rec.source_url,
                rec.content_hash, rec.num_pages, rec.status, rec.indexed_at,
            ),
        )
        self.conn.commit()

    def get(self, doc_id: str) -> DocumentRecord | None:
        row = self.conn.execute(
            "SELECT * FROM documents WHERE doc_id=?", (doc_id,)
        ).fetchone()
        return DocumentRecord(**dict(row)) if row else None

    def is_indexed(self, doc_id: str) -> bool:
        rec = self.get(doc_id)
        return rec is not None and rec.status == "indexed"

    def has_content_hash(self, content_hash: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM documents WHERE content_hash=? AND status='indexed'",
            (content_hash,),
        ).fetchone()
        return row is not None

    def mark_indexed(self, doc_id: str, num_pages: int) -> None:
        self.conn.execute(
            "UPDATE documents SET status='indexed', num_pages=?, indexed_at=? WHERE doc_id=?",
            (num_pages, datetime.now(timezone.utc).isoformat(), doc_id),
        )
        self.conn.commit()

    def list_documents(self) -> list[DocumentRecord]:
        rows = self.conn.execute(
            "SELECT * FROM documents ORDER BY filing_date DESC"
        ).fetchall()
        return [DocumentRecord(**dict(r)) for r in rows]

    # ------------------------------------------------------- page texts
    def add_page_texts(self, doc_id: str, texts: list[str]) -> None:
        """Store per-page text (1-based page numbers) for the hybrid path."""
        self.conn.executemany(
            "INSERT OR REPLACE INTO page_texts (doc_id, page_num, text) VALUES (?,?,?)",
            [(doc_id, i + 1, t) for i, t in enumerate(texts)],
        )
        self.conn.commit()

    def get_page_texts(self) -> list[tuple[str, int, str]]:
        """All (doc_id, page_num, text) rows for building the hybrid index."""
        return [
            tuple(r)
            for r in self.conn.execute(
                "SELECT doc_id, page_num, text FROM page_texts ORDER BY doc_id, page_num"
            )
        ]

    def close(self) -> None:
        self.conn.close()
