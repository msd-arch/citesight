from citesight.store.manifest import DocumentRecord, Manifest


def _rec(doc_id="AAPL_10-K_acc1", content_hash="h1"):
    return DocumentRecord(
        doc_id=doc_id,
        ticker="AAPL",
        cik="0000320193",
        filing_type="10-K",
        filing_date="2024-11-01",
        accession="0000320193-24-000123",
        source_url="https://example.com/doc.htm",
        content_hash=content_hash,
    )


def test_roundtrip_and_indexing_state(tmp_path):
    m = Manifest(tmp_path / "manifest.db")
    m.upsert(_rec())
    assert not m.is_indexed("AAPL_10-K_acc1")
    assert not m.has_content_hash("h1")  # only indexed docs count for dedupe

    m.mark_indexed("AAPL_10-K_acc1", num_pages=42)
    assert m.is_indexed("AAPL_10-K_acc1")
    assert m.has_content_hash("h1")

    rec = m.get("AAPL_10-K_acc1")
    assert rec.num_pages == 42
    assert rec.indexed_at is not None
    m.close()


def test_upsert_is_idempotent(tmp_path):
    m = Manifest(tmp_path / "manifest.db")
    m.upsert(_rec())
    m.upsert(_rec(content_hash="h2"))
    assert len(m.list_documents()) == 1
    assert m.get("AAPL_10-K_acc1").content_hash == "h2"
    m.close()
