"""Render filings (PDF or EDGAR HTML) to page images via PyMuPDF.

Windows-friendly headless renderer (no poppler/system deps). HTML filings are
opened through MuPDF's reflowable-document engine and paginated at US-Letter.
Fidelity is adequate for retrieval; a Chromium print-to-PDF renderer can be
swapped in behind render_document() for pixel-perfect HTML (see constraints.md).
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

REFLOWABLE_SUFFIXES = {".htm", ".html", ".xhtml", ".txt"}


def render_document(
    doc_path: Path,
    out_dir: Path,
    dpi: int = 150,
    max_edge: int = 1540,
    max_pages: int | None = None,
) -> list[Path]:
    """Render each page to PNG under out_dir; returns page image paths (1-based names)."""
    import fitz  # PyMuPDF
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = doc_path.suffix.lower()
    if suffix in REFLOWABLE_SUFFIXES:
        doc = fitz.open(str(doc_path), filetype="html")
        doc.layout(rect=fitz.paper_rect("letter"))
    else:
        doc = fitz.open(str(doc_path))

    n_pages = doc.page_count if max_pages is None else min(doc.page_count, max_pages)
    paths: list[Path] = []
    for page_idx in range(n_pages):
        out_path = out_dir / f"{page_idx + 1:04d}.png"
        txt_path = out_path.with_suffix(".txt")
        if not txt_path.exists():
            # page text feeds the hybrid (BM25 + dense) retrieval path; the
            # visual path stays OCR-free — this comes from the source document
            txt_path.write_text(
                doc.load_page(page_idx).get_text("text"), encoding="utf-8"
            )
        if out_path.exists():
            paths.append(out_path)
            continue
        page = doc.load_page(page_idx)
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        longest = max(img.size)
        if longest > max_edge:
            scale = max_edge / longest
            img = img.resize(
                (round(img.width * scale), round(img.height * scale)),
                Image.LANCZOS,
            )
        img.save(out_path, format="PNG")
        paths.append(out_path)
    doc.close()
    logger.info("rendered %d/%d pages of %s", len(paths), n_pages, doc_path.name)
    return paths
