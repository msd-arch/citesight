from citesight.ingest.render import render_document


def _make_pdf(path, n_pages=3):
    import fitz

    doc = fitz.open()
    for i in range(n_pages):
        page = doc.new_page(width=612, height=792)  # US Letter @72dpi
        page.insert_text((72, 72), f"Total net sales page {i + 1}", fontsize=14)
    doc.save(path)
    doc.close()


def test_render_pdf_pages(tmp_path):
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf)
    out = render_document(pdf, tmp_path / "pages", dpi=96, max_edge=800)
    assert [p.name for p in out] == ["0001.png", "0002.png", "0003.png"]

    from PIL import Image

    img = Image.open(out[0])
    assert max(img.size) <= 800
    assert img.mode == "RGB"


def test_render_respects_max_pages_and_is_resumable(tmp_path):
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, n_pages=5)
    out1 = render_document(pdf, tmp_path / "pages", dpi=96, max_edge=800, max_pages=2)
    assert len(out1) == 2
    # second call reuses existing files
    out2 = render_document(pdf, tmp_path / "pages", dpi=96, max_edge=800, max_pages=2)
    assert out1 == out2


def test_render_html(tmp_path):
    html = tmp_path / "filing.htm"
    html.write_text(
        "<html><body><h1>FORM 10-K</h1>" + "<p>Revenue table row</p>" * 200 + "</body></html>",
        encoding="utf-8",
    )
    out = render_document(html, tmp_path / "pages", dpi=96, max_edge=800)
    assert len(out) >= 1
    assert out[0].exists()
