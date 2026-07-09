import math
import pytest
from sift import config, db, indexer, search
from sift.models.ocr import _BaseOcr, looks_like_text


class FakeImageEmbedder:
    dim = 4
    device = "cpu"

    def _v(self, s):
        v = [float(s.count(c)) for c in "abcd"]
        n = math.sqrt(sum((x * x for x in v))) or 1.0
        return [x / n for x in v]

    def encode_image(self, paths):
        return [self._v(p) for p in paths]

    def encode_text(self, texts):
        return [self._v(t) for t in texts]


class FakeOcr:
    def __init__(self, image_text="", pdf_pages=None):
        self.image_text = image_text
        self.pdf_pages = pdf_pages or []

    def ocr_image(self, path):
        return self.image_text

    def ocr_pdf(self, path):
        return self.pdf_pages


def _vec_db():
    try:
        return db.connect(":memory:", load_vec=True)
    except Exception as e:
        pytest.skip(f"sqlite-vec unavailable: {e}")


def test_image_ocr_text_is_searchable(tmp_path):
    (tmp_path / "poster.png").write_bytes(b"x")
    con = _vec_db()
    ocr = FakeOcr(image_text="SECRETWORD pizza party on friday")
    indexer.reindex(con, config.Config(), [tmp_path], None, FakeImageEmbedder(), None, ocr)
    hits = search.search(con, "SECRETWORD")
    assert hits and hits[0].path.endswith("poster.png")
    kinds = {r["kind"] for r in con.execute("SELECT kind FROM chunks")}
    assert "image" in kinds and "text" in kinds


def test_scanned_pdf_falls_back_to_ocr(tmp_path, monkeypatch):
    f = tmp_path / "scan.pdf"
    f.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr(indexer.extractors, "extract", lambda *a, **k: [])
    ocr = FakeOcr(pdf_pages=[{"page": 1, "text": "invoice total 4242 dollars due"}])
    con = db.connect(":memory:")
    res = indexer.reindex(con, config.Config(), [tmp_path], None, None, None, ocr)
    assert res.indexed == 1
    hits = search.search(con, "4242")
    assert hits and hits[0].path.endswith("scan.pdf")
    assert hits[0].page == 1


def test_pdf_with_text_layer_skips_ocr(tmp_path, monkeypatch):
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr(
        indexer.extractors,
        "extract",
        lambda *a, **k: [{"kind": "text", "text": "a real text layer with plenty of words " * 5}],
    )
    called = {"ocr": False}

    class SpyOcr(FakeOcr):
        def ocr_pdf(self, path):
            called["ocr"] = True
            return []

    con = db.connect(":memory:")
    indexer.reindex(con, config.Config(), [tmp_path], None, None, None, SpyOcr())
    assert called["ocr"] is False


GARBAGE = [
    "£",
    ">",
    "S",
    "XN\n\n<Y\n\n5",
    "e\ng__z_\ngl\n-l",
    "i iy\n\n., ‘",
    "s N\no\n‘ﬁ\n\nL&",
    "‘\\\n\nz",
    "A\nVs\n-t\nA4 oy b\n- ) o\nG (\n=g",
    "|\n\n{hike o\n\n&\n\ni\n\nL\n\nL3\n\nJi\n\n\\\n\n(Wit",
    "2297 et | R 1 AN A7) e Tin Ned 7 %\n7z b PN / | | BN 7 s p N A 7/ g W\nI",
    'S\n\n~«./,:“§‘—’" HON\n7 2% TT e BN\nYm0 AN =0 ik\n\n= [\ni 28\n—— E',
]

REAL_TEXT = [
    "Hello world",
    "invoice total 4242 dollars due on receipt",
    "SECRETWORD pizza party on friday",
    "The quick brown fox jumps over the lazy dog",
    "Quarterly Budget Report\nPrepared by the finance team",
]


@pytest.mark.parametrize("text", GARBAGE)
def test_ocr_noise_from_photographs_is_rejected(text):
    assert looks_like_text(text) is False


@pytest.mark.parametrize("text", REAL_TEXT)
def test_real_text_survives_the_filter(text):
    assert looks_like_text(text) is True


def test_looks_like_text_rejects_empty_and_whitespace():
    assert looks_like_text("") is False
    assert looks_like_text("   \n\t ") is False


class _StubOcr(_BaseOcr):
    def __init__(self, text):
        self._out = text

    def _text(self, image):
        return self._out


def test_ocr_image_returns_empty_for_noise(tmp_path, monkeypatch):
    image_mod = pytest.importorskip("PIL.Image", reason="pillow is an optional extra")

    img = image_mod.new("RGB", (4, 4))
    p = tmp_path / "photo.png"
    img.save(p)
    assert _StubOcr("£\n\n>\n\nS").ocr_image(str(p)) == ""
    assert _StubOcr("invoice total due on receipt").ocr_image(str(p)) != ""


def test_scanned_pdf_page_of_noise_is_dropped(tmp_path, monkeypatch):
    pytest.importorskip("PIL.Image", reason="pillow is an optional extra")
    pages = []

    class Pix:
        width = height = 1
        samples = b"\x00\x00\x00"

    class Page:
        def get_pixmap(self, matrix=None):
            return Pix()

    class Doc:
        def __enter__(self):
            return [Page(), Page()]

        def __exit__(self, *a):
            return False

    import sift.models.ocr as ocr_mod

    fitz = type("fitz", (), {"Matrix": lambda *a: None, "open": staticmethod(lambda p: Doc())})
    monkeypatch.setitem(__import__("sys").modules, "fitz", fitz)

    texts = iter(["£", "invoice total 4242 dollars due on receipt"])

    class Stub(ocr_mod._BaseOcr):
        def _text(self, image):
            return next(texts)

    pages = Stub().ocr_pdf("x.pdf")
    assert len(pages) == 1
    assert pages[0]["page"] == 2
    assert "invoice" in pages[0]["text"]


def test_progress_callback_reports_stages(tmp_path):
    (tmp_path / "a.txt").write_text("hello world")
    (tmp_path / "b.md").write_text("# notes")
    con = db.connect(":memory:")
    events = []
    indexer.reindex(
        con, config.Config(), [tmp_path], progress=lambda d, t, p, s: events.append((s, t))
    )
    stages = {s for s, _ in events}
    assert "text" in stages and "done" in stages
    assert events[-1][1] == 2
