import math
import pytest
from sift import config, db, indexer, search

GROUPS = [["cat", "kitten", "feline"], ["ocean", "sea", "beach"], ["mountain", "hill"]]
_W2G = {w: i for i, g in enumerate(GROUPS) for w in g}


def _vec(tokens):
    v = [0.0] * len(GROUPS)
    for t in tokens:
        t = "".join((c for c in t.lower() if c.isalpha()))
        if t in _W2G:
            v[_W2G[t]] += 1.0
    norm = math.sqrt(sum((x * x for x in v))) or 1.0
    return [x / norm for x in v]


class FakeImageEmbedder:
    dim = len(GROUPS)
    device = "cpu"

    def encode_image(self, paths):
        out = []
        for p in paths:
            stem = p.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            out.append(_vec(stem.replace("_", " ").split()))
        return out

    def encode_text(self, texts):
        return [_vec(t.split()) for t in texts]


def _vec_db():
    try:
        return db.connect(":memory:", load_vec=True)
    except Exception as e:
        pytest.skip(f"sqlite-vec unavailable: {e}")


def test_images_indexed_and_vectorized(tmp_path):
    (tmp_path / "cat.png").write_bytes(b"x")
    (tmp_path / "ocean_view.jpg").write_bytes(b"x")
    con = _vec_db()
    res = indexer.reindex(con, config.Config(), [tmp_path], None, FakeImageEmbedder())
    assert res.indexed == 2 and res.deferred == 0
    assert con.execute("SELECT count(*) c FROM vec_image").fetchone()["c"] == 2
    kinds = {r["kind"] for r in con.execute("SELECT kind FROM chunks")}
    assert kinds == {"image"}


def test_text_to_image_retrieval(tmp_path):
    (tmp_path / "cat.png").write_bytes(b"x")
    (tmp_path / "mountain.png").write_bytes(b"x")
    con = _vec_db()
    ie = FakeImageEmbedder()
    indexer.reindex(con, config.Config(), [tmp_path], None, ie)
    assert search.search(con, "kitten") == []
    hits = search.search(con, "kitten", image_embedder=ie)
    assert hits and hits[0].path.endswith("cat.png")
    assert hits[0].kind == "image"


def test_image_reindex_replaces_vector(tmp_path):
    f = tmp_path / "cat.png"
    f.write_bytes(b"x")
    con = _vec_db()
    ie = FakeImageEmbedder()
    indexer.reindex(con, config.Config(), [f], None, ie)
    import os
    import time

    os.utime(f, (time.time() + 10, time.time() + 10))
    indexer.reindex(con, config.Config(), [f], None, ie)
    assert con.execute("SELECT count(*) c FROM vec_image").fetchone()["c"] == 1
