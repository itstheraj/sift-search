import math
import pytest
from sift import config, db, indexer, search

SYNONYM_GROUPS = [
    ["car", "automobile", "vehicle"],
    ["dog", "puppy", "canine"],
    ["money", "budget", "finance", "fiscal"],
]
_WORD2GROUP = {w: i for i, g in enumerate(SYNONYM_GROUPS) for w in g}


class FakeEmbedder:
    dim = len(SYNONYM_GROUPS)
    device = "cpu"

    def encode(self, texts):
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for tok in t.lower().split():
                tok = "".join((ch for ch in tok if ch.isalpha()))
                if tok in _WORD2GROUP:
                    v[_WORD2GROUP[tok]] += 1.0
            norm = math.sqrt(sum((x * x for x in v))) or 1.0
            out.append([x / norm for x in v])
        return out


def _vec_db():
    try:
        return db.connect(":memory:", load_vec=True)
    except Exception as e:
        pytest.skip(f"sqlite-vec unavailable: {e}")


def test_vec_roundtrip_knn():
    con = _vec_db()
    fid = db.upsert_file(
        con, path="/a", mtime=1, size=1, sha256=None, kind="text", status="indexed"
    )
    db.add_chunks(con, fid, [{"kind": "text", "text": "I adore my automobile"}])
    con.commit()
    indexer._embed_file_chunks(con, fid, FakeEmbedder())
    con.commit()
    emb = FakeEmbedder()
    from sift.models.text_embed import serialize

    rows = db.vec_knn(con, "vec_text", serialize(emb.encode(["car"])[0]), 5)
    assert rows, "KNN returned nothing"
    assert all(len(r) == 2 for r in rows), "vec_knn must return (chunk_id, distance)"
    distances = [d for _, d in rows]
    assert distances == sorted(distances), "vec_knn must stay ordered by distance"


def test_semantic_finds_what_fts_misses():
    con = _vec_db()
    cfg = config.Config()
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        (p / "ride.txt").write_text("I adore my shiny automobile every morning")
        indexer.reindex(con, cfg, [p], FakeEmbedder())
        assert search.search(con, "car") == []
        hits = search.search(con, "car", embedder=FakeEmbedder())
        assert any((h.path.endswith("ride.txt") for h in hits))


def test_reindex_replaces_vectors(tmp_path):
    con = _vec_db()
    cfg = config.Config()
    f = tmp_path / "x.txt"
    f.write_text("the dog runs")
    indexer.reindex(con, cfg, [f], FakeEmbedder())
    n1 = con.execute("SELECT count(*) c FROM vec_text").fetchone()["c"]
    assert n1 >= 1
    import os
    import time

    f.write_text("the budget grows")
    os.utime(f, (time.time() + 10, time.time() + 10))
    indexer.reindex(con, cfg, [f], FakeEmbedder())
    n2 = con.execute("SELECT count(*) c FROM vec_text").fetchone()["c"]
    assert n2 == 1
