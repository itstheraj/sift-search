from sift import config, db, indexer, search


def test_index_and_search_roundtrip(tmp_path):
    (tmp_path / "a.md").write_text("# Title\nElephants are large mammals.")
    (tmp_path / "b.txt").write_text("Photosynthesis converts light to energy.")
    sub = tmp_path / "code"
    sub.mkdir()
    (sub / "main.py").write_text("def add(a, b):\n    return a + b  # arithmetic")
    cfg = config.Config()
    con = db.connect(":memory:")
    res = indexer.reindex(con, cfg, [tmp_path])
    assert res.indexed == 3
    assert res.scanned == 3
    assert any((h.path.endswith("a.md") for h in search.search(con, "elephants")))
    assert any((h.path.endswith("main.py") for h in search.search(con, "arithmetic")))


def test_incremental_skip(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello world")
    cfg = config.Config()
    con = db.connect(":memory:")
    r1 = indexer.reindex(con, cfg, [tmp_path])
    assert r1.indexed == 1
    r2 = indexer.reindex(con, cfg, [tmp_path])
    assert r2.skipped == 1 and r2.indexed == 0


def test_reindex_on_change(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("original content here")
    cfg = config.Config()
    con = db.connect(":memory:")
    indexer.reindex(con, cfg, [tmp_path])
    import os
    import time

    f.write_text("completely different replacement words")
    os.utime(f, (time.time() + 10, time.time() + 10))
    indexer.reindex(con, cfg, [tmp_path])
    assert any((h.path.endswith("a.txt") for h in search.search(con, "replacement")))
    assert search.search(con, "original") == []


def test_excludes(tmp_path):
    (tmp_path / "keep.txt").write_text("keepme content")
    junk = tmp_path / "node_modules"
    junk.mkdir()
    (junk / "skip.txt").write_text("skipme content")
    cfg = config.Config()
    con = db.connect(":memory:")
    res = indexer.reindex(con, cfg, [tmp_path])
    assert res.indexed == 1
    assert search.search(con, "skipme") == []


def test_image_deferred(tmp_path):
    (tmp_path / "pic.png").write_bytes(b"\x89PNG\r\n\x1a\n fake")
    cfg = config.Config()
    con = db.connect(":memory:")
    res = indexer.reindex(con, cfg, [tmp_path])
    assert res.deferred == 1
    job = con.execute("SELECT stage, state FROM jobs").fetchone()
    assert job["stage"] == "embed_image" and job["state"] == "pending"
