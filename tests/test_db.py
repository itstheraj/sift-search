from sift import db


def test_schema_and_fts_sync():
    con = db.connect(":memory:")
    fid = db.upsert_file(
        con, path="/x/a.txt", mtime=1.0, size=10, sha256="h", kind="text", status="indexed"
    )
    db.add_chunks(con, fid, [{"kind": "text", "text": "the quick brown fox"}])
    con.commit()
    rows = con.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?", ('"quick"*',)
    ).fetchall()
    assert len(rows) == 1


def test_clear_chunks_clears_fts():
    con = db.connect(":memory:")
    fid = db.upsert_file(
        con, path="/x/a.txt", mtime=1.0, size=10, sha256="h", kind="text", status="indexed"
    )
    db.add_chunks(con, fid, [{"kind": "text", "text": "alpha beta"}])
    con.commit()
    db.clear_chunks(con, fid)
    con.commit()
    rows = con.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?", ('"alpha"*',)
    ).fetchall()
    assert rows == []


def test_trigram_available():
    con = db.connect(":memory:")
    assert db.has_trigram(con) is True


def test_upsert_is_idempotent_by_path():
    con = db.connect(":memory:")
    a = db.upsert_file(con, path="/p", mtime=1, size=1, sha256=None, kind="text", status="indexed")
    b = db.upsert_file(con, path="/p", mtime=2, size=2, sha256=None, kind="text", status="indexed")
    assert a == b
    assert con.execute("SELECT count(*) c FROM files").fetchone()["c"] == 1
