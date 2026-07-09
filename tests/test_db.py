import stat

from sift import db


def _mode(path):
    return stat.S_IMODE(path.stat().st_mode)


def test_index_and_parent_dir_are_owner_only(tmp_path):
    dbp = tmp_path / "data" / "index.db"
    con = db.connect(dbp)
    con.execute("PRAGMA wal_checkpoint")
    assert _mode(dbp) == 0o600
    assert _mode(dbp.parent) == 0o700
    for sidecar in (dbp.with_name("index.db-wal"), dbp.with_name("index.db-shm")):
        if sidecar.exists():
            assert _mode(sidecar) == 0o600, sidecar


def test_connect_tightens_a_world_readable_index(tmp_path):
    """An index created by an older version must not stay world-readable."""
    dbp = tmp_path / "data" / "index.db"
    dbp.parent.mkdir(parents=True)
    dbp.parent.chmod(0o755)
    dbp.touch(mode=0o644)
    db.connect(dbp)
    assert _mode(dbp) == 0o600
    assert _mode(dbp.parent) == 0o700


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
