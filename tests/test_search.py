from sift import db, search


def _seed(con):
    fid1 = db.upsert_file(
        con, path="/d/notes.md", mtime=1, size=1, sha256=None, kind="text", status="indexed"
    )
    db.add_chunks(
        con, fid1, [{"kind": "text", "text": "The annual budget report covers fiscal spending."}]
    )
    fid2 = db.upsert_file(
        con, path="/d/recipe.txt", mtime=1, size=1, sha256=None, kind="text", status="indexed"
    )
    db.add_chunks(
        con, fid2, [{"kind": "text", "text": "Mix flour, sugar and butter for the cake."}]
    )
    con.commit()


def test_keyword_search():
    con = db.connect(":memory:")
    _seed(con)
    hits = search.search(con, "budget")
    assert hits and hits[0].path == "/d/notes.md"
    assert "budget" in hits[0].snippet.lower()


def test_fuzzy_substring_match():
    con = db.connect(":memory:")
    _seed(con)
    hits = search.search(con, "budg")
    assert any((h.path == "/d/notes.md" for h in hits))


def test_dedup_by_file():
    con = db.connect(":memory:")
    fid = db.upsert_file(
        con, path="/d/big.txt", mtime=1, size=1, sha256=None, kind="text", status="indexed"
    )
    db.add_chunks(
        con,
        fid,
        [
            {"kind": "text", "text": "budget budget budget one"},
            {"kind": "text", "text": "budget budget two"},
        ],
    )
    con.commit()
    hits = search.search(con, "budget")
    assert len([h for h in hits if h.path == "/d/big.txt"]) == 1


def test_no_results():
    con = db.connect(":memory:")
    _seed(con)
    assert search.search(con, "zzzznotpresent") == []
