import pytest
from sift import config, db, indexer, search

SEGMENTS = [
    {"text": "Welcome to the quarterly earnings call.", "start_ms": 0, "end_ms": 3200},
    {"text": "Revenue grew across every region this year.", "start_ms": 3200, "end_ms": 7000},
]


class FakeTranscriber:
    device = "cpu"

    def transcribe(self, path):
        return list(SEGMENTS)


class FakeEmbedder:
    dim = 4
    device = "cpu"

    def encode(self, texts):
        import math

        markers = ["revenue", "earnings", "region", "welcome"]
        out = []
        for t in texts:
            low = t.lower()
            v = [float(low.count(m)) for m in markers]
            n = math.sqrt(sum((x * x for x in v))) or 1.0
            out.append([x / n for x in v])
        return out


def _vec_db():
    try:
        return db.connect(":memory:", load_vec=True)
    except Exception as e:
        pytest.skip(f"sqlite-vec unavailable: {e}")


def test_media_transcribed_with_timestamps(tmp_path):
    media = tmp_path / "talk.mp4"
    media.write_bytes(b"not really a video")
    con = db.connect(":memory:")
    res = indexer.reindex(con, config.Config(), [tmp_path], None, None, FakeTranscriber())
    assert res.indexed == 1 and res.deferred == 0
    rows = con.execute("SELECT kind, text, start_ms, end_ms FROM chunks ORDER BY ord").fetchall()
    assert len(rows) == 2
    assert all((r["kind"] == "transcript" for r in rows))
    assert rows[0]["start_ms"] == 0 and rows[1]["start_ms"] == 3200


def test_transcript_is_searchable_with_timestamp(tmp_path):
    media = tmp_path / "talk.mp4"
    media.write_bytes(b"x")
    con = db.connect(":memory:")
    indexer.reindex(con, config.Config(), [tmp_path], None, None, FakeTranscriber())
    hits = search.search(con, "revenue")
    assert hits and hits[0].path.endswith("talk.mp4")
    assert hits[0].chunk_kind == "transcript"
    assert hits[0].start_ms == 3200


def test_transcript_semantic_indexed(tmp_path):
    media = tmp_path / "talk.mp4"
    media.write_bytes(b"x")
    con = _vec_db()
    indexer.reindex(con, config.Config(), [tmp_path], FakeEmbedder(), None, FakeTranscriber())
    assert con.execute("SELECT count(*) c FROM vec_text").fetchone()["c"] == 2


def test_media_deferred_without_transcriber(tmp_path):
    (tmp_path / "song.mp3").write_bytes(b"x")
    con = db.connect(":memory:")
    res = indexer.reindex(con, config.Config(), [tmp_path])
    assert res.deferred == 1
    job = con.execute("SELECT stage FROM jobs").fetchone()
    assert job["stage"] == "asr"
