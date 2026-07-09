import math
import pytest
from sift import config, db, indexer, search


def _unit(vals):
    n = math.sqrt(sum(v * v for v in vals)) or 1.0
    return [v / n for v in vals]


class DirectionalEmbedder:
    dim = 3

    def __init__(self, table=None):
        self.table = table or {}

    def _v(self, s):
        return _unit(self.table.get(s.strip(), [1.0, 0.0, 0.0]))

    def encode(self, texts):
        return [self._v(t) for t in texts]

    def encode_text(self, texts):
        return [self._v(t) for t in texts]

    def encode_image(self, paths):
        return [self._v(p) for p in paths]


def _vec_db():
    try:
        return db.connect(":memory:", load_vec=True)
    except Exception as e:
        pytest.skip(f"sqlite-vec unavailable: {e}")


def test_cosine_from_l2_inverts_normalized_distance():
    assert search.cosine_from_l2(0.0) == pytest.approx(1.0)
    assert search.cosine_from_l2(math.sqrt(2.0)) == pytest.approx(0.0, abs=1e-9)
    assert search.cosine_from_l2(2.0) == pytest.approx(-1.0)


def test_floor_drops_low_similarity_vector_candidates(tmp_path):
    (tmp_path / "note.txt").write_text("alpha beta gamma")
    con = _vec_db()
    emb = DirectionalEmbedder({"alpha beta gamma": [1.0, 0.0, 0.0], "zzz": [0.0, 1.0, 0.0]})
    indexer.reindex(con, config.Config(), [tmp_path], emb)

    assert search.search(con, "zzz", embedder=emb, text_min_similarity=0.0), (
        "with the floor off, the orthogonal chunk is still returned"
    )
    assert search.search(con, "zzz", embedder=emb, text_min_similarity=0.55) == [], (
        "a floor must drop a near orthogonal candidate"
    )


def test_floor_keeps_genuinely_similar_candidates(tmp_path):
    (tmp_path / "note.txt").write_text("alpha beta gamma")
    con = _vec_db()
    emb = DirectionalEmbedder({"alpha beta gamma": [1.0, 0.0, 0.0], "alpha": [1.0, 0.0, 0.0]})
    indexer.reindex(con, config.Config(), [tmp_path], emb)
    hits = search.search(con, "alpha", embedder=emb, text_min_similarity=0.55)
    assert hits and hits[0].path.endswith("note.txt")


def test_keyword_match_wins_an_rrf_tie(tmp_path):
    (tmp_path / "keyword.txt").write_text("reef")
    (tmp_path / "neighbour.txt").write_text("unrelated words entirely")
    con = _vec_db()
    embeds_everything_identically = DirectionalEmbedder()
    indexer.reindex(con, config.Config(), [tmp_path], embeds_everything_identically)

    hits = search.search(con, "reef", embedder=embeds_everything_identically)
    assert hits[0].path.endswith("keyword.txt")


def test_ranking_does_not_depend_on_modality_query_order(tmp_path):
    (tmp_path / "a.txt").write_text("alpha beta")
    (tmp_path / "photo.png").write_bytes(b"x")
    con = _vec_db()
    emb = DirectionalEmbedder()

    indexer.reindex(con, config.Config(), [tmp_path], emb, emb)
    with_both = [h.chunk_id for h in search.search(con, "alpha", embedder=emb, image_embedder=emb)]
    text_only = [h.chunk_id for h in search.search(con, "alpha", embedder=emb)]

    text_order_preserved = [c for c in with_both if c in text_only]
    assert text_order_preserved == [c for c in text_only if c in with_both], (
        "adding the image list must interleave, never reshuffle the text results"
    )


def test_ranking_is_deterministic(tmp_path):
    (tmp_path / "a.txt").write_text("alpha beta")
    (tmp_path / "b.txt").write_text("alpha gamma")
    con = _vec_db()
    emb = DirectionalEmbedder()
    indexer.reindex(con, config.Config(), [tmp_path], emb)
    runs = [[h.chunk_id for h in search.search(con, "alpha", embedder=emb)] for _ in range(5)]
    assert all(r == runs[0] for r in runs)


def test_two_weak_signals_beat_one_strong_signal():
    file_y, file_x = 1, 2
    y_keyword_chunk, x_keyword_chunk, x_vector_chunk = 10, 21, 22
    keyword_ranking = [y_keyword_chunk, x_keyword_chunk]
    vector_ranking = [x_vector_chunk]
    chunk_file = {y_keyword_chunk: file_y, x_keyword_chunk: file_x, x_vector_chunk: file_x}
    lexical = {y_keyword_chunk, x_keyword_chunk}

    scores, best = search._rrf_files([keyword_ranking, vector_ranking], chunk_file, lexical)
    assert scores[file_x] > scores[file_y], (
        "a file found by both keyword and vector must beat one found by keyword alone, "
        "even though fusing over chunks would tie them"
    )
    assert best[file_x][2] == x_vector_chunk, "a file is represented by its best ranked chunk"


def test_a_file_contributes_once_per_list():
    crowded_file, other_file = 1, 2
    crowded_ranking = [1, 2, 3, 4]
    other_ranking = [5]
    chunk_file = {1: crowded_file, 2: crowded_file, 3: crowded_file, 4: crowded_file, 5: other_file}

    scores, _ = search._rrf_files([crowded_ranking, other_ranking], chunk_file, set())
    assert scores[crowded_file] == pytest.approx(1.0 / (search.RRF_K + 1))
    assert scores[other_file] == pytest.approx(1.0 / (search.RRF_K + 1))


def test_search_returns_one_hit_per_file(tmp_path):
    (tmp_path / "many.txt").write_text("alpha " * 900)
    con = _vec_db()
    emb = DirectionalEmbedder()
    indexer.reindex(con, config.Config(), [tmp_path], emb)
    hits = search.search(con, "alpha", embedder=emb)
    assert len({h.file_id for h in hits}) == len(hits)


def test_config_reads_search_floors(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[search]\ntext_min_similarity = 0.7\nimage_min_similarity = 0.11\n")
    cfg = config.load(p)
    assert cfg.text_min_similarity == 0.7
    assert cfg.image_min_similarity == 0.11


def test_saving_config_preserves_search_floors(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[search]\ntext_min_similarity = 0.7\nimage_min_similarity = 0.11\n")
    config.save(config.load(p), p)
    reloaded = config.load(p)
    assert reloaded.text_min_similarity == 0.7, "saving must not silently reset a tuned floor"
    assert reloaded.image_min_similarity == 0.11


def test_similarity_floors_are_off_by_default():
    cfg = config.Config()
    reason = (
        "on bge-m3 junk peaks at cosine 0.5440 and loosely worded relevant text "
        "bottoms out at 0.5452, so no safe cutoff exists and floors ship disabled"
    )
    assert cfg.text_min_similarity == 0.0, reason
    assert cfg.image_min_similarity == 0.0, reason
