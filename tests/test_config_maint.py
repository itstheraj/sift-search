from sift import config, maint


def test_config_save_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    c = config.load()
    c.profile = "medium"
    c.features = config.Features(text=True, semantic=True, image=False, asr=False, ocr=True)
    c.folders = ["/home/x/Docs", "/data/media"]
    c.device = "cpu"
    config.save(c)

    r = config.load()
    assert r.profile == "medium"
    assert r.features.semantic and r.features.ocr
    assert not r.features.image and not r.features.asr
    assert r.folders == ["/home/x/Docs", "/data/media"]


def test_human_size():
    assert maint.human_size(0) == "0 B"
    assert maint.human_size(512) == "512 B"
    assert maint.human_size(1536).endswith("KB")
    assert maint.human_size(5 * 1024 * 1024).endswith("MB")


def test_index_clear(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    from sift import config as cfg, db, indexer

    f = tmp_path / "a.txt"
    f.write_text("hello world content")
    con = db.connect(cfg.db_path())
    indexer.reindex(con, cfg.Config(), [tmp_path])
    con.close()
    assert maint.index_size() > 0
    assert maint.index_stats()["files"] == 1

    maint.clear_index()
    assert maint.index_size() == 0
