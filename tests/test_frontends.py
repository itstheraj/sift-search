import json
import shutil
import stat
import pytest
from sift import cli, config, db, indexer, open_hit, search


def test_open_command_document():
    assert open_hit.build_open_command("/x/notes.txt") == ["xdg-open", "/x/notes.txt"]


def test_open_command_media_without_timestamp():
    assert open_hit.build_open_command("/x/clip.mp4", None) == ["xdg-open", "/x/clip.mp4"]
    assert open_hit.build_open_command("/x/clip.mp4", 0) == ["xdg-open", "/x/clip.mp4"]


@pytest.mark.skipif(not shutil.which("mpv"), reason="mpv not installed")
def test_open_command_media_with_timestamp_uses_mpv():
    cmd = open_hit.build_open_command("/x/clip.mp4", 65000)
    assert cmd[0] == "mpv" and cmd[1] == "--start=65.000" and (cmd[2] == "/x/clip.mp4")


def test_path_prefix_scopes_results(tmp_path):
    a = tmp_path / "projA"
    b = tmp_path / "projB"
    a.mkdir()
    b.mkdir()
    (a / "x.txt").write_text("budget planning meeting")
    (b / "y.txt").write_text("budget summary report")
    con = db.connect(":memory:")
    indexer.reindex(con, config.Config(), [tmp_path])
    all_hits = search.search(con, "budget")
    assert len(all_hits) == 2
    scoped = search.search(con, "budget", path_prefix=str(a))
    assert len(scoped) == 1 and scoped[0].path.endswith("projA/x.txt")


def _indexed_cli_env(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    docs = tmp_path / "docs"
    nested = docs / "reports"
    nested.mkdir(parents=True)
    (docs / "top.txt").write_text("quarterly budget overview")
    (nested / "deep.txt").write_text("quarterly budget appendix")
    config.db_path().parent.mkdir(parents=True, exist_ok=True)
    con = db.connect(config.db_path())
    indexer.reindex(con, config.Config(), [docs])
    con.close()
    return docs, nested


def test_search_json_emits_parsable_payload(tmp_path, monkeypatch, capsys):
    _indexed_cli_env(tmp_path, monkeypatch)
    rc = cli.main(["search", "budget", "--json", "--no-semantic", "--no-images"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["query"] == "budget"
    assert payload["count"] == 2
    assert len(payload["hits"]) == 2
    hit = payload["hits"][0]
    for field in ("path", "kind", "chunk_kind", "snippet", "score"):
        assert field in hit


def test_search_json_honours_path_scope(tmp_path, monkeypatch, capsys):
    _, nested = _indexed_cli_env(tmp_path, monkeypatch)
    rc = cli.main(
        ["search", "budget", "--json", "--path", str(nested), "--no-semantic", "--no-images"]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["count"] == 1
    assert payload["hits"][0]["path"].endswith("reports/deep.txt")


def test_search_json_empty_result_is_still_valid_json(tmp_path, monkeypatch, capsys):
    _indexed_cli_env(tmp_path, monkeypatch)
    rc = cli.main(["search", "zzzznomatch", "--json", "--no-semantic", "--no-images"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["count"] == 0 and payload["hits"] == []


def test_init_does_not_clobber_an_existing_config(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    assert cli.main(["init"]) == 0
    assert "Wrote default config" in capsys.readouterr().out

    config.config_path().write_text('profile = "light"\n')
    assert cli.main(["init"]) == 0
    assert "leaving it alone" in capsys.readouterr().out
    assert config.config_path().read_text() == 'profile = "light"\n'


def test_install_kde_writes_files(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    from sift import kde_install

    written = kde_install.install_kde()
    names = {p.name for p in written}
    assert "sift.desktop" in names
    assert "org.sift.krunner.service" in names
    manifest = tmp_path / "krunner" / "dbusplugins" / "sift.desktop"
    text = manifest.read_text()
    assert "X-Plasma-DBusRunner-Service=org.sift.krunner" in text
    assert "X-Plasma-DBusRunner-Path=/sift" in text
    menu = tmp_path / "kio" / "servicemenus" / "sift-search.desktop"
    assert menu.stat().st_mode & stat.S_IXUSR
    assert "Search here with Sift" in menu.read_text()
