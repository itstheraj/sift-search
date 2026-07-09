from __future__ import annotations
import sqlite3
import time
from pathlib import Path

SCHEMA = "\nPRAGMA journal_mode = WAL;\nPRAGMA foreign_keys = ON;\nPRAGMA synchronous = NORMAL;\n\nCREATE TABLE IF NOT EXISTS files (\n    id         INTEGER PRIMARY KEY,\n    path       TEXT UNIQUE NOT NULL,\n    mtime      REAL,\n    size       INTEGER,\n    sha256     TEXT,\n    kind       TEXT,\n    status     TEXT DEFAULT 'pending',\n    error      TEXT,\n    indexed_at REAL\n);\n\nCREATE TABLE IF NOT EXISTS chunks (\n    id       INTEGER PRIMARY KEY,\n    file_id  INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,\n    kind     TEXT NOT NULL,           -- text | transcript | image\n    text     TEXT,\n    start_ms INTEGER,                 -- media: segment start\n    end_ms   INTEGER,                 -- media: segment end\n    page     INTEGER,                 -- docs: page number\n    ord      INTEGER                  -- order within the file\n);\nCREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id);\n\n-- Stemmed keyword recall.\nCREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(\n    text, content='chunks', content_rowid='id', tokenize='porter unicode61'\n);\n-- Substring / typo-tolerant fuzzy matching.\nCREATE VIRTUAL TABLE IF NOT EXISTS chunks_tri USING fts5(\n    text, content='chunks', content_rowid='id', tokenize='trigram'\n);\n\nCREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN\n    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);\n    INSERT INTO chunks_tri(rowid, text) VALUES (new.id, new.text);\nEND;\nCREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN\n    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);\n    INSERT INTO chunks_tri(chunks_tri, rowid, text) VALUES ('delete', old.id, old.text);\nEND;\nCREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN\n    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);\n    INSERT INTO chunks_tri(chunks_tri, rowid, text) VALUES ('delete', old.id, old.text);\n    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);\n    INSERT INTO chunks_tri(rowid, text) VALUES (new.id, new.text);\nEND;\n\nCREATE TABLE IF NOT EXISTS jobs (\n    id         INTEGER PRIMARY KEY,\n    file_id    INTEGER REFERENCES files(id) ON DELETE CASCADE,\n    stage      TEXT NOT NULL,         -- extract | embed_text | embed_image | asr\n    state      TEXT NOT NULL DEFAULT 'pending',  -- pending | running | done | error\n    attempts   INTEGER DEFAULT 0,\n    error      TEXT,\n    updated_at REAL\n);\nCREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state, stage);\n\nCREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);\n"


def connect(path: Path | str, *, load_vec: bool = False) -> sqlite3.Connection:
    path = Path(path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    if load_vec:
        load_sqlite_vec(con)
    con.executescript(SCHEMA)
    con.commit()
    return con


def load_sqlite_vec(con: sqlite3.Connection) -> None:
    import sqlite_vec

    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)


def ensure_vec_table(con: sqlite3.Connection, name: str, dim: int) -> None:
    con.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS {name} USING vec0(chunk_id INTEGER PRIMARY KEY, embedding FLOAT[{dim}])"
    )
    con.execute(
        "INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (f"{name}_dim", str(dim)),
    )


def vec_table_dim(con: sqlite3.Connection, name: str) -> int | None:
    row = con.execute("SELECT value FROM meta WHERE key=?", (f"{name}_dim",)).fetchone()
    return int(row["value"]) if row else None


def add_vectors(con: sqlite3.Connection, table: str, rows: list[tuple[int, bytes]]) -> None:
    con.executemany(f"INSERT OR REPLACE INTO {table}(chunk_id, embedding) VALUES (?, ?)", rows)


def clear_vectors(con: sqlite3.Connection, table: str, file_id: int) -> None:
    if not _table_exists(con, table):
        return
    try:
        con.execute(
            f"DELETE FROM {table} WHERE chunk_id IN (SELECT id FROM chunks WHERE file_id=?)",
            (file_id,),
        )
    except sqlite3.OperationalError:
        pass


def _table_exists(con: sqlite3.Connection, name: str) -> bool:
    row = con.execute("SELECT 1 FROM sqlite_master WHERE name=?", (name,)).fetchone()
    return row is not None


def vec_knn(con: sqlite3.Connection, table: str, blob: bytes, k: int) -> list[tuple[int, float]]:
    if not _table_exists(con, table):
        return []
    rows = con.execute(
        f"SELECT chunk_id, distance FROM {table} WHERE embedding MATCH ? AND k = ? ORDER BY distance",
        (blob, k),
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def has_trigram(con: sqlite3.Connection) -> bool:
    try:
        con.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS _tri_probe USING fts5(x, tokenize='trigram')"
        )
        con.execute("DROP TABLE IF EXISTS _tri_probe")
        return True
    except sqlite3.OperationalError:
        return False


def upsert_file(
    con: sqlite3.Connection,
    *,
    path: str,
    mtime: float,
    size: int,
    sha256: str | None,
    kind: str,
    status: str,
    error: str | None = None,
) -> int:
    con.execute(
        "\n        INSERT INTO files(path, mtime, size, sha256, kind, status, error, indexed_at)\n        VALUES (:path, :mtime, :size, :sha256, :kind, :status, :error, :ts)\n        ON CONFLICT(path) DO UPDATE SET\n            mtime=:mtime, size=:size, sha256=:sha256, kind=:kind,\n            status=:status, error=:error, indexed_at=:ts\n        ",
        dict(
            path=path,
            mtime=mtime,
            size=size,
            sha256=sha256,
            kind=kind,
            status=status,
            error=error,
            ts=time.time(),
        ),
    )
    row = con.execute("SELECT id FROM files WHERE path=?", (path,)).fetchone()
    return int(row["id"])


def clear_chunks(con: sqlite3.Connection, file_id: int) -> None:
    con.execute("DELETE FROM chunks WHERE file_id=?", (file_id,))


def add_chunks(con: sqlite3.Connection, file_id: int, chunks: list[dict]) -> None:
    con.executemany(
        "\n        INSERT INTO chunks(file_id, kind, text, start_ms, end_ms, page, ord)\n        VALUES (:file_id, :kind, :text, :start_ms, :end_ms, :page, :ord)\n        ",
        [
            dict(
                file_id=file_id,
                kind=c.get("kind", "text"),
                text=c.get("text", ""),
                start_ms=c.get("start_ms"),
                end_ms=c.get("end_ms"),
                page=c.get("page"),
                ord=i,
            )
            for i, c in enumerate(chunks)
        ],
    )


def stats(con: sqlite3.Connection) -> dict:
    files = con.execute("SELECT count(*) c FROM files").fetchone()["c"]
    chunks = con.execute("SELECT count(*) c FROM chunks").fetchone()["c"]
    by_status = {
        r["status"]: r["c"]
        for r in con.execute("SELECT status, count(*) c FROM files GROUP BY status")
    }
    by_kind = {
        r["kind"]: r["c"] for r in con.execute("SELECT kind, count(*) c FROM files GROUP BY kind")
    }
    return {"files": files, "chunks": chunks, "by_status": by_status, "by_kind": by_kind}
