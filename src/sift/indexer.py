from __future__ import annotations
import fnmatch
import hashlib
import os
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from . import db, extractors
from .config import Config
from .extractors.chunking import chunk_text
from .models.asr import Transcriber
from .models.image_embed import ImageEmbedder, _stem_text
from .models.ocr import PDF_OCR_MIN_CHARS, OcrEngine
from .models.text_embed import Embedder, serialize

VEC_TEXT = "vec_text"
VEC_IMAGE = "vec_image"
EMBEDDABLE = {"text", "transcript"}
Reporter = Callable[[str], None]


def _noop(_stage: str) -> None:
    pass


@dataclass
class IndexResult:
    scanned: int = 0
    indexed: int = 0
    skipped: int = 0
    deferred: int = 0
    errors: int = 0


def _excluded(path: Path, patterns: list[str]) -> bool:
    s = str(path)
    return any((fnmatch.fnmatch(s, pat) for pat in patterns))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _apply_throttle(cfg: Config) -> None:
    try:
        os.nice(cfg.nice)
    except (OSError, AttributeError):
        pass


def iter_files(roots: list[Path], excludes: list[str]):
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            if not _excluded(root, excludes):
                yield root
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            d = Path(dirpath)
            dirnames[:] = [dn for dn in dirnames if not _excluded(d / dn, excludes)]
            for fn in filenames:
                fp = d / fn
                if not _excluded(fp, excludes):
                    yield fp


def _embed_file_chunks(con: sqlite3.Connection, file_id: int, embedder: Embedder) -> None:
    rows = con.execute(
        "SELECT id, text FROM chunks WHERE file_id=? AND kind IN ('text','transcript') ORDER BY ord",
        (file_id,),
    ).fetchall()
    if not rows:
        return
    db.ensure_vec_table(con, VEC_TEXT, embedder.dim)
    texts = [r["text"] or "" for r in rows]
    vectors = embedder.encode(texts)
    db.add_vectors(con, VEC_TEXT, [(rows[i]["id"], serialize(v)) for i, v in enumerate(vectors)])


def _index_image(
    con: sqlite3.Connection,
    cfg: Config,
    path: Path,
    st,
    image_embedder: ImageEmbedder | None,
    embedder: Embedder | None,
    ocr_engine: OcrEngine | None,
    report: Reporter,
) -> str:
    report("image")
    vec = None
    if image_embedder is not None:
        vecs = image_embedder.encode_image([str(path)])
        if not vecs or vecs[0] is None:
            db.upsert_file(
                con,
                path=str(path),
                mtime=st.st_mtime,
                size=st.st_size,
                sha256=None,
                kind="image",
                status="error",
                error="image decode failed",
            )
            con.commit()
            return "error"
        vec = vecs[0]
    file_id = db.upsert_file(
        con,
        path=str(path),
        mtime=st.st_mtime,
        size=st.st_size,
        sha256=None,
        kind="image",
        status="indexed",
    )
    db.clear_vectors(con, VEC_IMAGE, file_id)
    db.clear_vectors(con, VEC_TEXT, file_id)
    db.clear_chunks(con, file_id)
    rows = [{"kind": "image", "text": _stem_text(str(path))}]
    if ocr_engine is not None:
        report("ocr")
        text = ocr_engine.ocr_image(str(path))
        if text.strip():
            rows += [
                {"kind": "text", "text": c}
                for c in chunk_text(text, cfg.chunk_size, cfg.chunk_overlap)
            ]
    db.add_chunks(con, file_id, rows)
    if vec is not None:
        image_chunk_id = con.execute(
            "SELECT id FROM chunks WHERE file_id=? AND kind='image' ORDER BY ord LIMIT 1",
            (file_id,),
        ).fetchone()["id"]
        db.ensure_vec_table(con, VEC_IMAGE, image_embedder.dim)
        db.add_vectors(con, VEC_IMAGE, [(image_chunk_id, serialize(vec))])
    if embedder is not None:
        report("embed")
        _embed_file_chunks(con, file_id, embedder)
    con.commit()
    return "indexed"


def _index_media(
    con: sqlite3.Connection,
    path: Path,
    st,
    kind: str,
    transcriber: Transcriber,
    embedder: Embedder | None,
    report: Reporter,
) -> str:
    report("transcribe")
    segments = transcriber.transcribe(str(path))
    file_id = db.upsert_file(
        con,
        path=str(path),
        mtime=st.st_mtime,
        size=st.st_size,
        sha256=None,
        kind=kind,
        status="indexed",
    )
    db.clear_vectors(con, VEC_TEXT, file_id)
    db.clear_chunks(con, file_id)
    db.add_chunks(
        con,
        file_id,
        [
            {
                "kind": "transcript",
                "text": s["text"],
                "start_ms": s["start_ms"],
                "end_ms": s["end_ms"],
            }
            for s in segments
        ],
    )
    if embedder is not None:
        report("embed")
        _embed_file_chunks(con, file_id, embedder)
    con.commit()
    return "indexed"


def index_file(
    con: sqlite3.Connection,
    cfg: Config,
    path: Path,
    embedder: Embedder | None = None,
    image_embedder: ImageEmbedder | None = None,
    transcriber: Transcriber | None = None,
    ocr_engine: OcrEngine | None = None,
    report: Reporter = _noop,
) -> str:
    try:
        st = path.stat()
    except OSError:
        return "error"
    kind = extractors.classify(path)
    if kind == "unknown":
        return "skipped"
    existing = con.execute(
        "SELECT id, mtime, size, status FROM files WHERE path=?", (str(path),)
    ).fetchone()
    if (
        existing
        and existing["mtime"] == st.st_mtime
        and (existing["size"] == st.st_size)
        and (existing["status"] in ("indexed", "deferred"))
    ):
        return "skipped"
    sha = None
    try:
        if extractors.is_text_bearing(kind):
            report(kind)
            sha = _sha256(path)
            chunks = extractors.extract(
                path, kind, chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap
            )
            if (
                kind == "pdf"
                and ocr_engine is not None
                and (sum((len(c["text"]) for c in chunks)) < PDF_OCR_MIN_CHARS)
            ):
                report("ocr")
                chunks = [
                    {"kind": "text", "text": c, "page": pg["page"]}
                    for pg in ocr_engine.ocr_pdf(str(path))
                    for c in chunk_text(pg["text"], cfg.chunk_size, cfg.chunk_overlap)
                ]
            file_id = db.upsert_file(
                con,
                path=str(path),
                mtime=st.st_mtime,
                size=st.st_size,
                sha256=sha,
                kind=kind,
                status="indexed",
            )
            db.clear_vectors(con, VEC_TEXT, file_id)
            db.clear_chunks(con, file_id)
            db.add_chunks(con, file_id, chunks)
            if embedder is not None:
                report("embed")
                _embed_file_chunks(con, file_id, embedder)
            con.commit()
            return "indexed"
        elif kind == "image" and (image_embedder is not None or ocr_engine is not None):
            return _index_image(con, cfg, path, st, image_embedder, embedder, ocr_engine, report)
        elif kind in ("audio", "video") and transcriber is not None:
            return _index_media(con, path, st, kind, transcriber, embedder, report)
        else:
            file_id = db.upsert_file(
                con,
                path=str(path),
                mtime=st.st_mtime,
                size=st.st_size,
                sha256=None,
                kind=kind,
                status="deferred",
            )
            stage = "embed_image" if kind == "image" else "asr"
            con.execute(
                "INSERT INTO jobs(file_id, stage, state, updated_at) VALUES (?, ?, 'pending', strftime('%s','now'))",
                (file_id, stage),
            )
            con.commit()
            return "deferred"
    except Exception as e:
        db.upsert_file(
            con,
            path=str(path),
            mtime=st.st_mtime,
            size=st.st_size,
            sha256=sha,
            kind=kind,
            status="error",
            error=str(e)[:500],
        )
        con.commit()
        return "error"


def reindex(
    con: sqlite3.Connection,
    cfg: Config,
    roots: list[Path] | None = None,
    embedder: Embedder | None = None,
    image_embedder: ImageEmbedder | None = None,
    transcriber: Transcriber | None = None,
    ocr_engine: OcrEngine | None = None,
    progress: Callable[[int, int, Path, str], None] | None = None,
    should_continue: Callable[[], bool] | None = None,
) -> IndexResult:
    _apply_throttle(cfg)
    roots = roots if roots is not None else cfg.folder_paths
    files = list(iter_files(roots, cfg.excludes))
    total = len(files)
    res = IndexResult()
    for i, path in enumerate(files):
        if should_continue is not None and not should_continue():
            break
        res.scanned += 1
        report = (
            (lambda stage, idx=i, p=path: progress(idx, total, p, stage)) if progress else _noop
        )
        outcome = index_file(
            con, cfg, path, embedder, image_embedder, transcriber, ocr_engine, report
        )
        setattr(
            res,
            outcome if outcome != "error" else "errors",
            getattr(res, outcome if outcome != "error" else "errors") + 1,
        )
        if progress:
            progress(i + 1, total, path, "done")
    return res
