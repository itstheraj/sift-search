from __future__ import annotations
import re
import sqlite3
from dataclasses import dataclass

RRF_K = 60
_WORD = re.compile("\\w+", re.UNICODE)


def cosine_from_l2(distance: float) -> float:
    return 1.0 - (distance * distance) / 2.0


@dataclass
class Hit:
    chunk_id: int
    file_id: int
    path: str
    kind: str
    chunk_kind: str
    snippet: str
    score: float
    start_ms: int | None = None
    page: int | None = None
    mtime: float | None = None
    size: int | None = None


def _quote(term: str) -> str:
    return '"' + term.replace('"', '""') + '"'


def _porter_query(query: str) -> str | None:
    terms = _WORD.findall(query)
    if not terms:
        return None
    return " ".join((f"{_quote(t)}*" for t in terms))


def _trigram_query(query: str) -> str | None:
    q = query.strip()
    if len(q) < 3:
        return None
    return _quote(q)


def _fts_ranked(con: sqlite3.Connection, table: str, match: str, limit: int) -> list[int]:
    try:
        rows = con.execute(
            f"SELECT rowid FROM {table} WHERE {table} MATCH ? ORDER BY bm25({table}) LIMIT ?",
            (match, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [r[0] for r in rows]


def _vec_ranked(con, table: str, blob: bytes, limit: int, min_cos: float) -> list[int]:
    from . import db

    ids: list[int] = []
    for cid, distance in db.vec_knn(con, table, blob, limit):
        if cosine_from_l2(distance) < min_cos:
            break
        ids.append(cid)
    return ids


def _rrf(rankings: list[list[int]]) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
    return scores


def _rrf_files(
    rankings: list[list[int]], chunk_file: dict[int, int], lexical: set[int]
) -> tuple[dict[int, float], dict[int, tuple[int, int, int]]]:
    scores: dict[int, float] = {}
    best: dict[int, tuple[int, int, int]] = {}
    for ranking in rankings:
        seen: set[int] = set()
        for rank, cid in enumerate(ranking):
            fid = chunk_file.get(cid)
            if fid is None or fid in seen:
                continue
            seen.add(fid)
            scores[fid] = scores.get(fid, 0.0) + 1.0 / (RRF_K + rank + 1)
            key = (rank, 0 if cid in lexical else 1, cid)
            if fid not in best or key < best[fid]:
                best[fid] = key
    return scores, best


def _snippet(text: str, query: str, width: int = 240) -> str:
    if not text:
        return ""
    terms = _WORD.findall(query.lower())
    low = text.lower()
    pos = -1
    for t in terms:
        pos = low.find(t)
        if pos != -1:
            break
    if pos == -1:
        return text[:width].strip() + ("…" if len(text) > width else "")
    start = max(0, pos - width // 3)
    end = min(len(text), start + width)
    snip = text[start:end].strip()
    return ("…" if start > 0 else "") + snip + ("…" if end < len(text) else "")


def search(
    con: sqlite3.Connection,
    query: str,
    limit: int = 20,
    pool: int = 100,
    embedder=None,
    image_embedder=None,
    path_prefix: str | None = None,
    text_min_similarity: float = 0.0,
    image_min_similarity: float = 0.0,
) -> list[Hit]:
    from .models.text_embed import serialize

    rankings: list[list[int]] = []
    lexical: set[int] = set()
    pq = _porter_query(query)
    if pq:
        hits = _fts_ranked(con, "chunks_fts", pq, pool)
        lexical.update(hits)
        rankings.append(hits)
    tq = _trigram_query(query)
    if tq:
        hits = _fts_ranked(con, "chunks_tri", tq, pool)
        lexical.update(hits)
        rankings.append(hits)
    if embedder is not None:
        vecs = embedder.encode([query])
        if vecs:
            rankings.append(
                _vec_ranked(con, "vec_text", serialize(vecs[0]), pool, text_min_similarity)
            )
    if image_embedder is not None:
        ivecs = image_embedder.encode_text([query])
        if ivecs:
            rankings.append(
                _vec_ranked(con, "vec_image", serialize(ivecs[0]), pool, image_min_similarity)
            )
    if not any(rankings):
        return []

    candidates = {cid for ranking in rankings for cid in ranking}
    if not candidates:
        return []
    placeholders = ",".join("?" * len(candidates))
    sql = f"""
        SELECT c.id, c.file_id, c.kind AS chunk_kind, c.text, c.start_ms, c.page,
               f.path, f.kind AS file_kind, f.mtime, f.size
        FROM chunks c JOIN files f ON f.id = c.file_id
        WHERE c.id IN ({placeholders})
    """
    params = list(candidates)
    if path_prefix:
        sql += " AND f.path LIKE ?"
        params.append(path_prefix.rstrip("/") + "/%")
    rows = {r["id"]: r for r in con.execute(sql, params)}
    if not rows:
        return []

    chunk_file = {cid: r["file_id"] for cid, r in rows.items()}
    scores, best = _rrf_files(rankings, chunk_file, lexical)
    if not scores:
        return []
    ranked_files = sorted(scores, key=lambda f: (-scores[f], best[f][1], best[f][2]))

    hits: list[Hit] = []
    for fid in ranked_files[:limit]:
        cid = best[fid][2]
        r = rows[cid]
        hits.append(
            Hit(
                chunk_id=cid,
                file_id=fid,
                path=r["path"],
                kind=r["file_kind"],
                chunk_kind=r["chunk_kind"],
                snippet=_snippet(r["text"] or "", query),
                score=scores[fid],
                start_ms=r["start_ms"],
                page=r["page"],
                mtime=r["mtime"],
                size=r["size"],
            )
        )
    return hits
