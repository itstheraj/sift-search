from __future__ import annotations
from pathlib import Path
from .chunking import chunk_text

MAX_BYTES = 20 * 1024 * 1024


def extract(path: Path, chunk_size: int, chunk_overlap: int) -> list[dict]:
    if path.stat().st_size > MAX_BYTES:
        return []
    raw = path.read_text(encoding="utf-8", errors="replace")
    return [{"kind": "text", "text": c} for c in chunk_text(raw, chunk_size, chunk_overlap)]
