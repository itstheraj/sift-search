from __future__ import annotations
from pathlib import Path
from .chunking import chunk_text


def extract(path: Path, chunk_size: int, chunk_overlap: int) -> list[dict]:
    from bs4 import BeautifulSoup

    raw = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return [{"kind": "text", "text": c} for c in chunk_text(text, chunk_size, chunk_overlap)]
