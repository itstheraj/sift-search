from __future__ import annotations
from pathlib import Path
from .chunking import chunk_text


def extract(path: Path, chunk_size: int, chunk_overlap: int) -> list[dict]:
    import docx as _docx

    doc = _docx.Document(str(path))
    text = "\n".join((p.text for p in doc.paragraphs if p.text))
    return [{"kind": "text", "text": c} for c in chunk_text(text, chunk_size, chunk_overlap)]
