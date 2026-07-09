from __future__ import annotations
from pathlib import Path
from .chunking import chunk_text


def extract(path: Path, chunk_size: int, chunk_overlap: int) -> list[dict]:
    from pypdf import PdfReader

    out: list[dict] = []
    reader = PdfReader(str(path))
    for page_no, page in enumerate(reader.pages, start=1):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        for c in chunk_text(txt, chunk_size, chunk_overlap):
            out.append({"kind": "text", "text": c, "page": page_no})
    return out
