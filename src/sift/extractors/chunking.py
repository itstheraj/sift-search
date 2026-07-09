from __future__ import annotations


def chunk_text(text: str, size: int = 1000, overlap: int = 150) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    overlap = max(0, min(overlap, size - 1))
    chunks: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + size, n)
        if end < n:
            space = text.rfind(" ", i, end)
            if space > i:
                end = space
        piece = text[i:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        i = max(end - overlap, i + 1)
    return chunks
