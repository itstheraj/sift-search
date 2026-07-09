from __future__ import annotations
from pathlib import Path
from . import docx as docx_ex
from . import html as html_ex
from . import pdf as pdf_ex
from . import text as text_ex

TEXT_EXTS = {
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".org",
    ".log",
    ".csv",
    ".tsv",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".cc",
    ".rs",
    ".go",
    ".java",
    ".kt",
    ".rb",
    ".php",
    ".sh",
    ".bash",
    ".fish",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".ini",
    ".cfg",
    ".conf",
    ".sql",
    ".tex",
    ".xml",
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"}
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus", ".aac"}
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"}


def classify(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return "pdf"
    if ext in (".docx",):
        return "docx"
    if ext in (".html", ".htm"):
        return "html"
    if ext in TEXT_EXTS:
        return "text"
    if ext in IMAGE_EXTS:
        return "image"
    if ext in AUDIO_EXTS:
        return "audio"
    if ext in VIDEO_EXTS:
        return "video"
    return "unknown"


def extract(path: Path, kind: str, *, chunk_size: int, chunk_overlap: int) -> list[dict]:
    if kind == "text":
        return text_ex.extract(path, chunk_size, chunk_overlap)
    if kind == "pdf":
        return pdf_ex.extract(path, chunk_size, chunk_overlap)
    if kind == "docx":
        return docx_ex.extract(path, chunk_size, chunk_overlap)
    if kind == "html":
        return html_ex.extract(path, chunk_size, chunk_overlap)
    return []


def is_text_bearing(kind: str) -> bool:
    return kind in {"text", "pdf", "docx", "html"}
