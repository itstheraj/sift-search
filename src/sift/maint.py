from __future__ import annotations

import shutil
from pathlib import Path

from . import config, db


def human_size(n: int | float) -> str:
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def dir_size(path: Path) -> int:
    total = 0
    if path.exists():
        for f in path.rglob("*"):
            if f.is_symlink() or not f.is_file():
                continue
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


def models_dir() -> Path:
    return config.models_dir()


def model_dirs() -> list[Path]:
    md = config.models_dir()
    if not md.exists():
        return [p for p in []]
    return [p for p in sorted(md.iterdir()) if p.is_dir() and p.name.startswith("models--")]


def models_listing() -> list[tuple[str, int]]:
    return [(p.name.replace("models--", "").replace("--", "/"), dir_size(p)) for p in model_dirs()]


def _find_dirs(*needles: str) -> list[Path]:
    out = []
    for p in model_dirs():
        low = p.name.lower()
        if all(n.lower() in low for n in needles if n):
            out.append(p)
    return out


def logical_models(cfg) -> list[dict]:
    rows: list[dict] = []
    if cfg.features.semantic:
        rows.append(
            {
                "key": "semantic",
                "label": "Semantic text",
                "repo": cfg.text_model,
                "dirs": _find_dirs(*cfg.text_model.split("/")),
            }
        )
    if cfg.features.image:
        rows.append(
            {
                "key": "image",
                "label": "Image",
                "repo": cfg.image_model,
                "dirs": _find_dirs(cfg.image_model),
            }
        )
    if cfg.features.asr:
        rows.append(
            {
                "key": "asr",
                "label": "Audio/video",
                "repo": f"faster-whisper-{cfg.asr_model}",
                "dirs": _find_dirs("whisper", cfg.asr_model),
            }
        )
    for r in rows:
        r["size"] = sum(dir_size(d) for d in r["dirs"])
        r["downloaded"] = bool(r["dirs"])
    return rows


def delete_dirs(dirs: list[Path]) -> int:
    freed = sum(dir_size(d) for d in dirs)
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)
    return freed


def models_total() -> int:
    return dir_size(config.models_dir())


def clear_models() -> int:
    md = config.models_dir()
    freed = dir_size(md)
    if md.exists():
        shutil.rmtree(md, ignore_errors=True)
    return freed


def index_paths() -> list[Path]:
    base = config.db_path()
    return [Path(str(base) + suffix) for suffix in ("", "-wal", "-shm")]


def index_size() -> int:
    return sum(p.stat().st_size for p in index_paths() if p.exists())


def index_stats() -> dict:
    base = config.db_path()
    if not base.exists():
        return {"files": 0, "chunks": 0, "by_status": {}, "by_kind": {}}
    con = db.connect(base)
    try:
        return db.stats(con)
    finally:
        con.close()


def clear_index() -> int:
    freed = index_size()
    for p in index_paths():
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass
    return freed
