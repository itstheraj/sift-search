from __future__ import annotations
import os
import shutil
import subprocess
from .extractors import AUDIO_EXTS, VIDEO_EXTS

MEDIA_EXTS = AUDIO_EXTS | VIDEO_EXTS


def is_media(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in MEDIA_EXTS


def build_open_command(path: str, start_ms: int | None = None) -> list[str]:
    if start_ms and start_ms > 0 and is_media(path) and shutil.which("mpv"):
        return ["mpv", f"--start={start_ms / 1000:.3f}", path]
    return ["xdg-open", path]


def open_path(path: str, start_ms: int | None = None) -> None:
    cmd = build_open_command(path, start_ms)
    subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
