from __future__ import annotations
import os
import shutil
import stat
import sys
from pathlib import Path
from .config import _xdg


def _data_home() -> Path:
    return _xdg("XDG_DATA_HOME", Path.home() / ".local" / "share")


def _sift_exec() -> str:
    found = shutil.which("sift")
    if found:
        return found
    return f"{sys.executable} -m sift.cli"


def _write(path: Path, content: str, executable: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def install_kde() -> list[Path]:
    data = _data_home()
    sift = _sift_exec()
    written: list[Path] = []
    written.append(
        _write(
            data / "krunner" / "dbusplugins" / "sift.desktop",
            "[Desktop Entry]\nName=Sift\nComment=Local semantic + full-text file search\nType=Service\nX-KDE-PluginInfo-Name=sift\nX-KDE-PluginInfo-Version=1.0\nX-KDE-PluginInfo-EnabledByDefault=true\nX-Plasma-API=DBus\nX-Plasma-DBusRunner-Service=org.sift.krunner\nX-Plasma-DBusRunner-Path=/sift\nX-Plasma-Runner-Syntaxes=:q:\n",
        )
    )
    written.append(
        _write(
            data / "dbus-1" / "services" / "org.sift.krunner.service",
            f"[D-BUS Service]\nName=org.sift.krunner\nExec={sift} krunner\n",
        )
    )
    written.append(
        _write(
            data / "kio" / "servicemenus" / "sift-search.desktop",
            f"[Desktop Entry]\nType=Application\nName=Sift\nIcon=system-search\nMimeType=inode/directory;\nActions=siftSearchHere;siftReindexHere;\nNoDisplay=true\nX-KDE-Submenu=Sift\n\n[Desktop Action siftSearchHere]\nName=Search here with Sift\nIcon=system-search\nExec={sift} gui --path %f\n\n[Desktop Action siftReindexHere]\nName=Reindex this folder with Sift\nIcon=view-refresh\nExec={sift} reindex %f\n",
            executable=True,
        )
    )
    written.append(
        _write(
            data / "applications" / "sift-search.desktop",
            f"[Desktop Entry]\nType=Application\nName=Sift Search\nComment=Search your files locally\nIcon=system-search\nExec={sift} gui\nCategories=Utility;\nTerminal=false\n",
        )
    )
    print("Installed KDE integration:")
    for p in written:
        print(f"  {p}")
    print("\nReload KRunner to pick up the new runner:")
    print("  kquitapp6 krunner   (it restarts on next use)")
    print(
        "Then open KRunner (Alt+Space) and type a query, or right-click a folder in Dolphin → Sift."
    )
    if "XDG_DATA_HOME" in os.environ:
        print("\nnote: XDG_DATA_HOME is set; files went there, not ~/.local/share.")
    return written
