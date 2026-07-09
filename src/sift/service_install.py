from __future__ import annotations
import shutil
import subprocess
import sys
from pathlib import Path
from .config import _xdg


def _units_dir() -> Path:
    return _xdg("XDG_CONFIG_HOME", Path.home() / ".config") / "systemd" / "user"


def _sift_exec() -> str:
    found = shutil.which("sift")
    return found if found else f"{sys.executable} -m sift.cli"


def install_service(enable: bool = True) -> list[Path]:
    units = _units_dir()
    units.mkdir(parents=True, exist_ok=True)
    sift = _sift_exec()
    service = units / "sift-reindex.service"
    service.write_text(
        f"[Unit]\nDescription=Sift: reindex configured folders\n\n[Service]\nType=oneshot\nExecStart={sift} reindex\nNice=10\nIOSchedulingClass=idle\n"
    )
    timer = units / "sift-reindex.timer"
    timer.write_text(
        "[Unit]\nDescription=Sift: daily reindex\n\n[Timer]\nOnCalendar=daily\nPersistent=true\n\n[Install]\nWantedBy=timers.target\n"
    )
    print("Installed systemd user units:")
    print(f"  {service}")
    print(f"  {timer}")
    if enable:
        try:
            subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
            subprocess.run(
                ["systemctl", "--user", "enable", "--now", "sift-reindex.timer"], check=True
            )
            print("Enabled daily timer (sift-reindex.timer).")
            print("Check it with:  systemctl --user list-timers sift-reindex.timer")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"note: could not enable timer automatically ({e}).")
            print("Enable manually:  systemctl --user enable --now sift-reindex.timer")
    else:
        print("Enable when ready:  systemctl --user enable --now sift-reindex.timer")
    return [service, timer]
