from __future__ import annotations
import html
import os
import re
from datetime import datetime
from PySide6.QtCore import Qt, QThread, QTimer, QSize, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from .. import config, open_hit
from .settings import SettingsDialog

KIND_ICON = {
    "image": "🖼",
    "audio": "🎵",
    "video": "🎬",
    "pdf": "📄",
    "text": "📝",
    "docx": "📄",
    "html": "🌐",
}
_WORD = re.compile("\\w+", re.UNICODE)


def _fmt_ts(ms):
    if not ms:
        return ""
    s = ms // 1000
    return f"⏱ {s // 60}:{s % 60:02d}"


def _human_size(n):
    if not n:
        return ""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _fmt_date(ts):
    if not ts:
        return None
    try:
        return datetime.fromtimestamp(ts).strftime("%b %d, %Y %H:%M")
    except (OSError, ValueError, OverflowError):
        return None


def _created_ts(path):
    try:
        st = os.stat(path)
        return getattr(st, "st_birthtime", None) or st.st_ctime
    except OSError:
        return None


def _highlight(text, query, limit=160):
    text = " ".join(text.split())
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0] + "…"
    esc = html.escape(text)
    terms = [re.escape(t) for t in _WORD.findall(query) if len(t) >= 2]
    if terms:
        pat = re.compile("(" + "|".join(terms) + ")", re.IGNORECASE)
        esc = pat.sub('<span style="background:#f6c453;color:#101010;">\\1</span>', esc)
    return esc


def _thumbnail(hit):
    label = QLabel()
    label.setFixedSize(48, 48)
    label.setAlignment(Qt.AlignCenter)
    if hit.kind == "image":
        pm = QPixmap(hit.path)
        if not pm.isNull():
            label.setPixmap(pm.scaled(48, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            return label
    label.setText(KIND_ICON.get(hit.kind, "•"))
    f = label.font()
    f.setPointSize(20)
    label.setFont(f)
    return label


class ResultCard(QWidget):
    def __init__(self, hit, query):
        super().__init__()
        name = os.path.basename(hit.path)
        ext = os.path.splitext(hit.path)[1].lstrip(".").upper() or hit.kind.upper()
        meta_bits = []
        loc = _fmt_ts(hit.start_ms) or (f"p.{hit.page}" if hit.page else "")
        if loc:
            meta_bits.append(loc)
        mod = _fmt_date(hit.mtime)
        if mod:
            meta_bits.append(f"Modified {mod}")
        crt = _fmt_date(_created_ts(hit.path))
        if crt and crt != mod:
            meta_bits.append(f"Created {crt}")
        meta_bits.append(os.path.dirname(hit.path))

        outer = QHBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(8)
        outer.addWidget(_thumbnail(hit), 0, Qt.AlignTop)

        text = QVBoxLayout()
        text.setSpacing(1)
        title = QLabel(
            f"<b>{html.escape(name)}</b>&nbsp;&nbsp;"
            f"<span style='color:#8a929c;'>{ext} · {_human_size(hit.size)}</span>"
        )
        meta = QLabel(
            f"<span style='color:#8a929c;font-size:11px;'>{html.escape('   ·   '.join(meta_bits))}</span>"
        )
        snippet = QLabel(_highlight(hit.snippet, query))
        for lbl in (title, meta, snippet):
            lbl.setTextFormat(Qt.RichText)
            lbl.setTextInteractionFlags(Qt.NoTextInteraction)
            text.addWidget(lbl)
        outer.addLayout(text, 1)


class SearchWorker(QThread):
    ready = Signal()
    results = Signal(list)
    failed = Signal(str)

    def __init__(self, path=None):
        super().__init__()
        self._path = path
        self._engine = None
        self._pending = None
        self._running = True

    def run(self):
        try:
            from ..engine import SearchEngine

            self._engine = SearchEngine.from_config()
            self.ready.emit()
        except Exception as e:
            self.failed.emit(str(e))
            return
        while self._running:
            q = self._pending
            self._pending = None
            if q is not None:
                try:
                    hits = self._engine.search(q, limit=40, path_prefix=self._path)
                    self.results.emit(hits)
                except Exception as e:
                    self.failed.emit(str(e))
            self.msleep(40)

    def query(self, text: str):
        self._pending = text

    def stop(self):
        self._running = False


class MainWindow(QWidget):
    def __init__(self, worker: SearchWorker, initial_query: str = "", path=None):
        super().__init__()
        self.worker = worker
        self._path = path
        self.setWindowTitle("Sift" + (f" - {path}" if path else ""))
        self.resize(760, 520)
        layout = QVBoxLayout(self)
        self.input = QLineEdit()
        self.input.setPlaceholderText("Search your files…")
        self.input.setClearButtonEnabled(True)
        self.status = QLabel("Loading models…")
        self.list = QListWidget()
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list.setAlternatingRowColors(True)
        self.list.setSpacing(2)
        self.list.setUniformItemSizes(False)
        header = QHBoxLayout()
        header.addWidget(self.input)
        self.settings_btn = QPushButton("⚙")
        self.settings_btn.setFixedWidth(36)
        self.settings_btn.setToolTip("Settings")
        self.settings_btn.clicked.connect(self._open_settings)
        header.addWidget(self.settings_btn)
        layout.addLayout(header)
        layout.addWidget(self.status)
        layout.addWidget(self.list)
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(250)
        self._debounce.timeout.connect(self._fire)
        self.input.textChanged.connect(lambda _: self._debounce.start())
        self.input.returnPressed.connect(self._fire)
        self.list.itemActivated.connect(self._open)
        self._wire_worker()
        if initial_query:
            self.input.setText(initial_query)

    def _wire_worker(self):
        self.worker.ready.connect(self._on_ready)
        self.worker.results.connect(self._on_results)
        self.worker.failed.connect(lambda m: self.status.setText(f"Error: {m}"))

    def reload_engine(self):
        old = self.worker
        old.stop()
        old.wait(2000)
        self.list.clear()
        self.status.setText("Loading models…")
        self.worker = SearchWorker(path=self._path)
        self._wire_worker()
        self.worker.start()

    def _open_settings(self):
        SettingsDialog(self).exec()

    def _on_ready(self):
        self.status.setText("Ready")
        if self.input.text().strip():
            self._fire()

    def _fire(self):
        q = self.input.text().strip()
        if q:
            self.status.setText("Searching…")
            self.worker.query(q)

    def _on_results(self, hits):
        self.list.clear()
        query = self.input.text()
        for h in hits:
            card = ResultCard(h, query)
            item = QListWidgetItem(self.list)
            item.setData(Qt.UserRole, (h.path, h.start_ms))
            item.setSizeHint(QSize(0, card.sizeHint().height()))
            self.list.addItem(item)
            self.list.setItemWidget(item, card)
        self.status.setText(f"{len(hits)} result(s)" if hits else "No results")

    def _open(self, item: QListWidgetItem):
        path, start_ms = item.data(Qt.UserRole)
        open_hit.open_path(path, start_ms)

    def shutdown(self):
        try:
            self.worker.stop()
            self.worker.wait(2000)
        except RuntimeError:
            pass

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)


def _entrypoint() -> int:
    import argparse

    p = argparse.ArgumentParser(prog="sift-search")
    p.add_argument("query", nargs="?", default="")
    p.add_argument("--path", default=None)
    a = p.parse_args()
    return run_gui(initial_query=a.query, path=a.path)


def run_gui(initial_query: str = "", path: str | None = None) -> int:
    import signal

    config.write_default_config()
    app = QApplication.instance() or QApplication([])
    worker = SearchWorker(path=path)
    win = MainWindow(worker, initial_query=initial_query, path=path)
    app.aboutToQuit.connect(win.shutdown)

    signal.signal(signal.SIGINT, lambda *_: app.quit())
    waker = QTimer()
    waker.start(200)
    waker.timeout.connect(lambda: None)

    worker.start()
    win.show()
    return app.exec()
