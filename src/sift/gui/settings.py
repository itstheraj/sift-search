from __future__ import annotations

import os
import shutil

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import config, maint, open_hit

PROFILES = ["light", "medium", "heavy"]
GPU_DEVICES = {"auto", "rocm", "vulkan"}
DEVICES = ["cpu", "auto", "rocm", "vulkan"]
FEATURES = [
    ("semantic", "Semantic text"),
    ("image", "Image search"),
    ("asr", "Audio/video transcription"),
    ("ocr", "OCR"),
]


def gpu_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


class ReindexWorker(QThread):
    progress = Signal(int, int, str)
    done = Signal(object)
    failed = Signal(str)

    def run(self):
        try:
            from .. import db, engine, indexer

            cfg = config.load()
            emb = engine.build_embedder(cfg)
            img = engine.build_image_embedder(cfg)
            tr = engine.build_transcriber(cfg)
            ocr = engine.build_ocr_engine(cfg)
            con = db.connect(config.db_path(), load_vec=emb is not None or img is not None)

            def cb(d, t, p, stage):
                self.progress.emit(
                    d, t, "" if stage == "done" else f"{stage}: {os.path.basename(str(p))}"
                )

            res = indexer.reindex(
                con,
                cfg,
                None,
                emb,
                img,
                tr,
                ocr,
                progress=cb,
                should_continue=lambda: not self.isInterruptionRequested(),
            )
            con.close()
            self.done.emit(res)
        except Exception as e:
            self.failed.emit(str(e))


class DownloadWorker(QThread):
    done = Signal()
    failed = Signal(str)

    def __init__(self, key):
        super().__init__()
        self.key = key

    def run(self):
        try:
            from .. import download

            download.download_for(self.key, config.load())
            self.done.emit()
        except Exception as e:
            self.failed.emit(str(e))


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sift Settings")
        self.resize(640, 720)
        self.cfg = config.load()
        root = QVBoxLayout(self)
        root.addWidget(self._search_group())
        root.addWidget(self._folders_group())
        root.addWidget(self._models_group())
        root.addWidget(self._storage_group())

        actions = QHBoxLayout()
        reindex = QPushButton("Reindex now")
        reindex.clicked.connect(self._reindex)
        save = QPushButton("Save")
        save.clicked.connect(self._save)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        actions.addWidget(reindex)
        actions.addStretch(1)
        actions.addWidget(cancel)
        actions.addWidget(save)
        root.addLayout(actions)

        self._rebuild_models()
        self._refresh_storage()

    def _search_group(self) -> QWidget:
        box = QGroupBox("Search modules")
        form = QFormLayout(box)
        self.profile = QComboBox()
        self.profile.addItems(PROFILES)
        self.profile.setCurrentText(self.cfg.profile)
        self.profile.currentTextChanged.connect(self._profile_changed)
        form.addRow("Profile", self.profile)

        has_tess = tesseract_available()
        self.feature_boxes = {}
        for key, label in FEATURES:
            cb = QCheckBox(label)
            cb.setChecked(getattr(self.cfg.features, key))
            if key == "ocr" and not has_tess and self.cfg.ocr_engine == "tesseract":
                cb.setChecked(False)
                cb.setEnabled(False)
                cb.setToolTip("Install the 'tesseract' package to enable OCR.")
            self.feature_boxes[key] = cb
            form.addRow("", cb)

        self.device = QComboBox()
        self.device.addItems(DEVICES)
        self.device.setCurrentText(self.cfg.device)
        if not gpu_available():
            for i in range(self.device.count()):
                if self.device.itemText(i) in GPU_DEVICES:
                    self.device.model().item(i).setEnabled(False)
            if self.cfg.device in GPU_DEVICES:
                self.device.setCurrentText("cpu")
        form.addRow("Device", self.device)
        if not gpu_available():
            note = QLabel(
                "No GPU build of PyTorch detected. Install ROCm/CUDA torch to enable "
                "GPU, then reopen settings."
            )
            note.setWordWrap(True)
            note.setStyleSheet("color:#8a929c;font-size:11px;")
            form.addRow("", note)
        return box

    def _profile_changed(self, name: str):
        defaults = config.PROFILES.get(name, {})
        has_tess = tesseract_available()
        for key, cb in self.feature_boxes.items():
            if not cb.isEnabled():
                continue
            cb.setChecked(bool(defaults.get(key, False)))
        ocr = self.feature_boxes.get("ocr")
        if ocr and not has_tess and self.cfg.ocr_engine == "tesseract":
            ocr.setChecked(False)

    def _folders_group(self) -> QWidget:
        box = QGroupBox("Indexed folders")
        layout = QVBoxLayout(box)
        self.folders = QListWidget()
        self.folders.addItems(self.cfg.folders)
        layout.addWidget(self.folders)
        row = QHBoxLayout()
        add = QPushButton("Add folder")
        add.clicked.connect(self._add_folder)
        rm = QPushButton("Remove selected")
        rm.clicked.connect(self._remove_folder)
        row.addWidget(add)
        row.addWidget(rm)
        row.addStretch(1)
        layout.addLayout(row)
        return box

    def _add_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Choose a folder to index")
        if d:
            self.folders.addItem(d)

    def _remove_folder(self):
        for item in self.folders.selectedItems():
            self.folders.takeItem(self.folders.row(item))

    def _models_group(self) -> QWidget:
        box = QGroupBox("Models")
        layout = QVBoxLayout(box)
        self.models_table = QTableWidget(0, 4)
        self.models_table.setHorizontalHeaderLabels(["Model", "Disk", "", ""])
        self.models_table.verticalHeader().setVisible(False)
        self.models_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.models_table.setSelectionMode(QAbstractItemView.NoSelection)
        h = self.models_table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.Stretch)
        for c in (1, 2, 3):
            h.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        layout.addWidget(self.models_table)
        return box

    def _rebuild_models(self):
        rows = maint.logical_models(self.cfg)
        self.models_table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            self.models_table.setItem(i, 0, QTableWidgetItem(r["repo"]))
            size = maint.human_size(r["size"]) if r["downloaded"] else "not downloaded"
            self.models_table.setItem(i, 1, QTableWidgetItem(size))
            delete = QPushButton("Delete")
            delete.setEnabled(r["downloaded"])
            delete.clicked.connect(lambda _, row=r: self._delete_model(row))
            self.models_table.setCellWidget(i, 2, delete)
            redl = QPushButton("Redownload" if r["downloaded"] else "Download")
            redl.clicked.connect(lambda _, key=r["key"]: self._download_model(key))
            self.models_table.setCellWidget(i, 3, redl)
        self.models_table.resizeColumnsToContents()
        self.models_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)

    def _delete_model(self, row):
        if QMessageBox.question(self, "Delete model", f"Delete {row['repo']}?") != QMessageBox.Yes:
            return
        freed = maint.delete_dirs(row["dirs"])
        QMessageBox.information(self, "Deleted", f"Freed {maint.human_size(freed)}.")
        self._rebuild_models()
        self._refresh_storage()

    def _download_model(self, key):
        dlg = QProgressDialog("Downloading model…", None, 0, 0, self)
        dlg.setWindowTitle("Download")
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        self._dw = DownloadWorker(key)

        def finish_ok():
            dlg.reset()
            self._rebuild_models()
            self._refresh_storage()

        self._dw.done.connect(finish_ok)
        self._dw.failed.connect(
            lambda m: (dlg.reset(), QMessageBox.critical(self, "Download failed", m))
        )
        self._dw.start()
        dlg.show()

    def _storage_group(self) -> QWidget:
        box = QGroupBox("Storage")
        layout = QVBoxLayout(box)
        self.info = QLabel()
        self.info.setWordWrap(True)
        layout.addWidget(self.info)
        row = QHBoxLayout()
        open_btn = QPushButton("Open data folder")
        open_btn.clicked.connect(lambda: open_hit.open_path(str(config.data_dir())))
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(lambda: (self._rebuild_models(), self._refresh_storage()))
        clr_idx = QPushButton("Clear index")
        clr_idx.clicked.connect(self._clear_index)
        clr_mdl = QPushButton("Clear all models")
        clr_mdl.clicked.connect(self._clear_models)
        for b in (open_btn, refresh, clr_idx, clr_mdl):
            row.addWidget(b)
        layout.addLayout(row)
        return box

    def _refresh_storage(self):
        stats = maint.index_stats()
        self.info.setText(
            f"<b>Config:</b> {config.config_path()}<br>"
            f"<b>Index:</b> {config.db_path()} - "
            f"{maint.human_size(maint.index_size())}, "
            f"{stats['files']} files, {stats['chunks']} chunks<br>"
            f"<b>Models:</b> {maint.models_dir()} - "
            f"{maint.human_size(maint.models_total())} total"
        )

    def _collect(self) -> config.Config:
        cfg = config.load()
        cfg.profile = self.profile.currentText()
        cfg.device = self.device.currentText()
        cfg.features = config.Features(
            text=True,
            semantic=self.feature_boxes["semantic"].isChecked(),
            image=self.feature_boxes["image"].isChecked(),
            asr=self.feature_boxes["asr"].isChecked(),
            ocr=self.feature_boxes["ocr"].isChecked(),
        )
        cfg.folders = [self.folders.item(i).text() for i in range(self.folders.count())]
        return cfg

    def _persist_and_reload(self):
        config.save(self._collect())
        self.cfg = config.load()
        parent = self.parent()
        if parent is not None and hasattr(parent, "reload_engine"):
            parent.reload_engine()

    def _clear_index(self):
        if (
            QMessageBox.question(
                self,
                "Clear index",
                "Delete the search index and the list of indexed folders?",
            )
            != QMessageBox.Yes
        ):
            return
        freed = maint.clear_index()
        self.folders.clear()
        self._persist_and_reload()
        QMessageBox.information(self, "Index cleared", f"Freed {maint.human_size(freed)}.")
        self._refresh_storage()

    def _clear_models(self):
        if (
            QMessageBox.question(
                self,
                "Clear models",
                "Delete all downloaded models? They download again on next use.",
            )
            != QMessageBox.Yes
        ):
            return
        freed = maint.clear_models()
        QMessageBox.information(self, "Models cleared", f"Freed {maint.human_size(freed)}.")
        self._rebuild_models()
        self._refresh_storage()

    def _reindex(self):
        config.save(self._collect())
        cfg = config.load()
        if not cfg.folders:
            QMessageBox.warning(self, "Reindex", "No folders configured. Add a folder first.")
            return
        dlg = QProgressDialog("Loading models…", "Cancel", 0, 0, self)
        dlg.setWindowTitle("Reindexing")
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        self._rw = ReindexWorker()

        def on_progress(d, t, desc):
            if t:
                dlg.setMaximum(t)
                dlg.setValue(d)
            if desc:
                dlg.setLabelText(desc)

        def on_done(res):
            dlg.reset()
            self._rebuild_models()
            self._refresh_storage()
            parent = self.parent()
            if parent is not None and hasattr(parent, "reload_engine"):
                parent.reload_engine()
            QMessageBox.information(
                self,
                "Reindex complete",
                f"indexed={res.indexed} skipped={res.skipped} "
                f"deferred={res.deferred} errors={res.errors}",
            )

        self._rw.progress.connect(on_progress)
        self._rw.done.connect(on_done)
        self._rw.failed.connect(
            lambda m: (dlg.reset(), QMessageBox.critical(self, "Reindex failed", m))
        )
        dlg.canceled.connect(self._rw.requestInterruption)
        self._rw.start()
        dlg.show()

    def _save(self):
        self._persist_and_reload()
        QMessageBox.information(self, "Saved", "Settings saved and applied.")
        self.accept()
