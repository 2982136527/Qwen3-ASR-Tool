"""Batch transcription tab.

Left: file list (drop files/folders or use buttons), right: run controls, SRT
options and per-file progress. Start spins up a BatchWorker; every segment's
text streams into the shared log panel as it's decoded, and per-row status flips
to 等待/处理中/完成.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox,
                             QFileDialog, QGridLayout, QGroupBox, QHBoxLayout,
                             QLabel, QLineEdit, QMessageBox, QProgressBar,
                             QPushButton, QSpinBox, QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QWidget)

from ..audio import is_supported
from ..config import Settings
from ..transcriber import BatchWorker


class BatchWidget(QWidget):
    COL_FILE = 0
    COL_TYPE = 1
    COL_STATUS = 2
    COL_PROG = 3

    def __init__(self, model_manager, settings: Settings, log_panel, parent=None):
        super().__init__(parent)
        self.model_manager = model_manager
        self.settings = settings
        self.log_panel = log_panel
        self.worker: Optional[BatchWorker] = None
        self._build()
        self._sync_options()

    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # --- file list side ---
        left = QVBoxLayout()
        left.setSpacing(6)
        bar = QHBoxLayout()
        self.btn_add_files = QPushButton("添加文件")
        self.btn_add_dir = QPushButton("添加目录")
        self.btn_remove = QPushButton("移除所选")
        self.btn_clear = QPushButton("清空")
        for b in (self.btn_add_files, self.btn_add_dir, self.btn_remove, self.btn_clear):
            bar.addWidget(b)
        bar.addStretch(1)
        left.addLayout(bar)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["文件", "类型", "状态", "进度"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.setColumnWidth(self.COL_FILE, 420)
        self.table.setColumnWidth(self.COL_TYPE, 64)
        self.table.setColumnWidth(self.COL_STATUS, 80)
        self.table.horizontalHeader().setSectionResizeMode(self.COL_PROG, self.table.horizontalHeader().ResizeMode.Stretch)
        self.table.setAcceptDrops(True)
        self.table.setDragDropMode(QAbstractItemView.DragDropMode.DropOnly)
        self.table.dragEnterEvent = self._drag_enter
        self.table.dropEvent = self._drop
        self.table.setDragEnabled(False)
        left.addWidget(self.table, 1)

        self.btn_add_files.clicked.connect(self._pick_files)
        self.btn_add_dir.clicked.connect(self._pick_dir)
        self.btn_remove.clicked.connect(self._remove_selected)
        self.btn_clear.clicked.connect(self._clear)

        root.addLayout(left, 3)

        # --- options side ---
        right = QVBoxLayout()
        right.setSpacing(10)

        run_box = QGroupBox("运行")
        rb = QVBoxLayout(run_box)
        self.btn_start = QPushButton("开始转录")
        self.btn_start.setObjectName("primary")
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setEnabled(False)
        self.overall = QProgressBar(); self.overall.setFormat("总进度 %p%")
        rb.addWidget(self.overall)
        rh = QHBoxLayout()
        rh.addWidget(self.btn_start); rh.addWidget(self.btn_stop)
        rb.addLayout(rh)
        self.lbl_count = QLabel("0 个文件")
        rb.addWidget(self.lbl_count)
        right.addWidget(run_box)

        opt_box = QGroupBox("字幕与识别设置")
        og = QGridLayout(opt_box)
        og.setHorizontalSpacing(8); og.setVerticalSpacing(8)
        self.cmb_lang = QComboBox()
        self.cmb_lang.addItems(["auto", "Chinese", "汉字", "English", "Japanese", "Korean",
                                "Cantonese", "French", "German", "Spanish", "Russian", "Italian",
                                "Portuguese", "Thai", "Vietnamese", "Arabic", "Indonesian", "Malay"])
        self.cmb_align = QCheckBox("开启时间戳对齐(精准时间轴)")
        self.cmb_align.setToolTip("使用 Qwen3-ForcedAligner-0.6B 对齐字符/单词级时间戳,字幕时间轴精准匹配人声。关闭则用分段时长近似。")
        self.sp_chars = QSpinBox(); self.sp_chars.setRange(8, 64); self.sp_chars.setSuffix(" 字/行")
        self.sp_dur = QSpinBox(); self.sp_dur.setRange(1, 30); self.sp_dur.setSuffix(" 秒/段")
        self.cmb_merge = QCheckBox("合并空白字幕")
        og.addWidget(QLabel("语种识别"), 0, 0); og.addWidget(self.cmb_lang, 0, 1)
        og.addWidget(self.cmb_align, 1, 0, 1, 2)
        og.addWidget(self.sp_chars, 2, 0); og.addWidget(self.sp_dur, 2, 1)
        og.addWidget(self.cmb_merge, 3, 0, 1, 2)
        right.addWidget(opt_box)

        out_box = QGroupBox("输出")
        ob = QHBoxLayout(out_box)
        self.lbl_out = QLabel("")
        self.btn_out = QPushButton("选择…")
        ob.addWidget(self.lbl_out, 1); ob.addWidget(self.btn_out)
        right.addWidget(out_box)

        right.addStretch(1)
        root.addLayout(right, 1)

        self.btn_start.clicked.connect(self._start)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_out.clicked.connect(self._pick_output)
        self.cmb_align.stateChanged.connect(self._commit_options)
        self.sp_chars.valueChanged.connect(self._commit_options)
        self.sp_dur.valueChanged.connect(self._commit_options)
        self.cmb_merge.stateChanged.connect(self._commit_options)
        self.cmb_lang.currentTextChanged.connect(self._commit_options)

        self._set_output(self.settings.output_dir)

    # ---- options ----
    def _sync_options(self):
        self.cmb_lang.setCurrentText(self.settings.language)
        self.cmb_align.setChecked(self.settings.enable_aligner)
        self.sp_chars.setValue(self.settings.max_chars_per_line)
        self.sp_dur.setValue(int(self.settings.max_segment_seconds))
        self.cmb_merge.setChecked(self.settings.merge_blank_lines)

    def _commit_options(self, *a):
        self.settings.language = self.cmb_lang.currentText()
        self.settings.enable_aligner = self.cmb_align.isChecked()
        self.settings.max_chars_per_line = self.sp_chars.value()
        self.settings.max_segment_seconds = float(self.sp_dur.value())
        self.settings.merge_blank_lines = self.cmb_merge.isChecked()

    def _set_output(self, p: str):
        self.settings.output_dir = p
        pp = Path(p)
        if not pp.is_absolute():
            pp = Path(__file__).resolve().parent.parent.parent / p
        self.lbl_out.setText(str(pp))

    def _pick_output(self):
        d = QFileDialog.getExistingDirectory(self, "选择输出目录", self.lbl_out.text())
        if d:
            self._set_output(d)

    # ---- file list ----
    def _drag_enter(self, e):
        e.accept()

    def _drop(self, e):
        urls = e.mimeData().urls()
        paths = []
        for u in urls:
            p = u.toLocalFile()
            if p:
                paths.append(p)
        self._add_paths(paths)

    def _pick_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择音视频文件", str(Path.home() / "Downloads"),
            "音视频 (*.mp4 *.mkv *.mov *.avi *.mp3 *.m4a *.flac *.aac *.wav *.ogg *.opus *.webm *.ts *.strm *.m2ts);;所有文件 (*)")
        self._add_paths(files)

    def _pick_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择目录(递归扫描)", str(Path.home() / "Downloads"))
        if not d:
            return
        found = [str(p) for p in Path(d).rglob("*") if p.is_file() and is_supported(str(p))]
        self._add_paths(found)

    def _add_paths(self, paths):
        added = 0
        existing = {self.table.item(r, self.COL_FILE).text() for r in range(self.table.rowCount())}
        for p in paths:
            p = str(p)
            if not is_supported(p):
                continue
            if p in existing:
                continue
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setItem(r, self.COL_FILE, QTableWidgetItem(Path(p).name))
            self.table.item(r, self.COL_FILE).setToolTip(p)
            self.table.setItem(r, self.COL_TYPE, QTableWidgetItem(self._type_tag(p)))
            self.table.setItem(r, self.COL_STATUS, QTableWidgetItem("等待"))
            bar = QProgressBar(); bar.setRange(0, 100)
            self.table.setCellWidget(r, self.COL_PROG, bar)
            self.table.item(r, self.COL_FILE).setData(Qt.ItemDataRole.UserRole, p)
            added += 1
        self.lbl_count.setText(f"{self.table.rowCount()} 个文件")

    def _type_tag(self, p):
        pl = p.lower()
        if pl.endswith(".strm"):
            return "STRM"
        if pl.startswith(("http://", "https://")):
            return "URL"
        return Path(p).suffix.lstrip(".").upper() or "FILE"

    def _remove_selected(self):
        for r in sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(r)
        self.lbl_count.setText(f"{self.table.rowCount()} 个文件")

    def _clear(self):
        self.table.setRowCount(0)
        self.lbl_count.setText("0 个文件")

    def _files(self) -> List[str]:
        out = []
        for r in range(self.table.rowCount()):
            it = self.table.item(r, self.COL_FILE)
            p = it.data(Qt.ItemDataRole.UserRole) if it else None
            if p:
                out.append(str(p))
        return out

    # ---- run ----
    def _start(self):
        files = self._files()
        if not files:
            QMessageBox.information(self, "无文件", "请先添加音视频文件或 .strm")
            return
        for r in range(self.table.rowCount()):
            self.table.item(r, self.COL_STATUS).setText("等待")
            self.table.cellWidget(r, self.COL_PROG).setValue(0)
        self.overall.setValue(0)
        self.btn_start.setEnabled(False); self.btn_stop.setEnabled(True)

        self._commit_options()
        from ..config import save_settings
        save_settings(self.settings)
        self.worker = BatchWorker(self.model_manager, files, self.settings)
        self.worker.log.connect(self.log_panel.append)
        self.worker.file_started.connect(self._on_file_started)
        self.worker.file_progress.connect(self._on_file_progress)
        self.worker.file_done.connect(self._on_file_done)
        self.worker.finished_all.connect(self._on_finished_all)
        self.worker.start()

    def _stop(self):
        if self.worker:
            self.worker.cancel()
        self.btn_stop.setEnabled(False)

    def _row_for_index(self, idx: int, filename: str = "") -> int:
        for r in range(self.table.rowCount()):
            it = self.table.item(r, self.COL_FILE)
            f = it.data(Qt.ItemDataRole.UserRole) if it else ""
            if Path(str(f)).name == filename or r == idx:
                try:
                    if r == idx:
                        return r
                except Exception:
                    pass
        return idx if 0 <= idx < self.table.rowCount() else -1

    def _on_file_started(self, idx: int, filename: str):
        r = self._row_for_index(idx, filename)
        if r >= 0:
            self.table.item(r, self.COL_STATUS).setText("处理中")

    def _on_file_progress(self, idx: int, pct: float):
        r = idx if 0 <= idx < self.table.rowCount() else -1
        if r >= 0:
            self.table.cellWidget(r, self.COL_PROG).setValue(int(pct * 100))
        done = sum(1 for i in range(self.table.rowCount())
                   if self.table.item(i, self.COL_STATUS).text() in ("完成", "失败", "取消"))
        self.overall.setValue(int((done / max(1, self.table.rowCount())) * 100))

    def _on_file_done(self, idx: int, srt_path: str, ok: bool, message: str):
        r = idx if 0 <= idx < self.table.rowCount() else -1
        if r >= 0:
            self.table.item(r, self.COL_STATUS).setText("完成" if ok else "失败")
            self.table.cellWidget(r, self.COL_PROG).setValue(100 if ok else 0)
            if ok:
                self.table.item(r, self.COL_STATUS).setToolTip(srt_path)
        self.overall.setValue(int((idx + 1) / max(1, self.table.rowCount()) * 100))

    def _on_finished_all(self, done: int, failed: int):
        self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False)
        self.overall.setValue(100)
        self.log_panel.append(f"批量完成: 成功 {done} 个, 失败 {failed} 个", "ok" if failed == 0 else "warn")
