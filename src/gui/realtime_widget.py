"""Real-time dictation tab.

Start/Stop listening, a live VU meter, language + VAD sensitivity + pause
controls, and a running transcript that appends each recognized utterance as it
lands. Saves the full transcript to a .txt on demand.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                             QGroupBox, QHBoxLayout, QLabel, QMessageBox,
                             QPlainTextEdit, QProgressBar, QPushButton,
                             QSlider, QSpinBox, QVBoxLayout, QWidget)

from ..config import Settings
from ..realtime import RealtimeWorker


class RealtimeWidget(QWidget):
    def __init__(self, model_manager, settings: Settings, log_panel, parent=None):
        super().__init__(parent)
        self.model_manager = model_manager
        self.settings = settings
        self.log_panel = log_panel
        self.worker: RealtimeWorker | None = None
        self._build()
        self._sync_options()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12); root.setSpacing(10)

        top = QHBoxLayout()
        self.btn_start = QPushButton("开始听写")
        self.btn_start.setObjectName("primary")
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setEnabled(False)
        self.btnSave = QPushButton("保存文本…")
        self.btn_clear = QPushButton("清空")
        top.addWidget(self.btn_start); top.addWidget(self.btn_stop)
        top.addStretch(1)
        top.addWidget(self.btnSave); top.addWidget(self.btn_clear)
        root.addLayout(top)

        meter_row = QHBoxLayout()
        self.meter = QProgressBar(); self.meter.setRange(0, 100); self.meter.setMaximumHeight(14)
        self.meter.setFormat("音量")
        self.lbl_state = QLabel("待机")
        self.lbl_state.setFixedWidth(72)
        meter_row.addWidget(self.lbl_state); meter_row.addWidget(self.meter, 1)
        root.addLayout(meter_row)

        opts = QGroupBox("听写设置")
        og = QHBoxLayout(opts); og.setSpacing(10)
        self.cmb_lang = QComboBox()
        self.cmb_lang.addItems(["auto", "Chinese", "English", "Japanese", "Korean", "Cantonese"])
        sl = QLabel("灵敏度"); self.sl_sens = QSlider(Qt.Orientation.Horizontal)
        self.sl_sens.setRange(1, 5); self.sl_sens.setFixedWidth(120)
        self.lbl_sens = QLabel("3")
        self.sp_pause = QDoubleSpinBox(); self.sp_pause.setRange(0.2, 3.0); self.sp_pause.setSingleStep(0.1)
        self.sp_pause.setSuffix("s 停顿"); self.sp_pause.setFixedWidth(110)
        self.sp_max = QDoubleSpinBox(); self.sp_max.setRange(3.0, 30.0); self.sp_max.setSingleStep(1.0)
        self.sp_max.setSuffix("s 截断"); self.sp_max.setFixedWidth(120)
        og.addWidget(QLabel("语种")); og.addWidget(self.cmb_lang)
        og.addWidget(sl); og.addWidget(self.sl_sens); og.addWidget(self.lbl_sens)
        og.addWidget(self.sp_pause); og.addWidget(self.sp_max); og.addStretch(1)
        root.addWidget(opts)

        self.txt = QPlainTextEdit()
        self.txt.setPlaceholderText("识别结果会实时累积到这里…")
        self.txt.setStyleSheet("QPlainTextEdit{background:#16161e;color:#c0caf5;font-size:14px;border:1px solid #2a2b3c;}")
        root.addWidget(self.txt, 1)

        self.btn_start.clicked.connect(self._start)
        self.btn_stop.clicked.connect(self._stop)
        self.btnSave.clicked.connect(self._save)
        self.btn_clear.clicked.connect(self._clear)
        self.sl_sens.valueChanged.connect(self._on_sens)
        self.cmb_lang.currentTextChanged.connect(self._commit)
        self.sp_pause.valueChanged.connect(self._commit)
        self.sp_max.valueChanged.connect(self._commit)

    def _sync_options(self):
        self.cmb_lang.setCurrentText(self.settings.language)
        self.sl_sens.setValue(self.settings.vad_sensitivity)
        self.lbl_sens.setText(str(self.settings.vad_sensitivity))
        self.sp_pause.setValue(self.settings.pause_seconds)
        self.sp_max.setValue(self.settings.max_utterance_seconds)

    def _on_sens(self, v):
        self.lbl_sens.setText(str(v)); self._commit()

    def _commit(self, *a):
        self.settings.language = self.cmb_lang.currentText()
        self.settings.vad_sensitivity = self.sl_sens.value()
        self.settings.pause_seconds = float(self.sp_pause.value())
        self.settings.max_utterance_seconds = float(self.sp_max.value())

    def _start(self):
        self._commit()
        from ..config import save_settings
        save_settings(self.settings)
        self.txt.clear()
        self.btn_start.setEnabled(False); self.btn_stop.setEnabled(True)
        self.worker = RealtimeWorker(self.model_manager, self.settings)
        self.worker.log.connect(self.log_panel.append)
        self.worker.level.connect(self._on_level)
        self.worker.state.connect(self._on_state)
        self.worker.transcript_append.connect(self._append_text)
        self.worker.start()

    def _stop(self):
        if self.worker:
            self.worker.stop()
        self.btn_stop.setEnabled(False)
        QTimer.singleShot(400, lambda: self.btn_start.setEnabled(True))

    def _on_level(self, v: float):
        self.meter.setValue(int(min(100, v * 100)))

    def _on_state(self, s: str):
        m = {"idle": "待机", "listening": "聆听中", "speaking": "说话中", "decoding": "识别中…"}
        self.lbl_state.setText(m.get(s, s))

    def _append_text(self, t: str):
        # each recognized utterance lands on its own line
        cur = self.txt.textCursor()
        cur.movePosition(cur.MoveOperation.End)
        cur.insertText((t or "") + "\n")
        self.txt.setTextCursor(cur)

    def _save(self):
        if not self.txt.toPlainText():
            return
        d, _ = QFileDialog.getSaveFileName(self, "保存听写文本", str(Path.home() / "Downloads" / "realtime.txt"), "文本 (*.txt)")
        if d:
            Path(d).write_text(self.txt.toPlainText(), encoding="utf-8")
            self.log_panel.append(f"已保存: {d}", "ok")

    def _clear(self):
        self.txt.clear()

    def on_stop_external(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
