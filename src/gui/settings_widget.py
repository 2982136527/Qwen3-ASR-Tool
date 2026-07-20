"""Settings tab: model/device/precision + aligner + download + token params.

Shows the current model and disk status of every snapshot under ./models, lets
the user pick a precision and device, trigger an immediate download, and load
the model so the first transcription doesn't pay the startup tax mid-job.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal as Signal
from PyQt6.QtWidgets import (QComboBox, QFormLayout, QGroupBox, QHBoxLayout,
                             QLabel, QMessageBox, QProgressBar, QPushButton,
                             QSpinBox, QVBoxLayout, QWidget)

from ..config import (DOWNLOAD_SOURCES, Settings, save_settings, MODEL_OPTIONS, model_by_key)
from ..model_manager import ModelManager, available_devices, resolve_device


class DownloadWorker(QThread):
    log = Signal(str, str)
    done = Signal()

    def __init__(self, manager: ModelManager, settings: Settings, repo_id: str,
                 repo_name: str):
        super().__init__()
        self.manager = manager
        self.settings = settings
        self.repo_id = repo_id
        self.repo_name = repo_name

    def run(self):
        try:
            self.manager.ensure_downloaded(self.repo_id, self.repo_name,
                                           on_log=lambda lv, m: self.log.emit(m, lv))
        except Exception as e:
            self.log.emit(f"下载失败: {e}", "error")
        self.done.emit()


class LoadWorker(QThread):
    log = Signal(str, str)
    done = Signal(bool)

    def __init__(self, manager: ModelManager, settings: Settings):
        super().__init__()
        self.manager = manager
        self.settings = settings

    def run(self):
        try:
            self.manager.load(on_log=lambda lv, m: self.log.emit(m, lv),
                              on_stage=lambda s: self.log.emit(s, "stage"))
            self.log.emit(f"模型加载完成: {getattr(self.manager,'loaded_repo','')}", "ok")
            self.done.emit(True)
        except Exception as e:
            self.log.emit(f"加载失败: {e}", "error")
            self.done.emit(False)


class SettingsWidget(QWidget):
    def __init__(self, model_manager: ModelManager, settings: Settings, log_panel,
                 status_refresher=None, parent=None):
        super().__init__(parent)
        self.model_manager = model_manager
        self.settings = settings
        self.log_panel = log_panel
        self.status_refresher = status_refresher or (lambda: None)
        self._dl_worker: Optional[DownloadWorker] = None
        self._load_worker: Optional[LoadWorker] = None
        self._build()
        self.refresh_status()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12); root.setSpacing(10)

        box = QGroupBox("模型与硬件")
        f = QFormLayout(box)
        f.setHorizontalSpacing(10); f.setVerticalSpacing(8)

        self.cmb_asr = QComboBox()
        for m in MODEL_OPTIONS:
            self.cmb_asr.addItem(m["label"], m["key"])
        idx = max(0, self.cmb_asr.findData(self.settings.asr_model))
        self.cmb_asr.setCurrentIndex(idx)
        self.cmb_asr.currentIndexChanged.connect(self._on_model_changed)

        self.cmb_device = QComboBox()
        self.cmb_device.addItem("自动", "auto")
        for d in available_devices():
            label = {"mps": "Apple GPU (MPS)", "cpu": "CPU"}.get(d, d)
            self.cmb_device.addItem(label, d)
        if self.settings.device in [self.cmb_device.itemData(i) for i in range(self.cmb_device.count())]:
            self.cmb_device.setCurrentIndex(
                self.cmb_device.findData(self.settings.device))

        self.cmb_prec = QComboBox()
        for p in ["auto", "float16", "bfloat16", "float32"]:
            self.cmb_prec.addItem(p, p)
        if self.settings.precision in [self.cmb_prec.itemData(i) for i in range(self.cmb_prec.count())]:
            self.cmb_prec.setCurrentIndex(self.cmb_prec.findData(self.settings.precision))

        self.sp_tokens = QSpinBox(); self.sp_tokens.setRange(64, 8192); self.sp_tokens.setSingleStep(128)
        self.sp_tokens.setValue(self.settings.max_new_tokens)
        self.sp_batch = QSpinBox(); self.sp_batch.setRange(1, 32); self.sp_batch.setValue(self.settings.max_inference_batch_size)

        f.addRow("ASR 模型", self.cmb_asr)
        f.addRow("设备", self.cmb_device)
        f.addRow("精度", self.cmb_prec)
        f.addRow("最大生成 token", self.sp_tokens)
        f.addRow("推理 batch", self.sp_batch)

        dl_row = QHBoxLayout()
        self.cmb_source = QComboBox()
        for s in DOWNLOAD_SOURCES:
            self.cmb_source.addItem(s, s)
        self.cmb_source.setCurrentIndex(self.cmb_source.findData(self.settings.download_source))
        self.btn_download = QPushButton("立即下载所选 ASR 模型")
        self.btn_download_aligner = QPushButton("下载对齐器 (Qwen3-ForcedAligner-0.6B)")
        self.btn_download_all = QPushButton("下载全部 (ASR+对齐器)")
        dl_row.addWidget(self.cmb_source)
        dl_row.addWidget(self.btn_download)
        dl_row.addWidget(self.btn_download_aligner)
        dl_row.addWidget(self.btn_download_all)
        f.addRow("模型下载源", dl_row)

        self.btn_load = QPushButton("加载 / 重新加载模型")
        self.btn_load.setObjectName("primary")
        f.addRow("", self.btn_load)

        root.addWidget(box)

        st = QGroupBox("模型状态 (./models)")
        s2 = QVBoxLayout(st)
        self.lbl_status = QLabel("…")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setTextFormat(Qt.TextFormat.RichText)
        s2.addWidget(self.lbl_status)
        self.progress = QProgressBar(); self.progress.setMaximumHeight(12)
        s2.addWidget(self.progress)
        root.addWidget(st)

        info = QLabel(
            "<b>Qwen3-ASR 智能语音识别工具</b><br>"
            "• 1.7B 高精度版 / 0.6B 轻量版,本地离线,不上传云端<br>"
            "• 自动下载到 ./models 目录,首次运行较慢<br>"
            "• 时间戳对齐由 Qwen3-ForcedAligner-0.6B 提供,关闭可在 CPU/低端设备提速"
        )
        info.setWordWrap(True)
        root.addWidget(info)
        root.addStretch(1)

        # signals
        self.cmb_asr.currentIndexChanged.connect(self._commit)
        self.cmb_device.currentIndexChanged.connect(self._commit)
        self.cmb_prec.currentIndexChanged.connect(self._commit)
        self.sp_tokens.valueChanged.connect(self._commit)
        self.sp_batch.valueChanged.connect(self._commit)
        self.cmb_source.currentIndexChanged.connect(self._commit)
        self.btn_download.clicked.connect(lambda: self._download(asr_only=True, aligner=False))
        self.btn_download_aligner.clicked.connect(lambda: self._download(asr_only=False, aligner=True))
        self.btn_download_all.clicked.connect(lambda: self._download(asr_only=True, aligner=True))
        self.btn_load.clicked.connect(self._load)


    def _on_model_changed(self, idx):
        mi = MODEL_OPTIONS[idx]
        is_qwen_or_pipe = mi["backend"] in ("qwen", "pipeline")
        is_builtin = mi["backend"] in ("whisper", "sensevoice")
        self.btn_download.setVisible(is_qwen_or_pipe)
        self.btn_download_aligner.setVisible(is_qwen_or_pipe and mi.get("aligner_repo"))
        self.btn_download_all.setVisible(is_qwen_or_pipe)
        self.cmb_prec.setVisible(is_qwen_or_pipe)
        self.cmb_device.setVisible(is_qwen_or_pipe)
        if is_builtin:
            self.cmb_align.setChecked(True)
            self.cmb_align.setEnabled(False)
        elif mi["backend"] == "pipeline":
            self.cmb_align.setChecked(True)
            self.cmb_align.setEnabled(False)
        else:
            self.cmb_align.setEnabled(True)
            self.cmb_align.setChecked(self.settings.enable_aligner)
        self._commit()
    def _commit(self):
        self.settings.asr_model = self.cmb_asr.currentData()  # model key
        self.settings.device = self.cmb_device.currentData()
        self.settings.precision = self.cmb_prec.currentData()
        self.settings.max_new_tokens = int(self.sp_tokens.value())
        self.settings.max_inference_batch_size = int(self.sp_batch.value())
        self.settings.download_source = self.cmb_source.currentData()
        save_settings(self.settings)
        self.refresh_status()

    def refresh_status(self):
        try:
            asr_name = self.settings.model_info()["repo_name"]
            al = self.settings.model_info().get("aligner", "Qwen3-ForcedAligner-0.6B")
            asr_st = self.model_manager.model_status(asr_name) if asr_name else "missing"
            al_st = self.model_manager.model_status(al)
            d = self.model_manager.model_dir(asr_name)
            ad = self.model_manager.model_dir(al)
            size = self._dir_size(d) + self._dir_size(ad)
            color = {"ready": "#9ece6a", "partial": "#e0af68", "missing": "#f7768e"}.get
            html = (
                f"ASR <b>{asr_name}</b>: <span style='color:{color(asr_st, "#a9b1d6")}'>{asr_st}</span>"
                f"  ({asr_st})  &nbsp;  "
                f"对齐器 <b>{al}</b>: <span style='color:{color(al_st, "#a9b1d6")}'>{al_st}</span><br>"
                f"本地占用: {size/1e9:.2f} GB  &nbsp; 目录: ./models"
            )
            self.lbl_status.setText(html)
            if self.model_manager.is_loaded:
                self.lbl_status.setText(self.lbl_status.text() +
                    f"<br><span style='color:#9ece6a'>已加载: {self.model_manager.loaded_repo} @ {self.model_manager.device}</span>")
        except Exception as e:
            self.lbl_status.setText(f"状态刷新错误: {e}")

    def _dir_size(self, p: Path) -> int:
        if not p.exists():
            return 0
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())

    def _download(self, asr_only: bool, aligner: bool):
        if self._dl_worker and self._dl_worker.isRunning():
            QMessageBox.information(self, "忙碌", "已有下载任务在运行")
            return
        self._commit()
        tasks = []
        if asr_only:
            name = self.settings.model_info()["repo_name"]
            tasks.append((f"Qwen/{name}", name))
        if aligner:
            alr = self.settings.model_info().get("aligner_repo"); tasks.append((alr, self.settings.model_info()["aligner"])) if alr else None
        if not tasks:
            return
        self.progress.setRange(0, 0)
        self._dl_worker = _MultiDownloadWorker(self.model_manager, self.settings, tasks)
        self._dl_worker.log.connect(self.log_panel.append)
        self._dl_worker.done.connect(self._on_dl_done)
        self._dl_worker.start()

    def _on_dl_done(self):
        self.progress.setRange(0, 1); self.progress.setValue(1)
        self.refresh_status(); self.status_refresher()

    def _load(self):
        if self._load_worker and self._load_worker.isRunning():
            QMessageBox.information(self, "忙碌", "模型正在加载")
            return
        self._commit()
        self.btn_load.setEnabled(False)
        self.progress.setRange(0, 0)
        self._load_worker = LoadWorker(self.model_manager, self.settings)
        self._load_worker.log.connect(self.log_panel.append)
        self._load_worker.done.connect(self._on_load_done)
        self._load_worker.start()

    def _on_load_done(self, ok: bool):
        self.progress.setRange(0, 1); self.progress.setValue(1)
        self.progress.setFormat("就绪" if ok else "失败")
        self.btn_load.setEnabled(True)
        self.refresh_status(); self.status_refresher()


class _MultiDownloadWorker(QThread):
    log = Signal(str, str)
    done = Signal()

    def __init__(self, manager, settings, tasks):
        super().__init__()
        self.manager = manager; self.settings = settings; self.tasks = tasks

    def run(self):
        for repo_id, repo_name in self.tasks:
            try:
                self.manager.ensure_downloaded(repo_id, repo_name,
                                                on_log=lambda lv, m: self.log.emit(m, lv))
            except Exception as e:
                self.log.emit(f"下载失败 {repo_id}: {e}", "error")
        self.done.emit()
