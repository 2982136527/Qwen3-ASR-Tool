"""Top-level window: tabbed UI, dark theme, shared log panel.

Three tabs (批量转录 / 实时听写 / 设置) over one shared live log. A status bar
shows the current model + device so the user always knows what they're running
against. On quit we wait briefly for any worker threads and unload the model so
MPS memory is released back to the OS.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QFont, QIcon
from PyQt6.QtWidgets import (QMainWindow, QMessageBox, QSplitter,
                              QLabel, QMainWindow, QMessageBox, QSplitter,
                              QStatusBar, QTabWidget, QVBoxLayout, QWidget)

from ..config import (Settings, load_settings, save_settings)
from ..model_manager import ModelManager
from .batch_widget import BatchWidget
from .log_panel import LogPanel
from .realtime_widget import RealtimeWidget
from .settings_widget import SettingsWidget


DARK_QSS = """
QWidget { background:#1a1b26; color:#a9b1d6; font-size:13px; }
QMainWindow, QWidget#central { background:#1a1b26; }
QTabWidget::pane { border:1px solid #2a2b3c; top:-1px; }
QTabBar::tab { background:#16161e; color:#9aa5ce; padding:8px 16px; border:1px solid #2a2b3c;
  border-bottom:none; border-top-left-radius:4px; border-top-right-radius:4px; }
QTabBar::tab:selected { background:#24283b; color:#c0caf5; }
QTabBar::tab:hover:!selected { background:#1f2335; }
QGroupBox { border:1px solid #2a2b3c; border-radius:6px; margin-top:12px; padding:10px 8px 8px 8px; }
QGroupBox::title { subcontrol-origin: margin; left:10px; padding:0 4px; color:#7dcfff; }
QPushButton { background:#24283b; color:#a9b1d6; border:1px solid #2a2b3c; border-radius:5px;
  padding:6px 14px; }
QPushButton:hover { background:#2a2e42; }
QPushButton:disabled { color:#565f89; background:#1c1d29; }
QPushButton#primary { background:#3d59a1; color:#ffffff; border-color:#3d59a1; }
QPushButton#primary:hover { background:#4a6cc4; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background:#16161e; color:#c0caf5;
  border:1px solid #2a2b3c; border-radius:4px; padding:4px 6px; }
QComboBox QAbstractItemView { background:#16161e; color:#c0caf5; selection-background-color:#24283b; }
QProgressBar { background:#16161e; border:1px solid #2a2b3c; border-radius:4px; text-align:center; }
QProgressBar::chunk { background:#3d59a1; }
QTableWidget { background:#16161e; gridline-color:#2a2b3c; border:1px solid #2a2b3c; }
QHeaderView::section { background:#24283b; color:#9aa5ce; padding:4px; border:0; }
QSlider::groove:horizontal { background:#16161e; height:4px; border-radius:2px; }
QSlider::handle:horizontal { background:#3d59a1; width:14px; margin:-6px 0; border-radius:7px; }
QLabel { background:transparent; }
QScrollBar:vertical { background:#16161e; width:10px; }
QScrollBar::handle:vertical { background:#2a2b3c; border-radius:5px; min-height:20px; }
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Qwen3-ASR 智能语音识别工具")
        self.resize(1200, 780)
        self.settings: Settings = load_settings()
        self.model_manager = ModelManager(self.settings)

        central = QWidget(); central.setObjectName("central")
        outer = QVBoxLayout(central); outer.setContentsMargins(8, 8, 8, 8); outer.setSpacing(8)
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setHandleWidth(6)

        self.tabs = QTabWidget()
        self.log_panel = LogPanel()
        self.log_panel.capture_logging()

        self.batch = BatchWidget(self.model_manager, self.settings, self.log_panel)
        self.realtime = RealtimeWidget(self.model_manager, self.settings, self.log_panel)
        self.settings_tab = SettingsWidget(self.model_manager, self.settings, self.log_panel,
                                            status_refresher=self.refresh_status)

        self.tabs.addTab(self.batch, "批量转录")
        self.tabs.addTab(self.realtime, "实时听写")
        self.tabs.addTab(self.settings_tab, "设置")

        splitter.addWidget(self.tabs)
        splitter.addWidget(self.log_panel)
        splitter.setStretchFactor(0, 3); splitter.setStretchFactor(1, 2)
        splitter.setSizes([460, 320])
        outer.addWidget(splitter, 1)
        self.setCentralWidget(central)
        self.setStyleSheet(DARK_QSS)

        sb = QStatusBar()
        self.lbl_status = QLabel("模型: 未加载")
        sb.addWidget(self.lbl_status, 1)
        sb.showMessage("就绪. 请在「设置」标签页下载并加载模型, 然后在「批量转录」里拖入文件开始. ")
        self.setStatusBar(sb)

        self.log_panel.append("欢迎使用 Qwen3-ASR 智能识别工具 (本地离线, 阿里通义千问开源模型)", "info")
        self.log_panel.append("检测到 Apple M4 + MPS, 已默认使用 GPU 加速与 float16 精度. 可在「设置」中调整. ", "info")
        self.log_panel.append("模型首次下载约 5GB, 自动保存到 ./models 目录, 之后离线使用. ", "info")

        QTimer.singleShot(0, self._deferred_init)

    def _deferred_init(self):
        # Background status refresh + attempt auto-detect already-loaded paths
        try:
            self.settings_tab.refresh_status()
            self.refresh_status()
            from ..model_manager import available_devices
            self.log_panel.append(f"可用设备: {', '.join(available_devices())}", "info")
        except Exception as e:
            self.log_panel.append(f"初始化状态失败: {e}", "warn")

    def refresh_status(self):
        s = "未加载"
        if self.model_manager.is_loaded:
            s = f"{self.model_manager.loaded_repo} @ {self.model_manager.device}"
        else:
            # report what's on disk so the user knows next load will be cheap
            mi = self.settings.model_info()
            nm = mi.get("label", "")
            al = mi.get("aligner", "")
            asr_st = "auto" if not mi.get("needs_download") else self.model_manager.model_status(mi["repo_name"])
            al_st = "auto" if not mi.get("aligner_repo") else self.model_manager.model_status(al)
            s = f"{nm}: {asr_st}"
            if al:
                s += f"  |  对齐: {al} ({al_st})"
        self.lbl_status.setText(f"模型状态: {s}")

    def closeEvent(self, e):
        # Stop realtime mic capture first (audio resources need quick release)
        try:
            self.realtime.on_stop_external()
        except Exception:
            pass
        # Stop batch worker if running
        try:
            if getattr(self.batch, "worker", None) and self.batch.worker.isRunning():
                self.batch.worker.cancel()
        except Exception:
            pass

        settings_tab_busy = (self.settings_tab._dl_worker and self.settings_tab._dl_worker.isRunning()) or \
                            (self.settings_tab._load_worker and self.settings_tab._load_worker.isRunning())
        if settings_tab_busy:
            r = QMessageBox.question(self, "确认退出", "模型正在下载/加载, 确定退出?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if r != QMessageBox.StandardButton.Yes:
                e.ignore(); return

        save_settings(self.settings)
        # give workers a moment to wind down, then drop the model
        QTimer.singleShot(0, self._final_unload)
        e.accept()

    def _final_unload(self):
        try:
            self.model_manager.unload()
        except Exception:
            pass
        self.log_panel.capture_logging  # keep ref pyflakes quiet

    def refresh_all(self):
        self.refresh_status()
        self.settings_tab.refresh_status()
