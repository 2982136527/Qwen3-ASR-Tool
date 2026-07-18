"""Live log panel: every transcription result and backend event lands here.

A monospace QPlainTextEdit that color-codes lines by level, auto-scrolls while
the user is parked at the bottom, and keeps a rolling cap so a long movie job
doesn't balloon memory. Worker threads connect their ``log`` signal to
``append`` (queued, so cross-thread is safe).
"""
from __future__ import annotations

import logging
from datetime import datetime

from PyQt6.QtCore import pyqtSlot
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import QPlainTextEdit


LEVEL_COLORS = {
    "info": "#a9b1d6",
    "ok": "#9ece6a",
    "warn": "#e0af68",
    "error": "#f7768e",
    "transcript": "#c0caf5",
    "stage": "#7dcfff",
}


class _QtSignalLogHandler(logging.Handler):
    """Routes Python ``logging`` records to a Qt signal via ``append``."""

    def __init__(self, panel):
        super().__init__()
        self._panel = panel

    def emit(self, record: logging.LogRecord) -> None:  # type: ignore[override]
        try:
            msg = self.format(record)
            lv = "info"
            if record.levelno >= logging.ERROR:
                lv = "error"
            elif record.levelno >= logging.WARNING:
                lv = "warn"
            self._panel.append(msg, lv)
        except Exception:
            pass


class LogPanel(QPlainTextEdit):
    MAX_LINES = 4000

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(self.MAX_LINES)
        font = self.font()
        font.setFamily("Menlo")
        font.setStyleHint(font.StyleHint.Monospace)
        font.setPointSize(11)
        self.setFont(font)
        self.setStyleSheet(
            "QPlainTextEdit { background:#16161e; color:#a9b1d6;"
            " border:1px solid #2a2b3c; }"
        )

    @pyqtSlot(str, str)
    def append(self, message: str, level: str = "info") -> None:
        color = LEVEL_COLORS.get(level, LEVEL_COLORS["info"])
        ts = datetime.now().strftime("%H:%M:%S")
        safe = (message or "").replace("<", "&lt;").replace(">", "&gt;")
        # transcript lines keep a leading tag like <Chinese>; restore after escape
        line = f'<span style="color:#565f89">{ts}</span>  <span style="color:{color}">{safe}</span>'
        at_bottom = self.verticalScrollBar().value() >= self.verticalScrollBar().maximum() - 4
        cur = self.textCursor()
        cur.movePosition(QTextCursor.MoveOperation.End)
        cur.insertHtml(line + "<br>")
        if at_bottom:
            self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

    def capture_logging(self, logger_name: str = "qwen_asr") -> None:
        lg = logging.getLogger(logger_name)
        h = _QtSignalLogHandler(self)
        h.setFormatter(logging.Formatter("%(name)s | %(levelname)s | %(message)s"))
        lg.addHandler(h)
        lg.setLevel(logging.INFO)
        # transformers chatter is noisy; keep warnings only
        logging.getLogger("transformers").setLevel(logging.WARNING)
