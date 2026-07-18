"""Application entrypoint: ``python -m src.app`` (or via start.command).

Pre-checks ffmpeg and torch/MPS availability, prints friendly hints, then
launches the PyQt6 main window. We import heavy GUI modules late so a missing
dependency surfaces with a helpful message before Qt ever starts.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false")
os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")


def main() -> int:
    try:
        import torch  # noqa: F401
    except Exception as e:
        print("ERROR: 未安装 torch. 请在项目根目录运行:\n  source venv/bin/activate\n  pip install -r requirements.txt", file=sys.stderr)
        return 2

    try:
        from src.audio import resolve_ffmpeg
        resolve_ffmpeg("")
    except Exception as e:
        print(f"WARNING: {e}\n音频/视频解码需要 ffmpeg. 安装: brew install ffmpeg", file=sys.stderr)

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("INFO: Apple MPS 可用 (M4 GPU 加速)")
    elif getattr(torch.cuda, "is_available", lambda: False)():
        print("INFO: CUDA 可用")
    else:
        print("INFO: 未检测到 GPU 加速, 将使用 CPU (较慢)")
    # qwen-asr's transformers-backend only prints noisy warnings otherwise
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")

    try:
        # local-first: prevent HF/MS from re-hitting the network for unseen config
        os.environ.setdefault("HF_HUB_OFFLINE", "0")
    except Exception:
        pass

    from PyQt6.QtWidgets import QApplication
    from src.gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Qwen3-ASR Tool")
    app.setOrganizationName("AIBL")
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
