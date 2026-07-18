#!/bin/bash
# Double-clickable launcher for macOS Finder.
# Bootstraps a local venv on first run (so "out of the box" works), installs
# deps as needed, then launches the Qwen3-ASR GUI.
set -euo pipefail
cd "$(dirname "$0")"
PY=/opt/homebrew/bin/python3.12
if [[ ! -x "$PY" ]]; then
  PY=$(command -v python3) ; [[ "$PY" == "" ]] && PY=python3
fi
if [[ ! -d venv ]]; then
  echo "[setup] creating venv with $PY ..."
  "$PY" -m venv venv
  source venv/bin/activate
  pip install --upgrade pip >/dev/null
  pip install -r requirements.txt
else
  source venv/bin/activate
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[warn] 未检测到 ffmpeg, 音视频解压将不可用. 安装: brew install ffmpeg"
fi
python -m src.app
read -p "按回车关闭…" _ 2>/dev/null || true
