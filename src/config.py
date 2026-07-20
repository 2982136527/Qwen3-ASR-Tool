"""Persistent application settings for the Qwen3-ASR GUI tool.

Settings are written to ``config.json`` next to the project root. Defaults are
chosen for Apple Silicon (auto device selects ``mps``) but every option is
overridable from the Settings tab.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict

MODEL_OPTIONS = [
    {
        "key": "qwen-1.7b",
        "label": "Qwen3-ASR 1.7B (高精度)",
        "backend": "qwen",
        "repo_name": "Qwen3-ASR-1.7B",
        "aligner": "Qwen3-ForcedAligner-0.6B",
        "aligner_repo": "Qwen/Qwen3-ForcedAligner-0.6B",
        "needs_download": True,
    },
    {
        "key": "qwen-0.6b",
        "label": "Qwen3-ASR 0.6B (轻量版)",
        "backend": "qwen",
        "repo_name": "Qwen3-ASR-0.6B",
        "aligner": "Qwen3-ForcedAligner-0.6B",
        "aligner_repo": "Qwen/Qwen3-ForcedAligner-0.6B",
        "needs_download": True,
    },
    {
        "key": "whisper-large-v3",
        "label": "Whisper large-v3 (faster-whisper)",
        "backend": "whisper",
        "repo_name": "large-v3",
        "aligner": "内置词级时间戳",
        "aligner_repo": None,
        "needs_download": False,  # auto-download on first load
    },
    {
        "key": "sensevoice-small",
        "label": "SenseVoice-Small (多语言+情感)",
        "backend": "sensevoice",
        "repo_name": "small",
        "aligner": "内置 VAD 时间戳",
        "aligner_repo": None,
        "needs_download": False,
    },
]

def model_by_key(key: str) -> dict:
    for m in MODEL_OPTIONS:
        if m["key"] == key:
            return m
    return MODEL_OPTIONS[0]

DOWNLOAD_SOURCES = ["modelscope", "huggingface", "auto"]


@dataclass
class Settings:
    asr_model: str = "qwen-1.7b"      # key into MODEL_OPTIONS
    device: str = "auto"              # auto | mps | cuda | cpu
    precision: str = "auto"           # auto | float16 | bfloat16 | float32
    language: str = "auto"            # auto | Chinese | English | ...
    enable_aligner: bool = True
    max_new_tokens: int = 1024
    max_inference_batch_size: int = 8
    # SRT segmentation
    max_chars_per_line: int = 36
    max_segment_seconds: float = 10.0
    merge_blank_lines: bool = True
    # realtime dictation
    vad_sensitivity: int = 3          # 1..5
    pause_seconds: float = 0.7
    max_utterance_seconds: float = 14.0
    input_device_index: int = -1      # -1 = system default
    # download
    download_source: str = "modelscope"
    models_dir: str = "models"
    output_dir: str = "output"
    ffmpeg_path: str = ""
    # window
    theme: str = "dark"

    def model_info(self) -> dict:
        return model_by_key(self.asr_model)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Settings":
        s = cls()
        for k, v in (d or {}).items():
            if hasattr(s, k):
                if isinstance(getattr(s, k), int) and not isinstance(v, int):
                    try:
                        v = int(v)
                    except (TypeError, ValueError):
                        continue
                if isinstance(getattr(s, k), float) and not isinstance(v, float):
                    try:
                        v = float(v)
                    except (TypeError, ValueError):
                        continue
                if isinstance(getattr(s, k), bool) and not isinstance(v, bool):
                    try:
                        v = str(v).strip().lower() in ("1", "true", "yes", "on")
                    except Exception:
                        continue
                setattr(s, k, v)
        return s


def default_config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "config.json"


def load_settings() -> Settings:
    p = default_config_path()
    if p.exists():
        try:
            return Settings.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    return Settings()


def save_settings(s: Settings) -> None:
    p = default_config_path()
    p.write_text(json.dumps(s.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
