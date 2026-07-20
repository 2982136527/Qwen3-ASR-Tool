"""Load and manage ASR models via pluggable backends.

Switches between Qwen3-ASR, Whisper large-v3 (faster-whisper), and SenseVoice
at runtime.  Each backend conforms to the ASRBackend protocol so the rest of
the application never needs to know which engine is loaded.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import numpy as np

from .backends import ASRResult, QwenBackend, WhisperBackend, SenseVoiceBackend
from .config import Settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _import_torch():
    import torch
    return torch


def available_devices() -> list:
    torch = _import_torch()
    devs = ["cpu"]
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        devs.append("mps")
    if getattr(torch.cuda, "is_available", lambda: False)():
        devs.append("cuda")
        for i in range(torch.cuda.device_count()):
            devs.append(f"cuda:{i}")
    return devs


def resolve_device(settings_device: str) -> str:
    torch = _import_torch()
    if settings_device == "auto":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        if getattr(torch.cuda, "is_available", lambda: False)():
            return "cuda:0"
        return "cpu"
    return settings_device


def resolve_dtype(settings_precision: str, device: str):
    torch = _import_torch()
    if settings_precision == "auto":
        if device.startswith("cuda"):
            return torch.bfloat16
        if device == "mps":
            return torch.float16
        return torch.float32
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[settings_precision]


class ModelManager:
    """Holds one ASR backend at a time, routes transcribe calls through it."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.Lock()
        self._backend = None

    @property
    def is_loaded(self) -> bool:
        return self._backend is not None and self._backend.is_loaded

    @property
    def aligner_loaded(self) -> bool:
        if self._backend is not None:
            return getattr(self._backend, "aligner_loaded", False)
        return False

    @property
    def device(self) -> str:
        return self._backend.device_name if self._backend else ""

    @property
    def loaded_repo(self) -> str:
        return self._backend.model_id if self._backend else ""

    def models_root(self) -> Path:
        p = Path(self.settings.models_dir)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        p.mkdir(parents=True, exist_ok=True)
        return p

    def model_dir(self, name: str = "") -> Path:
        return self.models_root() / name

    def model_status(self, name: str = "") -> str:
        d = self.model_dir(name)
        if (d / ".downloaded").exists() and _looks_complete(d):
            return "ready"
        if any(d.glob("*.safetensors")) or any(d.glob("*.json")):
            return "partial"
        return "missing"

    def ensure_downloaded(self, repo_id: str, repo_name: str, on_log=None) -> Path:
        target = self.model_dir(repo_name)
        target.mkdir(parents=True, exist_ok=True)
        if (target / ".downloaded").exists() and _looks_complete(target):
            return target
        from scripts.fetch import download_one
        source = self.settings.download_source
        if on_log:
            on_log("info", f"downloading {repo_id} via {source} -> {target}")
        download_one(repo_id, target, source)
        return target

    def load(self, on_log=None, on_stage=None) -> None:
        mi = self.settings.model_info()
        backend_type = mi["backend"]

        with self._lock:
            self._unload_locked()

            if backend_type == "qwen":
                self._backend = QwenBackend(str(self.models_root()))
                self._load_qwen(mi, on_log, on_stage)
            elif backend_type == "whisper":
                self._backend = WhisperBackend(str(self.models_root()))
                self._load_whisper(mi, on_log)
            elif backend_type == "sensevoice":
                self._backend = SenseVoiceBackend(str(self.models_root()))
                self._load_sensevoice(mi, on_log)
            else:
                raise ValueError(f"Unknown backend: {backend_type}")

    def _load_qwen(self, mi, on_log, on_stage):
        device = resolve_device(self.settings.device)
        dtype = resolve_dtype(self.settings.precision, device)

        repo_id = f"Qwen/{mi['repo_name']}"
        if on_stage:
            on_stage(f"checking {mi['repo_name']}")
        self.ensure_downloaded(repo_id, mi["repo_name"], on_log)
        asr_path = self.model_dir(mi["repo_name"])

        aligner_path = None
        if self.settings.enable_aligner and mi.get("aligner_repo"):
            if on_stage:
                on_stage(f"checking {mi['aligner']}")
            self.ensure_downloaded(mi["aligner_repo"], mi["aligner"], on_log)
            aligner_path = self.model_dir(mi["aligner"])

        if on_log:
            on_log("info", f"loading {mi['label']} on {device} ({dtype})")
        if on_stage:
            on_stage("loading model")
        self._backend.load(
            asr_path=str(asr_path),
            aligner_path=str(aligner_path) if aligner_path else None,
            device=device,
            dtype=dtype,
            max_inference_batch_size=self.settings.max_inference_batch_size,
            max_new_tokens=self.settings.max_new_tokens,
        )
        if on_log:
            on_log("info", f"ready: {mi['label']} on {device}")

    def _load_whisper(self, mi, on_log):
        device = resolve_device(self.settings.device)
        if on_log:
            on_log("info", f"loading {mi['label']} on {device} ...")
        self._backend.load(model_size=mi["repo_name"], device=device)
        if on_log:
            on_log("info", f"ready: {mi['label']}")

    def _load_sensevoice(self, mi, on_log):
        if on_log:
            on_log("info", f"loading {mi['label']} ...")
        self._backend.load(variant=mi["repo_name"], device="cpu")
        if on_log:
            on_log("info", f"ready: {mi['label']}")

    def _unload_locked(self):
        if self._backend is not None:
            self._backend.unload()
            self._backend = None

    def unload(self):
        with self._lock:
            self._unload_locked()

    def transcribe_one(self, wav: np.ndarray, language: Optional[str],
                       return_time_stamps: bool) -> ASRResult:
        if self._backend is None:
            raise RuntimeError("no backend loaded")
        lang = None if (language is None or str(language).lower() == "auto") else language
        with self._lock:
            return self._backend.transcribe(wav, lang, return_time_stamps)


def _looks_complete(model_dir: Path) -> bool:
    if not model_dir.is_dir():
        return False
    if not (model_dir / "config.json").exists():
        return False
    has_weights = any(model_dir.glob("*.safetensors")) or any(model_dir.glob("*.bin"))
    has_tok = (model_dir / "tokenizer.json").exists() or (model_dir / "tokenizer_config.json").exists()
    return has_weights and has_tok
