"""Load and manage Qwen3-ASR + Qwen3-ForcedAligner models.

Design goals:
  * local-first: weights live under ``models/`` in the project dir, downloaded
    via ModelScope (recommended in Mainland China) with HF fallback, never a
    remote cloud API.
  * Applefriendly defaults: device auto -> mps on Apple Silicon, dtype auto ->
    float16 on mps (MPS has partial bfloat16). CUDA still prefers bfloat16.
  * single-thread ownership: weights are large and the ASR engine is not
    thread-safe, so ModelManager serializes load/unload and inference through
    one owner. The GUI calls it from worker threads.
"""
from __future__ import annotations

import shutil
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from .config import ALIGNER_MODEL, ASR_MODELS, Settings

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
        # include enumerated cuda:0/1 for transparency
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


def resolve_dtype(settings_precision: str, device: str) -> "object":
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
    """Owns the loaded ASR model (+ optional aligner). Thread-safe enough for
    GUI worker threads; not for high concurrency."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.Lock()
        self.asr = None
        self.aligner_loaded = False
        self.device = None
        self.dtype = None
        self.loaded_repo = None

    @property
    def is_loaded(self) -> bool:
        return self.asr is not None

    def models_root(self) -> Path:
        p = Path(self.settings.models_dir)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        p.mkdir(parents=True, exist_ok=True)
        return p

    def model_dir(self, repo_name: str) -> Path:
        return self.models_root() / repo_name

    def model_status(self, repo_name: str) -> str:
        d = self.model_dir(repo_name)
        if (d / ".downloaded").exists() and _looks_complete(d):
            return "ready"
        if any(d.glob("*.safetensors")) or any(d.glob("*.json")):
            return "partial"
        return "missing"

    def ensure_downloaded(self, repo_id: str, repo_name: str, on_log=None) -> Path:
        """Download a single model if missing. Reuses scripts/fetch logic."""
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
        """Ensure model + aligner are downloaded, then load them onto device.

        Safe to call again after settings change: it unloads first."""
        torch = _import_torch()
        from qwen_asr import Qwen3ASRModel

        with self._lock:
            if self.is_loaded:
                self._unload_locked()

            repo_name = self.settings.model_repo()
            repo_id = f"Qwen/{repo_name}"
            device = resolve_device(self.settings.device)
            dtype = resolve_dtype(self.settings.precision, device)

            if on_stage:
                on_stage(f"checking {repo_name}")
            self.ensure_downloaded(repo_id, repo_name, on_log)
            asr_path = self.model_dir(repo_name)
            if on_log:
                on_log("info", f"loading ASR {repo_name} on {device} ({dtype})")

            aligner_path = None
            if self.settings.enable_aligner:
                if on_stage:
                    on_stage(f"checking {ALIGNER_MODEL}")
                self.ensure_downloaded(f"Qwen/{ALIGNER_MODEL}", ALIGNER_MODEL, on_log)
                aligner_path = self.model_dir(ALIGNER_MODEL)

            if on_stage:
                on_stage("loading model")
            kwargs = dict(
                device_map=device,
                dtype=dtype,
                max_inference_batch_size=self.settings.max_inference_batch_size,
                max_new_tokens=self.settings.max_new_tokens,
            )
            if aligner_path is not None:
                kwargs["forced_aligner"] = str(aligner_path)
                kwargs["forced_aligner_kwargs"] = dict(device_map=device, dtype=dtype)

            self.asr = Qwen3ASRModel.from_pretrained(str(asr_path), **kwargs)
            self.device = device
            self.dtype = dtype
            self.aligner_loaded = aligner_path is not None
            self.loaded_repo = repo_name
            if on_log:
                on_log("info", f"ready: {repo_name} on {device}")

    def _unload_locked(self) -> None:
        if self.asr is None:
            return
        try:
            del self.asr
        finally:
            self.asr = None
            self.aligner_loaded = False
            torch = _import_torch()
            try:
                if self.device and self.device.startswith("mps"):
                    torch.mps.empty_cache()
                elif self.device and self.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def unload(self) -> None:
        with self._lock:
            self._unload_locked()

    def transcribe_one(self, wav: np.ndarray, language: Optional[str],
                       return_time_stamps: bool):
        """Transcribe a single mono 16k float32 segment. Returns the ASR result
        object (language/text/time_stamps). Inference happens under the lock
        because the HF model is not reentrant."""
        if self.asr is None:
            raise RuntimeError("model not loaded")
        lang = None if (language is None or str(language).lower() == "auto") else language
        with self._lock:
            res = self.asr.transcribe(
                audio=(wav, 16000),
                language=lang,
                return_time_stamps=return_time_stamps,
            )
        return res[0]


def _looks_complete(model_dir: Path) -> bool:
    """A model dir is usable if it has a config and at least one weights file
    plus the tokenizer files we need for AutoProcessor."""
    if not model_dir.is_dir():
        return False
    if not (model_dir / "config.json").exists():
        return False
    has_weights = any(model_dir.glob("*.safetensors")) or any(model_dir.glob("*.bin"))
    has_tok = (model_dir / "tokenizer.json").exists() or (model_dir / "tokenizer_config.json").exists()
    # snapshot may include only preprocessor; require weights+tokenizer to load.
    return has_weights and has_tok
