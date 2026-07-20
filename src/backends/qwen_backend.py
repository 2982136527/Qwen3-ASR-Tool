"""Qwen3-ASR backend (existing engine, re-wrapped)."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .base import ASRResult, TimestampItem, TimestampResult


class QwenBackend:
    def __init__(self, models_root: str = "models"):
        self._asr = None
        self._device = ""
        self._dtype = None
        self._repo_name = ""
        self._aligner_loaded = False
        self._models_root = models_root

    @property
    def is_loaded(self) -> bool:
        return self._asr is not None

    @property
    def device_name(self) -> str:
        return self._device

    @property
    def model_id(self) -> str:
        return self._repo_name

    @property
    def aligner_loaded(self) -> bool:
        return self._aligner_loaded

    def load(
        self,
        asr_path: str,
        aligner_path: Optional[str] = None,
        device: str = "mps",
        dtype: object = None,
        max_inference_batch_size: int = 8,
        max_new_tokens: int = 1024,
        **__,
    ) -> None:
        import torch as _torch
        from qwen_asr import Qwen3ASRModel

        self.unload()
        kwargs = dict(
            device_map=device,
            dtype=dtype if dtype is not None else _torch.float16,
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
        )
        if aligner_path is not None:
            kwargs["forced_aligner"] = str(aligner_path)
            kwargs["forced_aligner_kwargs"] = dict(device_map=device, dtype=dtype)
            self._aligner_loaded = True
        else:
            self._aligner_loaded = False

        self._asr = Qwen3ASRModel.from_pretrained(str(asr_path), **kwargs)
        self._device = device
        self._dtype = dtype
        self._repo_name = asr_path

    def transcribe(
        self,
        wav: np.ndarray,
        language: Optional[str],
        return_time_stamps: bool,
    ) -> ASRResult:
        if self._asr is None:
            raise RuntimeError("Qwen backend not loaded")
        lang = None if (language is None or str(language).lower() == "auto") else language
        res = self._asr.transcribe(
            audio=(wav, 16000),
            language=lang,
            return_time_stamps=return_time_stamps,
        )
        r = res[0]
        ts = None
        if return_time_stamps and getattr(r, "time_stamps", None) is not None:
            items = []
            for it in r.time_stamps.items if r.time_stamps else []:
                items.append(TimestampItem(
                    text=str(getattr(it, "text", "")),
                    start_time=float(getattr(it, "start_time", 0.0)),
                    end_time=float(getattr(it, "end_time", 0.0)),
                ))
            ts = TimestampResult(items=items)
        return ASRResult(
            language=r.language or "",
            text=r.text or "",
            time_stamps=ts,
        )

    def unload(self) -> None:
        if self._asr is not None:
            import torch
            del self._asr
            self._asr = None
            self._aligner_loaded = False
            try:
                if self._device and self._device.startswith("mps"):
                    torch.mps.empty_cache()
                elif self._device and self._device.startswith("cuda"):
                    torch.cuda.empty_cache()
            except Exception:
                pass
