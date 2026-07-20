"""Whisper large-v3 backend powered by faster-whisper (CTranslate2).

Returns word-level timestamps natively — no external aligner needed.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .base import ASRResult, TimestampItem, TimestampResult


class WhisperBackend:
    def __init__(self, models_root: str = "models"):
        self._model = None
        self._device = ""
        self._model_size = ""
        self._models_root = models_root

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device_name(self) -> str:
        return self._device

    @property
    def model_id(self) -> str:
        return self._model_size

    @property
    def aligner_loaded(self) -> bool:
        # Whisper returns word timestamps natively
        return True

    def load(
        self,
        model_size: str = "large-v3",
        device: str = "auto",
        compute_type: str = "auto",
        **__,
    ) -> None:
        from faster_whisper import WhisperModel

        self.unload()
        if compute_type == "auto":
            if device.startswith("cuda"):
                compute_type = "float16"
            elif device == "mps":
                compute_type = "int8_float16"
            else:
                compute_type = "int8"
        # faster-whisper auto-downloads model to HF cache on first use
        self._model = WhisperModel(
            model_size,
            device="auto" if device in ("auto", "mps") else device,
            compute_type=compute_type,
            download_root=self._models_root or None,
            local_files_only=False,
        )
        self._device = device
        self._model_size = model_size

    def transcribe(
        self,
        wav: np.ndarray,
        language: Optional[str],
        return_time_stamps: bool,
    ) -> ASRResult:
        if self._model is None:
            raise RuntimeError("Whisper backend not loaded")
        lang = language.lower() if language and language != "auto" else None
        segments, info = self._model.transcribe(
            wav.astype(np.float32),
            language=lang,
            word_timestamps=return_time_stamps,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                threshold=0.35,
            ),
        )
        full_text = " ".join(s.text.strip() for s in segments)
        ts = None
        if return_time_stamps:
            items = []
            for seg in segments:
                words = getattr(seg, "words", None) or []
                for w in words:
                    items.append(TimestampItem(
                        text=str(w.word),
                        start_time=round(w.start, 3),
                        end_time=round(w.end, 3),
                    ))
            ts = TimestampResult(items=items) if items else None
        return ASRResult(
            language=info.language or "",
            text=full_text,
            time_stamps=ts,
        )

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            import torch
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
