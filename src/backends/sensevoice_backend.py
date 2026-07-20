"""SenseVoice backend via FunASR. Lightweight (~100 MB), supports 50+ languages,
emotion detection, and built-in VAD with sentence-level timestamps.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .base import ASRResult, TimestampItem, TimestampResult

SENSE_MODELS = {
    "small": "iic/SenseVoiceSmall",
    "large": "iic/SenseVoiceLarge",
}


class SenseVoiceBackend:
    def __init__(self, models_root: str = "models"):
        self._model = None
        self._device = ""
        self._variant = ""
        self._models_root = models_root

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device_name(self) -> str:
        return self._device

    @property
    def model_id(self) -> str:
        return self._variant

    @property
    def aligner_loaded(self) -> bool:
        return True  # VAD + sentence-level timestamps are built in

    def load(
        self,
        variant: str = "small",
        device: str = "cpu",
        **__,
    ) -> None:
        from funasr import AutoModel

        self.unload()
        model_name = SENSE_MODELS.get(variant, SENSE_MODELS["small"])
        self._model = AutoModel(
            model=model_name,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device="cpu",  # FunASR MPS support is limited; CPU is reliable + fast for small model
            disable_update=True,
        )
        self._device = device
        self._variant = variant

    def transcribe(
        self,
        wav: np.ndarray,
        language: Optional[str],
        return_time_stamps: bool,
    ) -> ASRResult:
        if self._model is None:
            raise RuntimeError("SenseVoice backend not loaded")
        lang = language if (language and language != "auto") else "auto"
        # FunASR expects a dict with audio and optional language hint
        result = self._model.generate(
            input=wav,
            language=lang,
            use_itn=True,
            batch_size_s=60,
        )
        # result is a list of dicts: [{"text": "...", "timestamp": [[st, et], ...], ...}]
        texts = []
        all_items = []
        for r in (result or []):
            t = (r.get("text") or "").strip()
            if t:
                texts.append(t)
            timestamps = r.get("timestamp") or []
            # SenseVoice timestamps are sentence-level: [[start_ms, end_ms], ...]
            # Each pair aligns with sub-sentence tokens
            for ts in timestamps:
                if isinstance(ts, (list, tuple)) and len(ts) >= 2:
                    all_items.append(
                        TimestampItem(
                            text="",   # sentence-level, text already in `texts`
                            start_time=round(float(ts[0]) / 1000.0, 3),
                            end_time=round(float(ts[1]) / 1000.0, 3),
                        )
                    )
        full_text = " ".join(texts)
        detected_lang = ""
        if result and len(result) > 0:
            detected_lang = result[0].get("lang", "")

        ts_result = None
        if return_time_stamps and all_items:
            ts_result = TimestampResult(items=all_items)
        return ASRResult(
            language=detected_lang or (language if language != "auto" else ""),
            text=full_text,
            time_stamps=ts_result,
        )

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
