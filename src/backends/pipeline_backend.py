"""Pipeline backend: Whisper transcribe → Qwen-ForcedAligner word align →
SenseVoice emotion/event tag → merged SRT-ready output.

This is the highest-quality path: Whisper large-v3 for accuracy, Qwen's
forced aligner for precision timestamps, and SenseVoice for rich metadata
(laughing, crying, music, applause …) all stitched into one timeline.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

import numpy as np

from .base import ASRResult, TimestampItem, TimestampResult
from .whisper_backend import WhisperBackend
from .sensevoice_backend import SenseVoiceBackend

# ── SenseVoice emotion / event labels ────────────────────────────
_EMOTION_RE = re.compile(r"<\|(HAPPY|SAD|ANGRY|NEUTRAL|FEARFUL|DISGUSTED|SURPRISED)\|>")
_EVENT_RE   = re.compile(r"<\|(Speech|Music|Laughter|Applause|Crying|BGM)\|>")


def _parse_sensevoice_tags(text: str) -> Tuple[str, str, str]:
    """Return (cleaned_text, emotion, event_type)."""
    emotion = ""
    m = _EMOTION_RE.search(text)
    if m:
        emotion = m.group(1)
    event_type = "Speech"
    m = _EVENT_RE.search(text)
    if m:
        event_type = m.group(1)
    # Strip all angle-bracket tags
    cleaned = re.sub(r"<\|[^|]+\|>", "", text)
    return cleaned.strip(), emotion, event_type


class PipelineBackend:
    """Orchestrates Whisper, Qwen-ForcedAligner, and SenseVoice.

    Memory usage is higher (~6 GB for all three), but output quality is
    the best the tool can deliver.
    """

    def __init__(self, models_root: str = "models"):
        self._whisper = WhisperBackend(models_root)
        self._sensevoice = SenseVoiceBackend(models_root)
        self._aligner = None       # Qwen3-ForcedAligner
        self._aligner_processor = None
        self._device = ""
        self._loaded = False

    # ── protocol ──────────────────────────────────────────────────
    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device_name(self) -> str:
        return self._device

    @property
    def model_id(self) -> str:
        return "pipeline (Whisper+QwenAligner+SenseVoice)"

    @property
    def aligner_loaded(self) -> bool:
        return self._aligner is not None

    # ── load / unload ─────────────────────────────────────────────
    def load(
        self,
        whisper_size: str = "large-v3",
        sensevoice_variant: str = "small",
        aligner_path: str = "",
        device: str = "auto",
        dtype=None,
        **__,
    ) -> None:
        import torch as _torch

        self.unload()
        self._device = device

        # 1. Whisper (main ASR)
        self._whisper.load(model_size=whisper_size, device=device)

        # 2. Qwen-ForcedAligner (word-level timestamps)
        #    Load standalone (not via Qwen3ASRModel) to save VRAM.
        if aligner_path:
            from qwen_asr import Qwen3ForcedAligner
            d = device if device != "auto" else "mps"
            dt = dtype if dtype is not None else (_torch.float16 if d == "mps" else _torch.bfloat16)
            self._aligner = Qwen3ForcedAligner.from_pretrained(
                aligner_path,
                device_map=d,
                dtype=dt,
            )

        # 3. SenseVoice (emotion / event tags)
        self._sensevoice.load(variant=sensevoice_variant, device="cpu")

        self._loaded = True

    def unload(self) -> None:
        self._whisper.unload()
        if self._aligner is not None:
            del self._aligner
            self._aligner = None
        self._sensevoice.unload()
        self._loaded = False
        import torch
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

    # ── transcribe (the orchestration) ────────────────────────────
    def transcribe(
        self,
        wav: np.ndarray,
        language: Optional[str],
        return_time_stamps: bool,
    ) -> ASRResult:
        # --- Step 1: Whisper → text + language ---
        whisper_result = self._whisper.transcribe(wav, language, False)
        text = whisper_result.text.strip()
        detected_lang = whisper_result.language or (language or "")

        if not text:
            return ASRResult(language=detected_lang, text="")

        # --- Step 2: SenseVoice → emotion / event tags ---
        emotion_tags: List[Tuple[float, float, str, str]] = []  # (st, et, emotion, event)
        if self._sensevoice.is_loaded:
            try:
                sv_result = self._sensevoice.transcribe(wav, language, True)
                # SenseVoice returns sentence-level text with inline tags.
                # We parse each sentence to extract emotion/event.
                if sv_result.time_stamps and sv_result.time_stamps.items:
                    for it in sv_result.time_stamps.items:
                        cleaned, emo, evt = _parse_sensevoice_tags(getattr(it, "text", ""))
                        if emo or evt:
                            # Offset: SenseVoice items have 0-based ms timestamps
                            # converted to seconds in the backend.
                            emotion_tags.append((
                                getattr(it, "start_time", 0.0),
                                getattr(it, "end_time", 0.0),
                                emo,
                                evt,
                            ))
            except Exception:
                pass  # SenseVoice is optional; don't fail the whole pipeline

        # --- Step 3: Qwen-ForcedAligner → word-level timestamps ---
        ts_result: Optional[TimestampResult] = None
        if return_time_stamps and self._aligner is not None and text:
            try:
                align_results = self._aligner.align(
                    audio=(wav, 16000),
                    text=text,
                    language=detected_lang or "auto",
                )
                if align_results and len(align_results) > 0:
                    items: List[TimestampItem] = []
                    for ar in align_results[0].items if align_results[0] else []:
                        word = str(getattr(ar, "text", ""))
                        st = float(getattr(ar, "start_time", 0.0))
                        et = float(getattr(ar, "end_time", 0.0))
                        # Check for overlapping emotion/event tag
                        tag = ""
                        for em_st, em_et, emo, evt in emotion_tags:
                            if st >= em_st - 0.1 and et <= em_et + 0.1:
                                parts = []
                                if emo and emo != "NEUTRAL":
                                    parts.append(f"[{emo}]")
                                if evt and evt != "Speech":
                                    parts.append(f"[{evt}]")
                                tag = " ".join(parts)
                                break
                        if tag:
                            word = f"{word} {tag}"
                        items.append(TimestampItem(text=word, start_time=st, end_time=et))
                    ts_result = TimestampResult(items=items)
            except Exception:
                # Fall back to Whisper word timestamps
                wr = self._whisper.transcribe(wav, language, return_time_stamps=True)
                ts_result = wr.time_stamps

        return ASRResult(
            language=detected_lang,
            text=text,
            time_stamps=ts_result,
        )
