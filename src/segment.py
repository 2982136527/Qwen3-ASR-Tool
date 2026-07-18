"""Fixed-length segmentation for batch transcription.

We cut the decoded 16k mono waveform into ~target-second windows slightly
snapped to low-energy frames so we don't slice mid-word. Returns (chunk, offset)
pairs just like qwen_asr's internal splitter but with a configurable target that
is small enough (default 18s) to keep each transcription result snippet-sized and
visible in the live log.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

SAMPLE_RATE = 16000


def _energy(wav: np.ndarray, win: int) -> np.ndarray:
    a = np.abs(wav).astype(np.float32)
    if win < 2:
        return a
    kernel = np.ones(win, dtype=np.float32) / win
    # cheap moving average of |x|
    return np.convolve(a, kernel, mode="same")


def segment_audio(wav: np.ndarray, sr: int, target_sec: float = 18.0,
                  search_sec: float = 4.0, min_segment_sec: float = 4.0
                  ) -> List[Tuple[np.ndarray, float]]:
    wav = np.asarray(wav, dtype=np.float32)
    total = wav.shape[0]
    if total == 0:
        return []
    target = int(target_sec * sr)
    search = int(search_sec * sr)
    win = max(160, int(0.03 * sr))

    if total <= target + search:
        return [(wav, 0.0)]

    out: List[Tuple[np.ndarray, float]] = []
    start = 0
    offset = 0.0
    while start < total:
        ideal = start + target
        if ideal >= total - min_segment_sec * sr:
            out.append((wav[start:total], offset))
            break
        lo = max(start, ideal - search)
        hi = min(total, ideal + search)
        if hi - lo < win:
            cut = ideal
        else:
            e = _energy(wav[lo:hi], win)
            # pick the lowest-energy center inside the window
            local = int(np.argmin(e))
            cut = lo + local
        cut = int(max(cut, start + int(min_segment_sec * sr)))
        cut = int(min(cut, total))
        out.append((wav[start:cut], offset))
        offset += (cut - start) / float(sr)
        start = cut
    return out
