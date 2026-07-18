"""Batch transcription worker.

For each file we decode audio to 16k mono PCM with ffmpeg, slice it into
fixed-length segments (energy boundaries), and transcribe each segment through
Qwen3-ASR. Each segment's text is emitted as soon as it's decoded so the GUI log
panel shows transcription rolling forward in real time ("每一条转录结果实时显示").
With the aligner on, word/char timestamps build a precisely timed SRT; otherwise
we fall back to segment-time-bounded subtitles.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np

from PyQt6.QtCore import QThread, pyqtSignal as Signal

from .audio import decode_audio, fetch_to_local
from .config import Settings
from .segment import segment_audio
from .srt import (group_units_to_subtitles, save_srt, subtitles_from_segments,
                  write_srt)


class _ShiftedItem:
    """Lightweight stand-in for ForcedAlignItem carrying already-shifted times."""
    __slots__ = ("text", "start_time", "end_time")

    def __init__(self, text: str, start_time: float, end_time: float):
        self.text = text
        self.start_time = start_time
        self.end_time = end_time


def _trim_silence(wav: np.ndarray, min_keep_sec: float = 5.0,
                  edge_energy: float = 0.012, win_sec: float = 0.03):
    """Drop leading/trailing low-energy frames. Returns (trimmed, head_sec)
    where head_sec is how far into the input the trimmed window starts, so
    downstream timestamps can be offset by (segment_offset + head_sec)."""
    import numpy as np
    sr = 16000
    win = max(64, int(win_sec * sr))
    n = wav.shape[0]
    if n < win * 2:
        return wav, 0.0
    energy = np.sqrt(np.convolve(wav.astype(np.float32) ** 2,
                                 np.ones(win, dtype=np.float32) / win, mode="same"))
    above = np.where(energy >= edge_energy)[0]
    if above.size == 0:
        return wav, 0.0
    cut_lo = max(0, int(above[0]) - win)
    cut_hi = min(n, int(above[-1]) + win)
    if cut_hi - cut_lo < int(min_keep_sec * sr):
        return wav, 0.0
    return wav[cut_lo:cut_hi].astype(np.float32), cut_lo / float(sr)
    """Drop leading/trailing low-energy frames so the model sees the speech
    window rather than seconds of music intro silence. Keeps at least
    ``min_keep_sec`` of audio so very short utterances still transcribe."""
    import numpy as np
    sr = 16000
    win = max(64, int(win_sec * sr))
    n = wav.shape[0]
    if n < win * 2:
        return wav
    energy = np.sqrt(np.convolve(wav.astype(np.float32) ** 2,
                                 np.ones(win, dtype=np.float32) / win, mode="same"))
    # first frame above edge from the start, last above edge from the end
    above = np.where(energy >= edge_energy)[0]
    if above.size == 0:
        return wav
    cut_lo = max(0, int(above[0]) - win)
    cut_hi = min(n, int(above[-1]) + win)
    # respect minimum keep length (centered crop fallback)
    if cut_hi - cut_lo < int(min_keep_sec * sr):
        return wav
    return wav[cut_lo:cut_hi].astype(np.float32)


class BatchWorker(QThread):
    log = Signal(str, str)
    file_started = Signal(int, str)
    file_progress = Signal(int, float)
    segment_result = Signal(int, int, str, str)
    file_done = Signal(int, str, bool, str)
    finished_all = Signal(int, int)

    def __init__(self, model_manager, files: List[str], settings: Settings):
        super().__init__()
        self.model_manager = model_manager
        self.files = files
        self.settings = settings
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        done = failed = 0
        try:
            if not self.model_manager.is_loaded:
                self.model_manager.load(on_log=self._log, on_stage=self._stage)
        except Exception as e:
            self.log.emit(f"模型加载失败: {e}", "error")
            self.finished_all.emit(0, len(self.files))
            return

        out_root = Path(self.settings.output_dir)
        if not out_root.is_absolute():
            out_root = Path(__file__).resolve().parent.parent / out_root
        out_root.mkdir(parents=True, exist_ok=True)

        for i, path in enumerate(self.files):
            if self._cancel:
                self.log.emit("已取消", "warn")
                break
            label = os.path.basename(path)
            self.file_started.emit(i, label)
            self.log.emit(f"[{i+1}/{len(self.files)}] 开始: {label}", "info")
            local_path = path
            needs_cleanup = False
            try:
                local_path = fetch_to_local(path, self.settings.ffmpeg_path)
                needs_cleanup = local_path != path
                if needs_cleanup:
                    self.log.emit("  已缓存到本地临时文件", "info")
                wav, dur = decode_audio(local_path, self.settings.ffmpeg_path, on_log=self._probe_log)
                self.log.emit(f"  解码完成: {dur:.1f}s, {wav.shape[0]} samples", "info")
            except Exception as e:
                self.log.emit(f"  音频解码失败: {e}", "error")
                self.file_done.emit(i, "", False, str(e))
                if needs_cleanup:
                    os.unlink(local_path)
                failed += 1
                continue

            try:
                srt_path = self._transcribe(i, label, wav, dur, out_root)
                done += 1
                self.file_done.emit(i, srt_path, True, srt_path)
            except Exception as e:
                self.log.emit(f"  转录失败: {e}", "error")
                self.file_done.emit(i, "", False, str(e))
                failed += 1
            finally:
                if needs_cleanup:
                    try:
                        os.unlink(local_path)
                    except OSError:
                        pass
        self.finished_all.emit(done, failed)

    def _probe_log(self, kind, msg):
        if kind == "decode":
            self.log.emit(f"  ffmpeg: {msg}", "info")

    def _log(self, level, msg):
        self.log.emit(msg, level)

    def _stage(self, msg):
        self.log.emit(msg, "info")

    def _transcribe(self, idx: int, label: str, wav: np.ndarray, dur: float,
                    out_root: Path) -> str:
        # 120s segments: enough acoustic context for sparse-spoken media, well
        # under the aligner's 180s internal cap so per-chunk timestamps stay
        # word-level accurate.
        seg_len = 120.0
        segments = segment_audio(wav, 16000, seg_len)
        self.log.emit(f"  分段: {len(segments)} 段 (~{seg_len:.0f}s/段, 能量边界)", "info")

        use_ts = self.model_manager.aligner_loaded
        lang = None if self.settings.language == "auto" else self.settings.language

        seg_subs = []
        align_units_all = []
        for j, (seg_wav, off) in enumerate(segments):
            if self._cancel:
                self.log.emit("  取消", "warn")
                raise RuntimeError("cancelled")
            self.file_progress.emit(idx, j / max(1, len(segments)))
            if seg_wav.shape[0] < 16000 * 1.0:
                continue
            # Trim head/tail silence on each chunk before transcription so the
            # model sees a tighter speech window; keep at least 5s for stability.
            use_wav, head_off = _trim_silence(seg_wav, min_keep_sec=5.0)
            seg_orig_off = off + head_off
            if use_wav.shape[0] < 16000 * 1.0:
                continue
            result = self.model_manager.transcribe_one(use_wav, lang, return_time_stamps=use_ts)
            text = (result.text or "").strip()
            rl = result.language or ""
            self.segment_result.emit(idx, j, text, rl)
            self.log.emit(f"    [{j+1}/{len(segments)}] <{rl or '?'}> {text}", "transcript")
            ts = getattr(result, "time_stamps", None)
            if ts is not None and getattr(ts, "items", None):
                for it in ts.items:
                    align_units_all.append(_ShiftedItem(
                        getattr(it, "text", ""), round(getattr(it, "start_time", 0.0) + seg_orig_off, 3),
                        round(getattr(it, "end_time", 0.0) + seg_orig_off, 3)))
            else:
                seg_end = seg_orig_off + (use_wav.shape[0] / 16000.0)
                if text:
                    seg_subs.append((seg_orig_off, seg_end, text))

        if use_ts and align_units_all:
            subs = group_units_to_subtitles(
                align_units_all,
                self.settings.max_chars_per_line,
                self.settings.max_segment_seconds,
                self.settings.merge_blank_lines,
            )
        else:
            subs = subtitles_from_segments(
                seg_subs,
                self.settings.max_chars_per_line,
                self.settings.max_segment_seconds,
                self.settings.merge_blank_lines,
            )

        base = Path(label).stem
        srt_path = str(out_root / f"{base}.srt")
        save_srt(subs, srt_path)
        self.file_progress.emit(idx, 1.0)
        preview = write_srt(subs[:3]) if subs else "(空)"
        self.log.emit(f"  完成 -> {srt_path}  ({len(subs)} 条字幕)", "ok")
        if subs:
            self.log.emit(f"  预览:\n{preview}", "info")
        return srt_path
