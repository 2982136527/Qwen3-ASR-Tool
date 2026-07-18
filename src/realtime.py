"""Real-time (near-realtime) dictation engine.

Qwen3-ASR streaming is vLLM/CUDA-only, so on Apple Silicon/CPU we approximate
real-time dictation: capture microphone audio continuously, do energy-based
endpointing (a trailing silence of ``pause_seconds`` ends an utterance, plus a
``max_utterance_seconds`` safety cut). Each completed utterance is transcribed
offline and its text is emitted live. Latency is the size of the utterance plus
a couple seconds of decode on the 1.7B model.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from PyQt6.QtCore import QThread, pyqtSignal as Signal

from .config import Settings

SAMPLE_RATE = 16000
FRAME_SEC = 0.03


class RealtimeWorker(QThread):
    log = Signal(str, str)
    level = Signal(float)            # instantaneous RMS 0..1
    state = Signal(str)             # "idle" | "listening" | "speaking" | "decoding"
    utterance = Signal(str, str)     # text, language
    transcript_append = Signal(str)  # text (raw appended into running transcript)

    def __init__(self, model_manager, settings: Settings):
        super().__init__()
        self.model_manager = model_manager
        self.settings = settings
        self._stop = False
        self._stream = None
        self._pa = None

    def stop(self):
        self._stop = True

    def _rms(self, x: np.ndarray) -> float:
        if x.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(x.astype(np.float32) ** 2)))

    def run(self):
        try:
            import pyaudio
        except Exception as e:
            self.log.emit(f"PyAudio 不可用: {e}", "error")
            return
        try:
            if not self.model_manager.is_loaded:
                self.model_manager.load(on_log=lambda lv, m: self.log.emit(m, lv),
                                        on_stage=lambda s: self.log.emit(s, "info"))
        except Exception as e:
            self.log.emit(f"模型加载失败: {e}", "error")
            return
        if not self.model_manager.is_loaded:
            return

        # 1=least aggressive .. 5=most aggressive (react to quieter rooms)
        sens = max(1, min(5, int(self.settings.vad_sensitivity)))
        on_thresh = {1: 0.035, 2: 0.025, 3: 0.018, 4: 0.012, 5: 0.008}[sens]
        off_mult = {1: 0.45, 2: 0.50, 3: 0.55, 4: 0.60, 5: 0.65}[sens]
        pause_s = float(self.settings.pause_seconds)
        max_utt_s = float(self.settings.max_utterance_seconds)

        frame = int(SAMPLE_RATE * FRAME_SEC)
        try:
            self._pa = pyaudio.PyAudio()
            dev = self.settings.input_device_index if self.settings.input_device_index >= 0 else None
            self._stream = self._pa.open(
                format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE,
                input=True, input_device_index=dev, frames_per_buffer=frame,
            )
        except Exception as e:
            self.log.emit(f"麦克风打开失败: {e}", "error")
            self._cleanup()
            return

        self.log.emit("实时听写已开始", "ok")
        self.state.emit("listening")
        buf = np.zeros(0, dtype=np.float32)
        speaking = False
        silence_samples = 0
        silence_limit = int(pause_s / FRAME_SEC)
        max_frames = int(max_utt_s / FRAME_SEC)
        consecutive = 0

        try:
            while not self._stop:
                try:
                    raw = self._stream.read(frame, exception_on_overflow=False)
                except Exception as e:
                    self.log.emit(f"读取麦克风出错: {e}", "warn")
                    break
                if self._stop:
                    break
                x = np.frombuffer(raw, dtype=np.float32)
                x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                r = self._rms(x)
                self.level.emit(min(1.0, r / max(on_thresh * 2, 1e-4)))

                if r >= on_thresh:
                    speaking = True
                    silence_samples = 0
                    if consecutive == 0:
                        self.state.emit("speaking")
                    consecutive += 1
                else:
                    if speaking:
                        silence_samples += 1

                buf = np.concatenate([buf, x]) if buf.size else x.copy()

                flush = False
                if speaking and silence_samples >= silence_limit:
                    flush = True
                elif buf.shape[0] / SAMPLE_RATE >= max_utt_s and speaking:
                    flush = True

                if flush:
                    # trim a bit of tailing silence for the model
                    keep = max(int((buf.shape[0] / SAMPLE_RATE - 0.15) * SAMPLE_RATE), 1)
                    utt = buf[:keep].copy()
                    buf = np.zeros(0, dtype=np.float32)
                    speaking = False
                    silence_samples = 0
                    consecutive = 0
                    self.state.emit("decoding")
                    if utt.shape[0] < SAMPLE_RATE * 0.4:
                        self.state.emit("listening")
                        continue
                    try:
                        lang = None if self.settings.language == "auto" else self.settings.language
                        res = self.model_manager.transcribe_one(utt, lang, return_time_stamps=False)
                        text = (res.text or "").strip()
                        rl = res.language or ""
                        if text:
                            self.utterance.emit(text, rl)
                            self.transcript_append.emit(text)
                            self.log.emit(f"<{rl or '?'}> {text}", "transcript")
                    except Exception as e:
                        self.log.emit(f"识别出错: {e}", "error")
                    self.state.emit("listening")
        finally:
            self._cleanup()
            self.log.emit("实时听写已停止", "info")
            self.state.emit("idle")

    def _cleanup(self):
        s = getattr(self, "_stream", None)
        if s is not None:
            try:
                if s.is_active():
                    s.stop_stream()
                s.close()
            except Exception:
                pass
            self._stream = None
        p = getattr(self, "_pa", None)
        if p is not None:
            try:
                p.terminate()
            except Exception:
                pass
            self._pa = None
