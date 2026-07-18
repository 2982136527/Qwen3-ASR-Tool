"""Audio input handling: ffmpeg-backed decoding for files, URLs and .strm.

Qwen3-ASR wants mono 16kHz float32 PCM in [-1, 1]. We decode everything with
ffmpeg (already present on the user's machine), so MP4/MKV/FLAC/MP3/m4a and
remote streams all flow through one path. ``.strm`` files are tiny text files
that point at a (usually LAN) streaming URL, so we resolve them to that URL.
"""
from __future__ import annotations

import shutil
import os
import tempfile
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np

SAMPLE_RATE = 16000

AUDIO_EXTS = {
    ".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".opus", ".wma",
    ".aiff", ".aif", ".amr", ".webm",
}
VIDEO_EXTS = {
    ".mp4", ".mkv", ".mov", ".avi", ".flv", ".ts", ".m2ts", ".vob",
    ".wmv", ".3gp", ".mpg", ".mpeg", ".m4v",
}
SUPPORTED_EXTS = AUDIO_EXTS | VIDEO_EXTS | {".strm"}


def resolve_ffmpeg(ffmpeg_path: str = "") -> str:
    if ffmpeg_path and Path(ffmpeg_path).is_file():
        return ffmpeg_path
    found = shutil.which("ffmpeg")
    if not found:
        raise FileNotFoundError("ffmpeg not found. Install with: brew install ffmpeg")
    return found


def resolve_ffprobe(ffmpeg_path: str = "") -> str:
    base = Path(resolve_ffmpeg(ffmpeg_path)).parent
    probe = base / "ffprobe"
    if probe.exists():
        return str(probe)
    found = shutil.which("ffprobe")
    if not found:
        raise FileNotFoundError("ffprobe not found. Install with: brew install ffmpeg")
    return found


def is_url(s: str) -> bool:
    try:
        u = urlparse(s)
        return u.scheme in ("http", "https", "rtmp", "rtsp", "rtp") and bool(u.netloc)
    except Exception:
        return False


def is_strm(path: str) -> bool:
    return path.lower().endswith(".strm")


def read_strm_url(path: str) -> str:
    """A .strm file is one URL on its first non-empty line."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                return line
    raise ValueError(f"empty .strm file (no URL found): {path}")


def resolve_input(path_or_url: str) -> str:
    """Map a user-facing path/URL to what ffmpeg should read: URL stays a URL,
    .strm resolves to its inner URL, anything else stays as a filesystem path."""
    s = str(path_or_url).strip()
    if is_url(s):
        return s
    if is_strm(s):
        return read_strm_url(s)
    return s


def support_preview(path_or_url: str) -> str:
    """Human readable tag for UI lists."""
    if is_url(path_or_url):
        return "URL"
    if is_strm(path_or_url):
        return "STRM"
    return Path(path_or_url).suffix.lstrip(".").upper() or "FILE"


def is_supported(path_or_url: str) -> bool:
    s = str(path_or_url)
    if is_url(s) or is_strm(s):
        return True
    return Path(s).suffix.lower() in SUPPORTED_EXTS


def probe_duration_seconds(path_or_url: str, ffmpeg_path: str = "") -> Optional[float]:
    ffprobe = resolve_ffprobe(ffmpeg_path)
    src = resolve_input(path_or_url)
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", src],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=30,
        )
    except Exception:
        return None
    if out.returncode != 0:
        return None
    try:
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def decode_audio(path_or_url: str, ffmpeg_path: str = "",
                 on_log=None) -> Tuple[np.ndarray, float]:
    """Decode any supported input to mono 16kHz float32 PCM in [-1, 1].

    Returns (waveform, duration_seconds). ffmpeg streams f32le PCM to stdout;
    we read it into a float32 numpy array. Used for both batch files and the
    realtime engine's utterance buffers.
    """
    ffmpeg = resolve_ffmpeg(ffmpeg_path)
    src = resolve_input(path_or_url)

    # Aggressive demuxer settings: re doux silence on ffmpeg for flaky LAN
    # redirect servers, plus no stdin interaction and generous err detection.
    cmd = [
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
        "-err_detect", "ignore_err",
        "-i", src,
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-f", "f32le", "-",
    ]
    if on_log:
        on_log("decode", f"ffmpeg: {src}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    chunks: List[bytes] = []
    try:
        while True:
            data = proc.stdout.read(1024 * 256)
            if not data:
                break
            chunks.append(data)
    finally:
        proc.stdout.close()
        stderr = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
        proc.wait()
    if proc.returncode != 0 and not chunks:
        raise RuntimeError(f"ffmpeg failed: {stderr.strip() or 'unknown error'}")
    if not chunks:
        raise RuntimeError("ffmpeg produced no audio (check the file/URL is reachable)")
    pcm = np.frombuffer(b"".join(chunks), dtype=np.float32)
    if pcm.size and float(np.max(np.abs(pcm))) > 1.0:
        peak = float(np.max(np.abs(pcm)))
        if peak > 0:
            pcm = (pcm / peak).astype(np.float32)
    pcm = np.clip(pcm, -1.0, 1.0)
    dur = pcm.shape[0] / float(SAMPLE_RATE)
    return pcm, dur


def decode_window(path_or_url: str, start_sec: float, duration_sec: float,
                  ffmpeg_path: str = "", on_log=None) -> Tuple[np.ndarray, float]:
    """Decode a window [start_sec, start_sec+duration_sec) of any input.

    Uses ffmpeg's accurate seek: ``-ss`` *after* ``-i`` so the seek is sample
    accurate (input seeking skips to the nearest keyframe and is faster but
    imprecise). For local files this is still near-instant; for remote URLs it
    means ffmpeg pulls bytes up to the seek point, but that is unavoidable.

    This is the right primitive for transcribing hour-long sources (e.g. the
    2.5h .strm LAN stream): we never hold the whole PCM in memory, and each
    window reads only what the model needs.
    """
    ffmpeg = resolve_ffmpeg(ffmpeg_path)
    src = resolve_input(path_or_url)
    cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
           "-err_detect", "ignore_err",
           "-ss", f"{start_sec:.3f}", "-t", f"{duration_sec:.3f}",
           "-i", src,
           "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
           "-f", "f32le", "-"]
    if on_log:
        on_log("decode", f"ffmpeg -ss {start_sec:.1f} -t {duration_sec:.1f}: {src}")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 and not proc.stdout:
        err = proc.stderr.decode("utf-8", "replace")
        raise RuntimeError(f"ffmpeg failed at {start_sec:.1f}s: {err.strip() or 'unknown'}")
    if not proc.stdout:
        raise RuntimeError(f"ffmpeg returned no audio at {start_sec:.1f}s (check URL)")
    pcm = np.frombuffer(proc.stdout, dtype=np.float32)
    if pcm.size and float(np.max(np.abs(pcm))) > 1.0:
        peak = float(np.max(np.abs(pcm)))
        if peak > 0:
            pcm = (pcm / peak).astype(np.float32)
    pcm = np.clip(pcm, -1.0, 1.0)
    dur = pcm.shape[0] / float(SAMPLE_RATE)
    return pcm, dur


def fetch_to_local(path_or_url: str, ffmpeg_path: str = "",
                   suffix: str = ".m4a") -> str:
    """Download audio from a remote source to a local temp file.

    For .strm / URL inputs this pulls the audio stream once into a local
    container (~150 MB for 2.5 h AAC).  For plain local files it returns the
    input path unchanged.  The returned path lives until the caller deletes
    it (``os.unlink`` or ``Path.unlink``).
    """
    ffmpeg = resolve_ffmpeg(ffmpeg_path)
    src = resolve_input(path_or_url)
    if not is_url(src):
        return path_or_url

    fd, tmp = tempfile.mkstemp(suffix=suffix, prefix="qwen3asr_")
    os.close(fd)
    cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
           "-err_detect", "ignore_err",
           "-i", src,
           "-vn", "-acodec", "copy",
           "-y", tmp]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace")
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise RuntimeError(f"ffmpeg download failed: {err.strip() or 'unknown'}")
    return tmp


def filter_supported(paths: List[str]) -> List[str]:
    return [p for p in paths if is_supported(p)]
