"""Headless smoke test: transcribe the first N seconds of a .strm/URL/file.

For .strm / remote URLs the audio stream is first downloaded to a local temp
file (single pass), then every decode_window is instant because the ffmpeg -ss
seek is against a local disk file.  Temp files are cleaned up automatically.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys; sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.audio import decode_window, probe_duration_seconds, fetch_to_local
from src.config import Settings
from src.model_manager import ModelManager
from src.segment import segment_audio
from src.srt import (group_units_to_subtitles, save_srt,
                      subtitles_from_segments, write_srt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--seconds", type=float, default=240.0)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    s = Settings()
    s.device = "mps"
    s.precision = "float16"
    s.enable_aligner = True

    mm = ModelManager(s)
    t = time.time()
    mm.load(on_log=lambda lv, m: print(lv, m), on_stage=lambda x: print("STAGE", x))
    print(f"model loaded: {time.time()-t:.1f}s on {mm.device} ({mm.dtype})")

    local_path = args.path
    needs_cleanup = False
    try:
        local_path = fetch_to_local(args.path, s.ffmpeg_path)
        needs_cleanup = local_path != args.path
        if needs_cleanup:
            print(f"downloaded to local: {local_path}")

        total_sec = probe_duration_seconds(local_path) or args.seconds
        seg_len = 120.0
        use_seconds = min(args.seconds, total_sec)

        print(f"decoding first {use_seconds:.0f}s from local file ...")
        wav, _ = decode_window(local_path, 0.0, use_seconds, s.ffmpeg_path)
        dur = wav.shape[0] / 16000.0
        print(f"audio: {wav.shape[0]} samples, {dur:.1f}s")
        segs = segment_audio(wav, 16000, seg_len)
        print(f"segments: {len(segs)}, target {seg_len:.0f}s/segment")

        units = []; seg_subs = []
        for j, (w, off) in enumerate(segs):
            r = mm.transcribe_one(w, None, return_time_stamps=mm.aligner_loaded)
            print(f"[{j+1}/{len(segs)}] <{r.language or '?'}> {r.text}")
            ts = getattr(r, "time_stamps", None)
            if ts is not None and getattr(ts, "items", None):
                for it in ts.items:
                    units.append(type(it)(text=it.text,
                                          start_time=round(it.start_time + off, 3),
                                          end_time=round(it.end_time + off, 3)))
            else:
                if r.text:
                    seg_subs.append((off, off + w.shape[0]/16000.0, r.text))

        if units:
            subs = group_units_to_subtitles(units, s.max_chars_per_line,
                                            s.max_segment_seconds, True)
        else:
            subs = subtitles_from_segments(seg_subs, s.max_chars_per_line,
                                           s.max_segment_seconds, True)

        out = args.output or f"output/headtest_{Path(args.path).stem}.srt"
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        save_srt(subs, out)
        print(f"\nSAVED: {out}  ({len(subs)} subs)")
        if subs:
            print(write_srt(subs[:8]))
    finally:
        if needs_cleanup:
            try:
                os.unlink(local_path)
                print(f"cleaned up temp file")
            except OSError:
                pass


if __name__ == "__main__":
    main()
