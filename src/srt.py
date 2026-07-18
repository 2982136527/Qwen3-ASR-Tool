"""SRT subtitle generation from Qwen3-ForcedAligner timestamps.

When the aligner runs we get per-unit (word/character) spans with start/end
times. Subtitles want lines, so we greedily pack units into a line that stays
under the chosen chars-per-line and max-duration limits, breaking on a CJK
boundary when the running text would overflow. The result is a standards-compliant
SRT that Jellyfin/EMBY and editors load directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Subtitle:
    index: int
    start: float
    end: float
    text: str


def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return (
        0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF
        or 0x20000 <= o <= 0x2A6DF or 0xF900 <= o <= 0xFAFF
        or 0x3040 <= o <= 0x30FF or 0xAC00 <= o <= 0xD7AF
    )


def _count_chars_for_break(text: str) -> int:
    return len(text)


def _fmt_ts(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    ms = int(round((seconds - int(seconds)) * 1000))
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


MAX_LINES = 2   # at most 2 visible lines inside one subtitle entry


def group_units_to_subtitles(units, max_chars: int, max_duration: float,
                              merge_blanks: bool = True) -> List[Subtitle]:
    """``units`` = iterable of objects with .text/.start_time/.end_time."""
    # Gap (seconds) between adjacent items that counts as a "pause" worthy
    # of breaking into a new subtitle line.  Shorter gaps are continuous
    # speech and should stay together in one entry.
    GAP_THRESHOLD = 1.0
    MIN_DURATION = 1.5   # minimum seconds a subtitle stays on screen
    items = []
    for u in units:
        items.append((str(getattr(u, "text", "")), float(getattr(u, "start_time", 0.0)),
                      float(getattr(u, "end_time", 0.0))))

    def _join_keep_space(prev: str, cur: str) -> str:
        # Latinate tokens (English/French/etc) come back without surrounding
        # spaces from the aligner, so insert one between adjacent Latin tokens.
        # CJK does not need a separator; Japanese/Korean use spaces where the
        # aligner already provides them.
        if not prev or not cur:
            return prev + cur
        last = prev[-1]
        first = cur[0]
        sep = ""
        if (last not in (" ", "\t")) and (first not in (" ", "\t")):
            latin_last = not _is_cjk(last)
            latin_first = not _is_cjk(first)
            # add a space only between two latin letters/digits/punct
            if latin_last and latin_first:
                # avoid sticking a space before sentence-final punctuation
                if first in ".,;:!?)]}'\"":
                    sep = ""
                else:
                    sep = " "
        return prev + sep + cur

    subs: List[Subtitle] = []
    cur_text = ""
    cur_start: float | None = None
    cur_end: float = 0.0

    def flush():
        nonlocal cur_text, cur_start, cur_end
        s = cur_text.strip()
        if not s or cur_start is None:
            cur_text, cur_start = "", None
            cur_end = 0.0
            return
        if subs and cur_start - subs[-1].end < 0.001 and not subs[-1].text:
            pass
        if merge_blanks and subs and cur_start < subs[-1].end + 0.05 and not s:
            cur_text, cur_start = "", None
            cur_end = 0.0
            return
        subs.append(Subtitle(index=0, start=cur_start, end=(cur_end if cur_end > cur_start else cur_start + 0.5), text=s))
        cur_text, cur_start = "", None
        cur_end = 0.0

    prev_et: float | None = None
    cur_lines: int = 0
    for text, st, et in items:
        if not text or not text.strip():
            continue
        # A meaningful pause between words → flush the current line and
        # start a fresh one.  Short gaps (sub-second) are natural speech
        # rhythm and should not be broken.
        if cur_text and prev_et is not None and (st - prev_et) > GAP_THRESHOLD:
            flush()
            cur_start = st
        if cur_start is None:
            cur_start = st
        # line break within same subtitle entry (not a new entry)
        if cur_text and _count_chars_for_break(cur_text + text) > max_chars and cur_lines < MAX_LINES:
            cur_text = cur_text.rstrip() + "\n"
            cur_lines += 1
            candidate = cur_text + text
        else:
            candidate = _join_keep_space(cur_text, text) if cur_text else text
        over_chars = _count_chars_for_break(candidate) > max_chars
        over_dur = (et - cur_start) > max_duration
        if (over_chars or over_dur) and cur_text:
            # only flush to a brand-new entry on over_dur or gap; over_chars
            # is now handled by the inline line-break above.
            if over_dur:
                flush()
                cur_start = st
                candidate = _join_keep_space("", text)
                cur_lines = 0
        cur_text = candidate.strip()
        cur_end = et
        prev_et = et
    flush()

    for i, sb in enumerate(subs, 1):
        sb.index = i
        # ensure minimum on-screen duration so short entries don't flash by
        if sb.end - sb.start < MIN_DURATION:
            sb.end = round(sb.start + MIN_DURATION, 3)
        # normalize double newlines that may have crept in
        sb.text = sb.text.replace("\n\n", "\n")
    return subs


def subtitles_from_segments(segments, max_chars: int, max_duration: float,
                             merge_blanks: bool = True) -> List[Subtitle]:
    """segments: list of (start_sec, end_sec, text). Build one subtitle per
    non-empty segment, snapped to the segment's time bounds (used when the
    aligner is off)."""
    subs: List[Subtitle] = []
    idx = 1
    for st, et, text in segments:
        s = (text or "").strip()
        if not s:
            continue
        if et <= st:
            et = st + 0.6
        if et - st > max_duration:
            et = st + max_duration
        subs.append(Subtitle(index=idx, start=float(st), end=float(et), text=s))
        idx += 1
    return subs


def write_srt(subs: List[Subtitle]) -> str:
    out = []
    for sb in subs:
        out.append(f"{sb.index}\n{_fmt_ts(sb.start)} --> {_fmt_ts(sb.end)}\n{sb.text}\n")
    return "\n".join(out)


def save_srt(subs: List[Subtitle], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(write_srt(subs))
        f.write("\n")
    # also utf-8 BOM-free txt transcript for convenience
    txt_path = str(path)[:-4] + ".txt"
    try:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(sb.text for sb in subs))
    except Exception:
        pass
