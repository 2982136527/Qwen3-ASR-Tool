#!/usr/bin/env python3
"""Standalone model fetcher used to pre-populate ./models for the GUI tool.

Downloads Qwen3-ASR and Qwen3-ForcedAligner weights into a project-local
directory. Source is ModelScope (default, recommended for Mainland China) with
an automatic fallback to Hugging Face.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path


def _download_modelscope(repo_id: str, local_dir: Path) -> None:
    from modelscope.hub.snapshot_download import snapshot_download

    snapshot_download(
        model_id=repo_id,
        local_dir=str(local_dir),
        ignore_file_pattern=[".git", "*.pth", "*.ckpt", "*.onnx", "*.gguf", "*.bin", "*.ot"],
    )


def _download_huggingface(repo_id: str, local_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        ignore_patterns=[".git*", "*.pth", "*.ckpt", "*.onnx", "*.gguf", "*.bin", "original/*", "imgs/*"],
    )


def download_one(repo_id: str, local_dir: Path, source: str = "auto") -> str:
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    marker = local_dir / ".downloaded"
    if marker.exists():
        return f"[skip] {repo_id} already present at {local_dir}"

    sources = ["modelscope", "huggingface"] if source == "auto" else [source]
    last_err = None
    for src in sources:
        try:
            print(f"[fetch] {repo_id} via {src} -> {local_dir}", flush=True)
            if src == "modelscope":
                _download_modelscope(repo_id, local_dir)
            else:
                _download_huggingface(repo_id, local_dir)
            marker.write_text(f"{repo_id}\n{src}\n{time.time()}\n", encoding="utf-8")
            return f"[ok] {repo_id} downloaded via {src}"
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[warn] {src} failed for {repo_id}: {e}", flush=True)
            traceback.print_exc()
    raise RuntimeError(f"All download sources failed for {repo_id}: {last_err}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent / "models"))
    ap.add_argument(
        "--models",
        nargs="*",
        default=["Qwen/Qwen3-ASR-1.7B", "Qwen/Qwen3-ForcedAligner-0.6B", "Qwen/Qwen3-ASR-0.6B"],
    )
    ap.add_argument("--source", default="modelscope", choices=["modelscope", "huggingface", "auto"])
    args = ap.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    for repo_id in args.models:
        name = repo_id.split("/")[-1]
        target = root / name
        try:
            print(download_one(repo_id, target, args.source), flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[err] {repo_id}: {e}", flush=True)
            traceback.print_exc()
    print("[done] all downloads processed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
