#!/usr/bin/env python3
"""Audit VoxCPM2 text-tokenizer behavior on a Malayalam JSONL manifest."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from transformers import LlamaTokenizerFast

from voxcpm.model.utils import mask_multichar_chinese_tokens
from voxcpm.training.text import normalize_training_text


def _is_malayalam(text: str) -> bool:
    return any("\u0d00" <= char <= "\u0d7f" for char in text)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return ordered[index]


def _iter_texts(manifest: Path, limit: int):
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if limit and line_number > limit:
                break
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {manifest}:{line_number}: {exc}") from exc
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Missing non-empty 'text' at {manifest}:{line_number}")
            yield line_number, normalize_training_text(text, "NFC")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained-path", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=10_000, help="Rows to inspect; 0 means all rows")
    parser.add_argument("--show-samples", type=int, default=3)
    args = parser.parse_args()

    base_tokenizer = LlamaTokenizerFast.from_pretrained(args.pretrained_path)
    tokenizer = mask_multichar_chinese_tokens(base_tokenizer)
    unk_id = base_tokenizer.unk_token_id

    token_counts: list[int] = []
    token_char_ratios: list[float] = []
    unknown_tokens = 0
    malayalam_rows = 0
    total_rows = 0

    for line_number, text in _iter_texts(args.manifest, args.limit):
        total_rows += 1
        if _is_malayalam(text):
            malayalam_rows += 1
        token_ids = tokenizer(text)
        token_counts.append(len(token_ids))
        token_char_ratios.append(len(token_ids) / max(1, len(text)))
        if unk_id is not None:
            unknown_tokens += sum(token_id == unk_id for token_id in token_ids)

        if total_rows <= args.show_samples:
            print(
                f"sample line={line_number} chars={len(text)} tokens={len(token_ids)} "
                f"ratio={token_char_ratios[-1]:.2f}"
            )

    if total_rows == 0:
        print("No manifest rows were audited.", file=sys.stderr)
        return 2

    print(f"rows={total_rows}")
    print(f"malayalam_rows={malayalam_rows}")
    print(f"unknown_tokens={unknown_tokens}")
    print(f"tokens_mean={statistics.mean(token_counts):.1f}")
    print(f"tokens_p95={_percentile([float(value) for value in token_counts], 0.95):.0f}")
    print(f"tokens_per_character_mean={statistics.mean(token_char_ratios):.2f}")
    print(f"tokens_per_character_p95={_percentile(token_char_ratios, 0.95):.2f}")

    if malayalam_rows == 0:
        print("ERROR: no Malayalam code points were found in the audited rows.", file=sys.stderr)
        return 2
    if unknown_tokens:
        print("ERROR: tokenizer emitted unknown tokens; do not start training yet.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
