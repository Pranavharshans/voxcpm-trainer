#!/usr/bin/env python3
"""Stream a deterministic Praha-Labs/TTS-Ml subset into VoxCPM manifests."""

from __future__ import annotations

import argparse
import io
import json
import sys
import unicodedata
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from scipy.signal import resample_poly


DATASET_ID = "Praha-Labs/TTS-Ml"
DATASET_REVISION = "33cef946925f89ee48511951da3049f5281cfd2e"
SAMPLE_RATE = 16_000
PREPARATION_VERSION = 2


def _is_malayalam(text: str) -> bool:
    return any("\u0d00" <= char <= "\u0d7f" for char in text)


def _extract_audio(row: dict) -> tuple[np.ndarray, int]:
    audio = row.get("audio")
    if not isinstance(audio, dict):
        raise ValueError("row does not contain an audio object")

    if audio.get("bytes") is not None:
        samples, sampling_rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=False)
    elif audio.get("path"):
        samples, sampling_rate = sf.read(audio["path"], dtype="float32", always_2d=False)
    elif "array" in audio:
        samples = np.asarray(audio["array"], dtype=np.float32)
        sampling_rate = int(audio.get("sampling_rate", 0))
    else:
        raise ValueError("audio object has neither bytes, path, nor decoded samples")

    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=0 if samples.shape[0] <= 2 else 1)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError(f"expected non-empty mono audio, got shape {samples.shape}")
    if not np.isfinite(samples).all():
        raise ValueError("audio contains NaN or infinite values")
    if sampling_rate <= 0:
        raise ValueError(f"invalid source sampling rate: {sampling_rate}")

    source_sampling_rate = int(sampling_rate)
    if source_sampling_rate != SAMPLE_RATE:
        divisor = gcd(source_sampling_rate, SAMPLE_RATE)
        samples = resample_poly(
            samples,
            up=SAMPLE_RATE // divisor,
            down=source_sampling_rate // divisor,
        ).astype(np.float32, copy=False)
        if samples.size == 0 or not np.isfinite(samples).all():
            raise ValueError(f"resampling from {source_sampling_rate} Hz produced invalid audio")

    return samples, source_sampling_rate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--total-samples", type=int, default=20_000)
    parser.add_argument("--validation-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=1_000)
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.total_samples <= 0:
        parser.error("--total-samples must be positive")
    if not 0 <= args.validation_samples < args.total_samples:
        parser.error("--validation-samples must be between 0 and total-samples - 1")
    if args.shuffle_buffer <= 0:
        parser.error("--shuffle-buffer must be positive")

    output_dir = args.output_dir.resolve()
    audio_root = output_dir / "audio"
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_root.mkdir(parents=True, exist_ok=True)

    final_train = output_dir / "train.jsonl"
    final_val = output_dir / "val.jsonl"
    completion = output_dir / "subset_info.json"
    if completion.is_file():
        metadata = json.loads(completion.read_text(encoding="utf-8"))
        if (
            metadata.get("preparation_version") == PREPARATION_VERSION
            and metadata.get("dataset") == DATASET_ID
            and metadata.get("dataset_revision") == DATASET_REVISION
            and metadata.get("total_samples") == args.total_samples
            and metadata.get("validation_samples") == args.validation_samples
            and metadata.get("seed") == args.seed
            and final_train.is_file()
            and final_val.is_file()
        ):
            print(f"Subset already complete at {output_dir}")
            return 0

    stream = load_dataset(
        DATASET_ID,
        split="train",
        revision=DATASET_REVISION,
        streaming=True,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
    ).cast_column("audio", Audio(decode=False))
    stream = stream.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    train_tmp = output_dir / "train.jsonl.tmp"
    val_tmp = output_dir / "val.jsonl.tmp"
    accepted = 0
    skipped = 0
    resampled = 0
    source_rows_seen = 0

    with train_tmp.open("w", encoding="utf-8") as train_handle, val_tmp.open("w", encoding="utf-8") as val_handle:
        for row in stream:
            if accepted >= args.total_samples:
                break
            source_rows_seen += 1
            try:
                text = row.get("text")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("empty transcript")
                text = unicodedata.normalize("NFC", text.strip())
                if not _is_malayalam(text):
                    raise ValueError("transcript has no Malayalam code points")
                samples, source_sampling_rate = _extract_audio(row)
            except Exception as exc:
                skipped += 1
                if skipped <= 20:
                    print(f"Skipping source row {source_rows_seen}: {exc}", file=sys.stderr)
                continue

            if source_sampling_rate != SAMPLE_RATE:
                resampled += 1

            shard_dir = audio_root / f"{accepted // 1000:03d}"
            shard_dir.mkdir(parents=True, exist_ok=True)
            audio_path = shard_dir / f"{accepted:06d}.flac"
            # Always rewrite partial output. A previous interrupted run may
            # have used different acceptance or resampling logic, in which
            # case reusing a numbered file could pair it with the wrong text.
            temp_audio = audio_path.with_suffix(".flac.tmp")
            sf.write(temp_audio, samples, SAMPLE_RATE, format="FLAC", subtype="PCM_16")
            temp_audio.replace(audio_path)

            manifest_row = {
                "audio": str(audio_path),
                "text": text,
                "dataset_id": 0,
            }
            destination = val_handle if accepted < args.validation_samples else train_handle
            destination.write(json.dumps(manifest_row, ensure_ascii=False) + "\n")
            accepted += 1

            if accepted % 1_000 == 0:
                print(f"Prepared {accepted}/{args.total_samples} samples (resampled={resampled})")

    if accepted != args.total_samples:
        raise RuntimeError(
            f"Dataset stream ended after {accepted} accepted samples "
            f"({source_rows_seen} rows inspected, {skipped} skipped)"
        )

    train_tmp.replace(final_train)
    val_tmp.replace(final_val)
    metadata = {
        "preparation_version": PREPARATION_VERSION,
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "split": "train",
        "total_samples": args.total_samples,
        "training_samples": args.total_samples - args.validation_samples,
        "validation_samples": args.validation_samples,
        "sample_rate": SAMPLE_RATE,
        "seed": args.seed,
        "shuffle_buffer": args.shuffle_buffer,
        "source_rows_seen": source_rows_seen,
        "skipped_rows": skipped,
        "resampled_rows": resampled,
    }
    completion.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
