#!/usr/bin/env python3
"""Stream only ai4bharat/Rasa's Malayalam config into VoxCPM manifests."""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import unicodedata
from collections import Counter
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from scipy.signal import resample_poly


DATASET_ID = "ai4bharat/Rasa"
DATASET_CONFIG = "Malayalam"
DATASET_REVISION = "632f55c7ac590219d41cd7adffce5b440e4604f5"
TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "test"
SAMPLE_RATE = 16_000
PREPARATION_VERSION = 1
TEXT_COLUMNS = ("text", "transcription", "sentence", "normalized_text")
AUDIO_COLUMNS = ("audio", "speech", "wav")


def _is_malayalam(text: str) -> bool:
    return any("\u0d00" <= char <= "\u0d7f" for char in text)


def _first_present(row: dict, columns: tuple[str, ...], kind: str):
    for column in columns:
        if column in row and row[column] is not None:
            return row[column]
    raise ValueError(f"row has no supported {kind} column; columns={sorted(row)}")


def _extract_text(row: dict) -> str:
    text = _first_present(row, TEXT_COLUMNS, "text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty transcript")
    text = unicodedata.normalize("NFC", text.strip())
    if not _is_malayalam(text):
        raise ValueError("transcript has no Malayalam code points")
    return text


def _extract_audio(row: dict) -> tuple[np.ndarray, int]:
    audio = _first_present(row, AUDIO_COLUMNS, "audio")
    if not isinstance(audio, dict):
        raise ValueError("audio value is not an Audio object")

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


def _stream(split: str, seed: int, shuffle_buffer: int, cache_dir: Path | None):
    # Selecting DATASET_CONFIG is the boundary that prevents datasets from
    # resolving or downloading the other 21 language directories.
    dataset = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split=split,
        revision=DATASET_REVISION,
        streaming=True,
        token=True,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    dataset = dataset.cast_column("audio", Audio(decode=False))
    return dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)


def _write_split(
    *,
    split: str,
    destination: Path,
    output_dir: Path,
    limit: int | None,
    seed: int,
    shuffle_buffer: int,
    cache_dir: Path | None,
) -> dict:
    audio_root = output_dir / "audio" / split
    audio_root.mkdir(parents=True, exist_ok=True)
    temp_manifest = destination.with_suffix(destination.suffix + ".tmp")
    accepted = 0
    skipped = 0
    resampled = 0
    source_rows_seen = 0
    genders: Counter[str] = Counter()

    with temp_manifest.open("w", encoding="utf-8") as handle:
        for row in _stream(split, seed, shuffle_buffer, cache_dir):
            if limit is not None and accepted >= limit:
                break
            source_rows_seen += 1
            try:
                text = _extract_text(row)
                samples, source_sampling_rate = _extract_audio(row)
            except Exception as exc:
                skipped += 1
                if skipped <= 20:
                    print(f"Skipping {split} row {source_rows_seen}: {exc}", file=sys.stderr)
                continue

            if source_sampling_rate != SAMPLE_RATE:
                resampled += 1

            shard_dir = audio_root / f"{accepted // 1000:03d}"
            shard_dir.mkdir(parents=True, exist_ok=True)
            audio_path = shard_dir / f"{accepted:06d}.flac"
            temp_audio = audio_path.with_suffix(".flac.tmp")
            sf.write(temp_audio, samples, SAMPLE_RATE, format="FLAC", subtype="PCM_16")
            temp_audio.replace(audio_path)

            gender = str(row.get("gender", "unknown")).strip().lower() or "unknown"
            genders[gender] += 1
            handle.write(
                json.dumps(
                    {"audio": str(audio_path), "text": text, "dataset_id": 0},
                    ensure_ascii=False,
                )
                + "\n"
            )
            accepted += 1
            if accepted % 1_000 == 0:
                target = "all" if limit is None else str(limit)
                print(f"Prepared Rasa {split}: {accepted}/{target} (resampled={resampled})")

    if limit is not None and accepted != limit:
        raise RuntimeError(
            f"Rasa {split} ended after {accepted} accepted rows; expected {limit} "
            f"({source_rows_seen} inspected, {skipped} skipped)"
        )
    if accepted == 0:
        raise RuntimeError(f"Rasa {split} produced no valid rows")
    temp_manifest.replace(destination)
    return {
        "accepted": accepted,
        "source_rows_seen": source_rows_seen,
        "skipped": skipped,
        "resampled": resampled,
        "genders": dict(sorted(genders.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--validation-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=1_000)
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.validation_samples <= 0:
        parser.error("--validation-samples must be positive")
    if args.shuffle_buffer <= 0:
        parser.error("--shuffle-buffer must be positive")
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required for the gated ai4bharat/Rasa dataset")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final_train = output_dir / "train.jsonl"
    final_val = output_dir / "val.jsonl"
    completion = output_dir / "subset_info.json"

    if completion.is_file():
        metadata = json.loads(completion.read_text(encoding="utf-8"))
        if (
            metadata.get("preparation_version") == PREPARATION_VERSION
            and metadata.get("dataset") == DATASET_ID
            and metadata.get("dataset_config") == DATASET_CONFIG
            and metadata.get("dataset_revision") == DATASET_REVISION
            and metadata.get("validation_samples_requested") == args.validation_samples
            and metadata.get("seed") == args.seed
            and final_train.is_file()
            and final_val.is_file()
            and (output_dir / "eval_prompt.flac").is_file()
            and (output_dir / "eval_prompt.txt").is_file()
        ):
            print(f"Rasa Malayalam subset already complete at {output_dir}")
            return 0

    validation = _write_split(
        split=VALIDATION_SPLIT,
        destination=final_val,
        output_dir=output_dir,
        limit=args.validation_samples,
        seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
        cache_dir=args.cache_dir,
    )
    first_validation = json.loads(final_val.read_text(encoding="utf-8").splitlines()[0])
    prompt_audio = output_dir / "eval_prompt.flac"
    prompt_text = output_dir / "eval_prompt.txt"
    shutil.copy2(first_validation["audio"], prompt_audio)
    prompt_text.write_text(first_validation["text"] + "\n", encoding="utf-8")

    training = _write_split(
        split=TRAIN_SPLIT,
        destination=final_train,
        output_dir=output_dir,
        limit=None,
        seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
        cache_dir=args.cache_dir,
    )

    metadata = {
        "preparation_version": PREPARATION_VERSION,
        "dataset": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "dataset_revision": DATASET_REVISION,
        "train_split": TRAIN_SPLIT,
        "validation_split": VALIDATION_SPLIT,
        "validation_samples_requested": args.validation_samples,
        "sample_rate": SAMPLE_RATE,
        "seed": args.seed,
        "shuffle_buffer": args.shuffle_buffer,
        "training": training,
        "validation": validation,
        "eval_prompt_audio": str(prompt_audio),
        "eval_prompt_text_file": str(prompt_text),
    }
    completion.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
