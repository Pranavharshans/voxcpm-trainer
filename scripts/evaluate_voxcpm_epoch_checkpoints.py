#!/usr/bin/env python3
"""Generate deferred evaluation artifacts from epoch checkpoints.

FSDP training saves portable full checkpoints. Once torchrun exits, this
script loads each epoch checkpoint as a regular single-GPU model and writes
WAV/TensorBoard artifacts without crossing FSDP's custom-forward boundary.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "scripts"))

import torch
from tensorboardX import SummaryWriter

from train_voxcpm_finetune import generate_sample_audio
from voxcpm.model import VoxCPM2Model, VoxCPMModel
from voxcpm.training import Accelerator, TrainingTracker, load_audio_text_datasets
from voxcpm.training.config import load_yaml_config
from voxcpm.training.text import normalize_training_text


def _checkpoint_step(checkpoint_dir: Path, fallback: int) -> int:
    state_path = checkpoint_dir / "training_state.json"
    if not state_path.is_file():
        return fallback
    with state_path.open("r", encoding="utf-8") as handle:
        resume_step = int(json.load(handle).get("step", fallback + 1))
    return max(0, resume_step - 1)


def _model_class(checkpoint_dir: Path):
    with (checkpoint_dir / "config.json").open("r", encoding="utf-8") as handle:
        architecture = str(json.load(handle).get("architecture", "voxcpm")).lower()
    return VoxCPM2Model if architecture == "voxcpm2" else VoxCPMModel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    args = parser.parse_args()

    config = load_yaml_config(args.config_path)
    is_fsdp = str(config.get("distributed_strategy", "ddp")).strip().lower() == "fsdp"
    num_samples = int(config.get("eval_audio_samples", 0))
    save_eval_audio = bool(config.get("save_eval_audio", False))
    num_epochs = int(config.get("num_epochs", 0))

    if not is_fsdp or not save_eval_audio or num_samples <= 0 or num_epochs <= 0:
        print("Deferred checkpoint evaluation is not enabled for this training profile.")
        return 0

    save_dir = Path(config["save_path"])
    run_dir = save_dir.parent
    val_manifest = str(config.get("val_manifest", ""))
    if not val_manifest:
        raise ValueError("Deferred evaluation requires val_manifest")

    prompt_audio = str(config.get("eval_prompt_audio", ""))
    prompt_text = str(config.get("eval_prompt_text", ""))
    if bool(prompt_audio) != bool(prompt_text):
        raise ValueError("eval_prompt_audio and eval_prompt_text must both be set or both be empty")
    if prompt_audio and not Path(prompt_audio).is_file():
        raise FileNotFoundError(f"Evaluation prompt audio does not exist: {prompt_audio}")

    checkpoint_dirs = [save_dir / f"epoch_{epoch:02d}" for epoch in range(1, num_epochs + 1)]
    missing = [str(path) for path in checkpoint_dirs if not path.is_dir()]
    if missing:
        raise FileNotFoundError("Missing epoch checkpoint(s): " + ", ".join(missing))

    sample_rate = int(config.get("sample_rate", 16_000))
    out_sample_rate = int(config.get("out_sample_rate", 0))
    eval_ds, _ = load_audio_text_datasets(
        train_manifest=val_manifest,
        sample_rate=sample_rate,
    )
    val_texts = [
        normalize_training_text(text, str(config.get("text_normalization", "NFC")))
        for text in eval_ds["text"]
    ]

    accelerator = Accelerator(amp=False)
    if accelerator.world_size != 1:
        raise RuntimeError("Deferred checkpoint evaluation must run as a single process")

    tracker = TrainingTracker(
        writer=None,
        log_file=str(run_dir / "deferred_eval.log"),
        rank=0,
    )
    writer = SummaryWriter(log_dir=str(config.get("tensorboard", run_dir / "tensorboard")))
    try:
        for epoch, checkpoint_dir in enumerate(checkpoint_dirs, start=1):
            tracker.print(f"[Deferred Eval] Loading {checkpoint_dir}")
            model_cls = _model_class(checkpoint_dir)
            model = model_cls.from_local(
                str(checkpoint_dir),
                optimize=False,
                training=False,
            )
            audio_vae = model.audio_vae
            step = _checkpoint_step(checkpoint_dir, fallback=epoch)

            generate_sample_audio(
                model,
                eval_ds,
                audio_vae,
                writer,
                step,
                accelerator,
                sample_rate,
                out_sample_rate=out_sample_rate,
                val_texts=val_texts,
                tokenizer=model.text_tokenizer,
                valid_interval=0,
                tracker=tracker,
                num_samples=num_samples,
                audio_output_dir=run_dir / "eval_samples",
                audio_label=f"epoch_{epoch:02d}",
                prompt_audio_path=prompt_audio or None,
                prompt_text=prompt_text or None,
                write_outputs=True,
            )

            del model, audio_vae
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        writer.close()

    tracker.print(f"[Deferred Eval] Completed. WAV files: {run_dir / 'eval_samples'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
