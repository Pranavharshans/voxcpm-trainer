# Malayalam fine-tuning on FAU Alex

This repository uses a conservative first-stage plan: keep the released
VoxCPM2 text vocabulary, normalize Malayalam transcripts to Unicode NFC, audit
the actual tokenization statistics, and run LoRA adaptation on both the
text-semantic LM and DiT.

The pilot dataset is `Praha-Labs/TTS-Ml`. Its public `train` split contains
71,608 examples with Malayalam text and mixed source audio rates. The
preparation job uses streaming mode, resamples accepted audio to VoxCPM's
required 16 kHz rate, and applies a fixed seed and bounded shuffle buffer before
stopping at 20,000 accepted examples. It writes 19,500 training examples and
500 validation examples as FLAC plus VoxCPM JSONL manifests; it does not
download or unpack the complete dataset. The dataset revision is pinned for
reproducibility.

## Why “tokenizer-free” does not remove the text tokenizer

VoxCPM is tokenizer-free on the **audio** side: it models continuous AudioVAE
representations instead of discrete codec tokens. Text is still tokenized. The
VoxCPM2 implementation loads `LlamaTokenizerFast`, and the training script
converts every transcript to text token IDs.

The released VoxCPM2 tokenizer has byte fallback enabled, so Malayalam Unicode
can be represented without unknown tokens. It does not, however, contain
Malayalam-script vocabulary entries, so Malayalam text is likely to use more
tokens than a natively trained vocabulary.

For the initial model, do **not** extend the vocabulary. New vocabulary entries
would require new embedding rows, model resizing, checkpoint-format changes,
and training those rows; that is not a clean LoRA-only adaptation. First run:

```bash
python scripts/audit_malayalam_tokenizer.py \
  --pretrained-path /path/to/VoxCPM2 \
  --manifest /path/to/malayalam_train.jsonl
```

The audit fails if it sees unknown tokens or no Malayalam text and reports the
mean and p95 token-to-character ratios. Revisit vocabulary extension only if
the pilot shows unacceptable sequence inflation or poor grapheme-to-acoustic
learning despite sufficient clean data.

## Why this uses native Slurm instead of NeMo-Run

NVIDIA NeMo-Run's `SlurmExecutor` assumes Pyxis and a container image. FAU's
Alex documentation describes native Slurm jobs with environment modules/Conda
and supports containers through Apptainer; it does not document Pyxis. The
portable integration for Alex is therefore a native `sbatch` script around the
repository's existing `torchrun`/DDP support.

References:

- [FAU Alex cluster](https://doc.nhr.fau.de/clusters/alex/)
- [NeMo-Run SlurmExecutor](https://docs.nvidia.com/nemo/run/nightly/guides/executors/slurm.html)

## Compute-node-only cluster setup

The login node should only submit and inspect Slurm jobs. Repository cloning,
environment creation, package installation, model download, dataset download,
audio conversion, and training must run inside allocations.

First submit a small CPU-only job that clones `main`. Replace the paths with
the values used for this project:

```bash
CLONE_JOB=$(sbatch --parsable \
  --partition=a100mig --ntasks=1 --cpus-per-task=1 --time=00:30:00 \
  --job-name=voxcpm-clone --output="$WORK/voxcpm-clone-%j.out" \
  --wrap='set -euo pipefail; mkdir -p "$WORK/projects"; git clone --branch main --single-branch https://github.com/Pranavharshans/voxcpm-trainer.git "$WORK/projects/vox-trainer"')
echo "$CLONE_JOB"
```

After that job succeeds, submit the preparation job. It creates a persistent
Conda environment, downloads `openbmb/VoxCPM2`, streams the 20k dataset subset,
and writes `$WORK/voxcpm-runtime/alex.env` for the training job:

```bash
PREP_JOB=$(sbatch --parsable \
  --output="$WORK/voxcpm-prep-%j.out" \
  "$WORK/projects/vox-trainer/scripts/slurm/prepare_voxcpm_malayalam_alex.sbatch" \
  "$WORK/projects/vox-trainer" \
  "$WORK/voxcpm-runtime")
echo "$PREP_JOB"

TRAIN_JOB=$(sbatch --parsable \
  --dependency="afterok:$PREP_JOB" \
  --output="$WORK/voxcpm-train-%j.out" \
  "$WORK/projects/vox-trainer/scripts/slurm/train_voxcpm_malayalam_alex.sbatch" \
  "$WORK/voxcpm-runtime/alex.env")
echo "$TRAIN_JOB"
```

The manifests use the standard VoxCPM JSONL shape:

```json
{"audio": "/absolute/path/audio-0001.wav", "text": "ഇത് ഒരു മലയാളം പരിശീലന ഉദാഹരണമാണ്."}
```

Keep transcripts verbatim and consistently punctuated. Training applies NFC;
it does not transliterate Malayalam or expand Malayalam numbers/abbreviations.

## Submit the pilot

The training job above is submitted immediately but remains pending until the
preparation job succeeds. If preparation fails, Slurm will not start training.
To submit training separately instead:

```bash
TRAIN_JOB=$(sbatch --parsable \
  --output="$WORK/voxcpm-train-%j.out" \
  "$WORK/projects/vox-trainer/scripts/slurm/train_voxcpm_malayalam_alex.sbatch" \
  "$WORK/voxcpm-runtime/alex.env")
echo "$TRAIN_JOB"
```

The default requests one A100 GPU, 16 CPU cores, and Alex's maximum 24-hour
wall time. The job runs the tokenizer audit first, launches training through
`torchrun`, and writes `slurm-voxcpm-ml-<job-id>.out` in the submission
directory.

The default distributed mode is DDP, which replicates the complete training
state on every GPU. Increasing the GPU count speeds supported jobs up but does
not make a full-SFT model fit into smaller GPUs. Use the FSDP launcher described
in the full-parameter SFT section for A100 40 GB training.

For a DDP-compatible workload such as LoRA, request four A100 GPUs by overriding
both GPU and CPU allocations:

```bash
sbatch --gres=gpu:a100:4 --cpus-per-task=64 \
  "$WORK/projects/vox-trainer/scripts/slurm/train_voxcpm_malayalam_alex.sbatch" \
  "$WORK/voxcpm-runtime/alex.env"
```

The effective global batch is `batch_size × grad_accum_steps × GPU count`.
Configurations with `global_batch_size` resolve gradient accumulation
automatically and reject GPU counts that cannot preserve the requested batch.

Monitor with:

```bash
squeue --me
sacct -j <job-id>
tail -f slurm-voxcpm-ml-<job-id>.out
```

Five minutes before the wall-time limit, Slurm signals the launcher. The
training process saves `VOXCPM_RUN_DIR/checkpoints/latest`. Submitting the same
job again resumes from that checkpoint automatically.

The template intentionally supports one node. Alex multi-node jobs require
project-level enablement from NHR@FAU support and a separate rendezvous-aware
launch configuration.

## Full dataset, three-epoch run

After the pilot, use the `full` preparation profile for a fresh run from the
released VoxCPM2 base model. It consumes the complete pinned `Praha-Labs/TTS-Ml`
split, retains 500 fixed validation rows, resamples valid audio to 16 kHz, and
writes a separate `alex-full.env`. The pilot data, logs, and checkpoints are
not overwritten.

The full configuration derives its optimizer-step count from the number of
training rows that remain after audio/text validation and length filtering.
It runs three passes with an effective batch of 16, a 500-step warmup, and a
lower peak learning rate of `5e-5`. Validation and checkpointing occur at each
epoch boundary. Four fixed held-out samples are generated at every boundary
and saved both to TensorBoard and as ordinary WAV files. Those samples use
`assets/eval/Maya.wav` as the fixed voice prompt with its supplied English
transcript, while the synthesis targets remain held-out Malayalam sentences.

From the login node, update the NVMe repository and only submit/inspect jobs:

```bash
VOX_STORAGE=$(ws_find voxcpm-ml)
git -C "$VOX_STORAGE/vox-trainer" pull --ff-only origin main

PREP_JOB=$(sbatch --parsable \
  --output="$VOX_STORAGE/voxcpm-full-prep-%j.out" \
  "$VOX_STORAGE/vox-trainer/scripts/slurm/prepare_voxcpm_malayalam_alex.sbatch" \
  "$VOX_STORAGE/vox-trainer" \
  "$VOX_STORAGE/voxcpm-runtime" \
  full)

TRAIN_JOB=$(sbatch --parsable \
  --partition=a100 \
  --gres=gpu:a100:1 \
  --constraint=a100_80 \
  --cpus-per-task=16 \
  --dependency="afterok:$PREP_JOB" \
  --output="$VOX_STORAGE/voxcpm-full-train-%j.out" \
  "$VOX_STORAGE/vox-trainer/scripts/slurm/train_voxcpm_malayalam_alex.sbatch" \
  "$VOX_STORAGE/voxcpm-runtime/alex-full.env")

echo "Preparation: $PREP_JOB"
echo "Training: $TRAIN_JOB"
```

Monitor both jobs without running preparation or training on the login node:

```bash
watch -n 10 "squeue -j $PREP_JOB,$TRAIN_JOB -o '%.18i %.12P %.18j %.2t %.10M %.10l %R'"
tail --retry -F \
  "$VOX_STORAGE/voxcpm-full-prep-$PREP_JOB.out" \
  "$VOX_STORAGE/voxcpm-full-train-$TRAIN_JOB.out"
```

The outputs are kept under:

```text
voxcpm-runtime/runs/malayalam-full-3epoch/
├── checkpoints/epoch_01/
├── checkpoints/epoch_02/
├── checkpoints/epoch_03/
├── eval_samples/epoch_01/
├── eval_samples/epoch_02/
├── eval_samples/epoch_03/
└── tensorboard/
```

Each evaluation directory contains the generated WAV, held-out ground-truth
WAV, Malayalam target transcript, `voice_prompt.wav`, and
`voice_prompt_text.txt`. The same Maya prompt and target sentences are used at
all three epoch boundaries, making the audio comparisons consistent.

## Full-parameter SFT follow-up

Use the `sft` profile only after reviewing the LoRA baseline. It reuses the
existing full Malayalam dataset but starts from the original VoxCPM2 base
checkpoint, not from the LoRA adapter. The AudioVAE stays frozen; all other
model parameters are updated for exactly two epochs at a conservative `1e-5`
learning rate. This clean separation makes LoRA and SFT results comparable.

The SFT run writes to `runs/malayalam-full-sft-2epoch`, so it cannot resume
from or overwrite the LoRA checkpoints. It produces `epoch_01` and `epoch_02`
full-model checkpoints and the same four Maya-conditioned evaluation samples
at both boundaries.

After the LoRA job completes, pull `main` and submit the lightweight setup job:

```bash
VOX_STORAGE=$(ws_find voxcpm-ml)
REPO="$VOX_STORAGE/vox-trainer"
RUNTIME="$VOX_STORAGE/voxcpm-runtime"

git -C "$REPO" pull --ff-only origin main

SFT_PREP_JOB=$(sbatch --parsable \
  --output="$VOX_STORAGE/voxcpm-sft-prep-%j.out" \
  "$REPO/scripts/slurm/prepare_voxcpm_malayalam_alex.sbatch" \
  "$REPO" \
  "$RUNTIME" \
  sft)
```

Full SFT exceeded the memory of one A100 80 GB on a long sample. The SFT
configuration therefore enables native PyTorch FSDP `FULL_SHARD`, transformer
activation checkpointing, a 6,144-token batch cap, and an effective global
batch of 16. On four GPUs, accumulation resolves automatically from 16 to 4.
Periodic recovery checkpoints are written every 500 optimizer steps; only the
newest two periodic checkpoints are retained, while epoch checkpoints remain.

Create a clean production environment. The production job saves a full
checkpoint after its first optimizer step, so the first few minutes verify
FSDP forward/backward, optimizer stepping, and checkpoint consolidation
without requiring a second allocation for a separate smoke job:

```bash
cp "$RUNTIME/alex-sft.env" "$RUNTIME/alex-sft-fsdp.env"
sed -i \
  "s|^VOXCPM_RUN_DIR=.*|VOXCPM_RUN_DIR=$RUNTIME/runs/malayalam-full-sft-2epoch-fsdp|" \
  "$RUNTIME/alex-sft-fsdp.env"

SFT_JOB=$(sbatch --parsable \
  --partition=a100,a40,rtxpro6k \
  --gres=gpu:4 \
  --cpus-per-task=64 \
  --time=15:00:00 \
  --output="$VOX_STORAGE/voxcpm-sft-fsdp-%j.out" \
  "$REPO/scripts/slurm/train_voxcpm_malayalam_alex.sbatch" \
  "$RUNTIME/alex-sft-fsdp.env")

echo "SFT preparation: $SFT_PREP_JOB"
echo "SFT training: $SFT_JOB"
```

The untyped four-GPU request lets Slurm choose A100, A40, or RTX PRO 6000.
The job retains the same global batch of 16 on every supported GPU type.
After distributed training exits, the same allocation loads `epoch_01` and
`epoch_02` one at a time on GPU 0 and generates the Maya-conditioned WAV and
TensorBoard artifacts. This avoids unsupported custom generation inside an
FSDP forward context and does not require another queued job.

Full SFT uses substantially more VRAM and writes much larger checkpoints than
LoRA. Confirm GPU memory after startup with:

```bash
srun --jobid="$SFT_JOB" --overlap --ntasks=1 --cpus-per-task=1 \
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
  --format=csv
```

The generated SFT environment enables `VOXCPM_STAGE_MODEL_TO_TMPDIR=1`.
At job startup, the launcher sequentially copies the approximately 5 GB base
model from shared `/anvme` storage into the allocation's node-local `$TMPDIR`.
This avoids prolonged safetensors mmap page faults on a congested shared
filesystem. The staged copy is temporary and is deleted automatically when the
Slurm job ends; the persistent source model is unchanged.

SFT outputs are kept under:

```text
voxcpm-runtime/runs/malayalam-full-sft-2epoch-fsdp/
├── checkpoints/epoch_01/
├── checkpoints/epoch_02/
├── eval_samples/epoch_01/
├── eval_samples/epoch_02/
└── tensorboard/
```
