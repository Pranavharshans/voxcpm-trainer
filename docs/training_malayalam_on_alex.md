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

To request four A100 GPUs on one node, override both GPU and CPU allocations:

```bash
sbatch --gres=gpu:a100:4 --cpus-per-task=64 \
  "$WORK/projects/vox-trainer/scripts/slurm/train_voxcpm_malayalam_alex.sbatch" \
  "$WORK/voxcpm-runtime/alex.env"
```

The effective global batch is `batch_size × grad_accum_steps × GPU count`.
Adjust `grad_accum_steps` if changing GPU count so optimization behavior stays
comparable.

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
