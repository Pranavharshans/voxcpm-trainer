#!/bin/bash -l
#SBATCH --job-name=voxcpm-ml
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --signal=B:TERM@300
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
unset SLURM_EXPORT_ENV

JOB_ENV=${1:-conf/slurm/alex.env}
if [[ "${JOB_ENV}" != /* ]]; then
    JOB_ENV="${SLURM_SUBMIT_DIR}/${JOB_ENV}"
fi
if [[ ! -f "${JOB_ENV}" ]]; then
    echo "Missing job environment file: ${JOB_ENV}" >&2
    echo "Copy conf/slurm/alex.env.example to conf/slurm/alex.env and fill in the paths." >&2
    exit 2
fi

set -a
source "${JOB_ENV}"
set +a

require_var() {
    local name=$1
    if [[ -z "${!name:-}" ]]; then
        echo "Required variable ${name} is missing from ${JOB_ENV}" >&2
        exit 2
    fi
}

require_var VOXCPM_REPO_DIR
require_var VOXCPM_CONDA_ENV
require_var VOXCPM_PRETRAINED_PATH
require_var MALAYALAM_TRAIN_MANIFEST
require_var MALAYALAM_VAL_MANIFEST
require_var VOXCPM_RUN_DIR

module load "${VOXCPM_PYTHON_MODULE:-python}"
conda activate "${VOXCPM_CONDA_ENV}"

cd "${VOXCPM_REPO_DIR}"
mkdir -p "${VOXCPM_RUN_DIR}"

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:-${VOXCPM_RUN_DIR}/huggingface-cache}

NPROC_PER_NODE=${SLURM_GPUS_ON_NODE:-1}
if [[ ! "${NPROC_PER_NODE}" =~ ^[0-9]+$ ]]; then
    echo "SLURM_GPUS_ON_NODE must be an integer, got: ${NPROC_PER_NODE}" >&2
    exit 2
fi

echo "job_id=${SLURM_JOB_ID} host=$(hostname) gpus=${NPROC_PER_NODE}"
nvidia-smi

python scripts/audit_malayalam_tokenizer.py \
    --pretrained-path "${VOXCPM_PRETRAINED_PATH}" \
    --manifest "${MALAYALAM_TRAIN_MANIFEST}"

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="${NPROC_PER_NODE}" \
    scripts/train_voxcpm_finetune.py \
    --config_path "${VOXCPM_TRAIN_CONFIG:-conf/voxcpm_v2/voxcpm_finetune_malayalam_lora.yaml}" &
TRAIN_PID=$!

forward_term() {
    echo "Slurm time-limit signal received; asking torchrun to checkpoint and stop." >&2
    kill -TERM "${TRAIN_PID}" 2>/dev/null || true
    wait "${TRAIN_PID}" || true
}
trap forward_term TERM INT

wait "${TRAIN_PID}"
