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

# Loading the multi-gigabyte safetensors file through mmap can be extremely
# slow when the shared /anvme filesystem is congested. Full SFT profiles can
# opt into one sequential copy to node-local storage, after which model loading
# and page faults are served from the compute node's NVMe SSD.
if [[ "${VOXCPM_STAGE_MODEL_TO_TMPDIR:-0}" == "1" ]]; then
    STAGED_MODEL_DIR=${TMPDIR}/VoxCPM2
    mkdir -p "${STAGED_MODEL_DIR}"
    echo "Staging VoxCPM2 model to node-local storage: ${STAGED_MODEL_DIR}"
    rsync -a --info=progress2 "${VOXCPM_PRETRAINED_PATH}/" "${STAGED_MODEL_DIR}/"
    VOXCPM_PRETRAINED_PATH=${STAGED_MODEL_DIR}
    export VOXCPM_PRETRAINED_PATH
    echo "Node-local model staging complete."
fi

cd "${VOXCPM_REPO_DIR}"
mkdir -p "${VOXCPM_RUN_DIR}"

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:-${VOXCPM_RUN_DIR}/huggingface-cache}
# Reduce CUDA allocator fragmentation for long, variable-duration TTS batches.
# Preserve an explicit value supplied by the job environment.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

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

TRAIN_CONFIG=${VOXCPM_TRAIN_CONFIG:-conf/voxcpm_v2/voxcpm_finetune_malayalam_lora.yaml}
ACTIVE_CHILD_PID=""

forward_term() {
    echo "Slurm time-limit signal received; asking the active process to checkpoint and stop." >&2
    if [[ -n "${ACTIVE_CHILD_PID}" ]]; then
        kill -TERM "${ACTIVE_CHILD_PID}" 2>/dev/null || true
        wait "${ACTIVE_CHILD_PID}" || true
    fi
}
trap forward_term TERM INT

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="${NPROC_PER_NODE}" \
    scripts/train_voxcpm_finetune.py \
    --config_path "${TRAIN_CONFIG}" &
ACTIVE_CHILD_PID=$!

wait "${ACTIVE_CHILD_PID}"
ACTIVE_CHILD_PID=""

# FSDP cannot safely execute VoxCPM's custom generate() path while sharded.
# The evaluator is a no-op for non-FSDP/smoke profiles. For production FSDP,
# it loads each portable epoch checkpoint on GPU 0 after torchrun has released
# all sharded model state, keeping training and evaluation in one allocation.
python scripts/evaluate_voxcpm_epoch_checkpoints.py \
    --config-path "${TRAIN_CONFIG}" &
ACTIVE_CHILD_PID=$!

wait "${ACTIVE_CHILD_PID}"
