#!/bin/bash -l
#SBATCH --job-name=voxcpm-prep
#SBATCH --partition=a100mig
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --export=NONE
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
unset SLURM_EXPORT_ENV

REPO_DIR=${1:?"usage: sbatch prepare_voxcpm_malayalam_alex.sbatch REPO_DIR RUNTIME_ROOT [pilot|full|sft]"}
RUNTIME_ROOT=${2:?"usage: sbatch prepare_voxcpm_malayalam_alex.sbatch REPO_DIR RUNTIME_ROOT [pilot|full|sft]"}
PROFILE=${3:-pilot}

REPO_DIR=$(realpath "${REPO_DIR}")
mkdir -p "${RUNTIME_ROOT}"
RUNTIME_ROOT=$(realpath "${RUNTIME_ROOT}")

CONDA_ENV=${RUNTIME_ROOT}/envs/voxcpm
MODEL_DIR=${RUNTIME_ROOT}/models/VoxCPM2
case "${PROFILE}" in
    pilot)
        DATA_DIR=${RUNTIME_ROOT}/datasets/tts-ml-20k
        RUN_DIR=${RUNTIME_ROOT}/runs/malayalam-pilot
        JOB_ENV=${RUNTIME_ROOT}/alex.env
        TRAIN_CONFIG=conf/voxcpm_v2/voxcpm_finetune_malayalam_lora.yaml
        DATASET_ARGS=(--total-samples 20000)
        STAGE_MODEL_TO_TMPDIR=0
        ;;
    full)
        DATA_DIR=${RUNTIME_ROOT}/datasets/tts-ml-full
        RUN_DIR=${RUNTIME_ROOT}/runs/malayalam-full-3epoch
        JOB_ENV=${RUNTIME_ROOT}/alex-full.env
        TRAIN_CONFIG=conf/voxcpm_v2/voxcpm_finetune_malayalam_full_3epoch_lora.yaml
        DATASET_ARGS=(--all-samples)
        STAGE_MODEL_TO_TMPDIR=0
        ;;
    sft)
        DATA_DIR=${RUNTIME_ROOT}/datasets/tts-ml-full
        RUN_DIR=${RUNTIME_ROOT}/runs/malayalam-full-sft-2epoch
        JOB_ENV=${RUNTIME_ROOT}/alex-sft.env
        TRAIN_CONFIG=conf/voxcpm_v2/voxcpm_finetune_malayalam_full_sft_2epoch.yaml
        DATASET_ARGS=(--all-samples)
        STAGE_MODEL_TO_TMPDIR=1
        ;;
    *)
        echo "Unknown profile '${PROFILE}'; expected pilot, full, or sft" >&2
        exit 2
        ;;
esac

module load python

# Alex compute nodes require the cluster proxy for external package/model/data
# downloads. The Python module's default Conda package cache is under /apps and
# is read-only, so force it into this project's writable runtime directory.
ALEX_PROXY_URL=${ALEX_PROXY_URL:-http://proxy.nhr.fau.de:80}
export http_proxy=${http_proxy:-${ALEX_PROXY_URL}}
export https_proxy=${https_proxy:-${ALEX_PROXY_URL}}
export HTTP_PROXY=${HTTP_PROXY:-${ALEX_PROXY_URL}}
export HTTPS_PROXY=${HTTPS_PROXY:-${ALEX_PROXY_URL}}
export CONDA_PKGS_DIRS=${RUNTIME_ROOT}/cache/conda-pkgs
mkdir -p "${CONDA_PKGS_DIRS}"

if [[ ! -x "${CONDA_ENV}/bin/python" ]]; then
    mkdir -p "$(dirname "${CONDA_ENV}")"
    conda create --prefix "${CONDA_ENV}" python=3.11 -y
fi
conda activate "${CONDA_ENV}"

export PIP_CACHE_DIR=${TMPDIR}/pip-cache
export HF_HOME=${TMPDIR}/huggingface-cache
export HF_DATASETS_CACHE=${TMPDIR}/huggingface-datasets
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

python -m pip install --upgrade pip
python -m pip install -e "${REPO_DIR}"

mkdir -p "${MODEL_DIR}" "${DATA_DIR}" "${RUN_DIR}"
if [[ ! -f "${MODEL_DIR}/config.json" || ! -f "${MODEL_DIR}/model.safetensors" ]]; then
    hf download openbmb/VoxCPM2 --local-dir "${MODEL_DIR}"
fi

cd "${REPO_DIR}"
python scripts/prepare_tts_ml_subset.py \
    --output-dir "${DATA_DIR}" \
    "${DATASET_ARGS[@]}" \
    --validation-samples 500 \
    --seed 42 \
    --shuffle-buffer 1000 \
    --cache-dir "${HF_DATASETS_CACHE}"

cat > "${JOB_ENV}" <<EOF
VOXCPM_REPO_DIR=${REPO_DIR}
VOXCPM_CONDA_ENV=${CONDA_ENV}
VOXCPM_PYTHON_MODULE=python
VOXCPM_PRETRAINED_PATH=${MODEL_DIR}
MALAYALAM_TRAIN_MANIFEST=${DATA_DIR}/train.jsonl
MALAYALAM_VAL_MANIFEST=${DATA_DIR}/val.jsonl
VOXCPM_RUN_DIR=${RUN_DIR}
VOXCPM_TRAIN_CONFIG=${TRAIN_CONFIG}
VOXCPM_STAGE_MODEL_TO_TMPDIR=${STAGE_MODEL_TO_TMPDIR}
HF_HOME=${RUNTIME_ROOT}/huggingface-cache
EOF

echo "Preparation profile '${PROFILE}' complete. Training environment: ${JOB_ENV}"
