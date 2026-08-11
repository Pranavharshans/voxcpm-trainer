#!/bin/bash -l
#SBATCH --job-name=voxcpm-prep
#SBATCH --partition=a100mig
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --export=NONE
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
unset SLURM_EXPORT_ENV

REPO_DIR=${1:?"usage: sbatch prepare_voxcpm_malayalam_alex.sbatch REPO_DIR RUNTIME_ROOT"}
RUNTIME_ROOT=${2:?"usage: sbatch prepare_voxcpm_malayalam_alex.sbatch REPO_DIR RUNTIME_ROOT"}

REPO_DIR=$(realpath "${REPO_DIR}")
mkdir -p "${RUNTIME_ROOT}"
RUNTIME_ROOT=$(realpath "${RUNTIME_ROOT}")

CONDA_ENV=${RUNTIME_ROOT}/envs/voxcpm
MODEL_DIR=${RUNTIME_ROOT}/models/VoxCPM2
DATA_DIR=${RUNTIME_ROOT}/datasets/tts-ml-20k
RUN_DIR=${RUNTIME_ROOT}/runs/malayalam-pilot
JOB_ENV=${RUNTIME_ROOT}/alex.env

module load python

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
    --total-samples 20000 \
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
VOXCPM_TRAIN_CONFIG=conf/voxcpm_v2/voxcpm_finetune_malayalam_lora.yaml
HF_HOME=${RUNTIME_ROOT}/huggingface-cache
EOF

echo "Preparation complete. Training environment: ${JOB_ENV}"
