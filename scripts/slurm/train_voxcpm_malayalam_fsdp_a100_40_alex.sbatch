#!/bin/bash -l
#SBATCH --job-name=voxcpm-fsdp
#SBATCH --partition=a100
#SBATCH --constraint=a100_40
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --signal=B:TERM@300
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

JOB_ENV=${1:-conf/slurm/alex.env}
if [[ "${JOB_ENV}" != /* ]]; then
    JOB_ENV="${SLURM_SUBMIT_DIR}/${JOB_ENV}"
fi
if [[ ! -f "${JOB_ENV}" ]]; then
    echo "Missing job environment file: ${JOB_ENV}" >&2
    exit 2
fi

# Slurm copies this wrapper into its private spool directory before executing
# it, so BASH_SOURCE[0] no longer points into the repository. Resolve the real
# launcher through the repository path recorded in the submitted job env.
set -a
source "${JOB_ENV}"
set +a
if [[ -z "${VOXCPM_REPO_DIR:-}" ]]; then
    echo "Required variable VOXCPM_REPO_DIR is missing from ${JOB_ENV}" >&2
    exit 2
fi

LAUNCHER="${VOXCPM_REPO_DIR}/scripts/slurm/train_voxcpm_malayalam_alex.sbatch"
if [[ ! -f "${LAUNCHER}" ]]; then
    echo "Missing training launcher: ${LAUNCHER}" >&2
    exit 2
fi

exec "${LAUNCHER}" "${JOB_ENV}"
