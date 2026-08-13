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

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${SCRIPT_DIR}/train_voxcpm_malayalam_alex.sbatch" "$@"
