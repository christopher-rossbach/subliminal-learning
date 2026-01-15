#!/bin/bash -xl
#
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=24:00:00
#SBATCH --output=./trigger/slurm/logs/train_%x.%j.log

unset SLURM_EXPORT_ENV

export https_proxy=http://proxy:80
export http_proxy=http://proxy:80

module load python/3.12-conda
module load cuda/12.6

source "$SLURM_SUBMIT_DIR/.venv/bin/activate"

# Run training script with all arguments passed through
python3 "$SLURM_SUBMIT_DIR/trigger/train_teacher.py" "${@:1}"
