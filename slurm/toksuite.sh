#!/usr/bin/env bash
#SBATCH --gpus-per-node=T4:1
#SBATCH --nodes=1
#SBATCH -t 0-23:00:0
#SBATCH --output=logs/toksuite/log-%j.out
#SBATCH -A NAISS2025-22-601
set -euo pipefail

module purge
if [ -z "${MIMER_DIR}" ]; then
  echo "$(date) Error: env MIMER_DIR is not defined" >&2
  exit 1
fi

echo "$(date) loading modules"
module load "CUDA/12.6.0"
module load "PyTorch/2.7.1-foss-2024a-CUDA-12.6.0"

echo "$(date) activating venv"
source "$MIMER_DIR/.venv/tokefx/bin/activate"

echo "$(date) starting evaluation for toksuite models"
cd "$MIMER_DIR/tokefx-dev/" || exit
./scripts/run --in_boundary_mode all --ablation_map_type attention_mass configs/toksuite_config.toml
echo "$(date) finished evaluation for toksuite models"
