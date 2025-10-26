#!/bin/bash
#SBATCH --job-name=bm_gw_96ep
#SBATCH --time=96:00:00
#SBATCH --qos=bbdefault
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=256G
#SBATCH --output=logs/train_gwilliam_96_%j.log
#SBATCH --output=/rds/projects/p/parkh-speech-decode1/brainmagick_default/logs/train_gwilliam_96_%j.out
#SBATCH --error=/rds/projects/p/parkh-speech-decode1/brainmagick_default/logs/train_gwilliam_96_%j.err

set -euo pipefail

module purge
module load bluebear
module load bear-apps/2022b
module load Miniforge3/24.1.2-0

# --- activate EXACT env path (work around MKL + nounset) ---
eval "$(${EBROOTMINIFORGE3}/bin/conda shell.bash hook)"
source "${EBROOTMINIFORGE3}/etc/profile.d/mamba.sh"

# Optional: define MKL interface to avoid scripts touching an undefined var
export MKL_INTERFACE_LAYER=${MKL_INTERFACE_LAYER:-LP64}

set +u
mamba activate /rds/projects/p/parkh-speech-decode1
set -u

# --- runtime env (CPU) ---
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
ulimit -n 16384

cd /rds/projects/p/parkh-speech-decode1/brainmagick_default
mkdir -p logs

# quick sanity
python - <<'PY'
import sys, torch, torchaudio
print("Python:", sys.executable)
print("Torch:", torch.__version__, "CUDA?", torch.cuda.is_available())
print("Torchaudio:", getattr(torchaudio, "__version__", "unknown"))
PY

# Go to repo
cd /rds/projects/p/parkh-speech-decode1/brainmagick_default

# Launch 
dora run \
  --clear \
  'dset.selections=[gwilliams2022]' \
  device=cpu \
  num_workers=0 \
  > logs/train_cpu_96h.log 2>&1
