#!/bin/bash
#SBATCH --job-name=bm_gw_96ep_gpu
#SBATCH --time=6:00:00
#SBATCH --qos=bbgpu
#SBATCH --account=parkh-speech-decode1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=256G
#SBATCH --output=/rds/projects/p/parkh-speech-decode1/brainmagick_default/logs/train_gwilliam_gpu_96_%j.out
#SBATCH --error=/rds/projects/p/parkh-speech-decode1/brainmagick_default/logs/train_gwilliam_gpu_96_%j.err

set -euo pipefail

module purge
module load bluebear
module load bear-apps/2022b
module load Miniforge3/24.1.2-0
# module load PyTorch/1.7.1-fosscuda-2020b

# --- activate EXACT env path (work around MKL + nounset) ---
eval "$(${EBROOTMINIFORGE3}/bin/conda shell.bash hook)"
source "${EBROOTMINIFORGE3}/etc/profile.d/mamba.sh"

# Optional: define MKL interface to avoid scripts touching an undefined var
export MKL_INTERFACE_LAYER=${MKL_INTERFACE_LAYER:-LP64}

set +u
mamba activate /rds/projects/p/parkh-speech-decode1
set -u

# Ensure CUDA-enabled PyTorch is installed in the env (A100 nodes).
# A100 GPUs support CUDA 11.1+, so pin to 11.1 for the PyTorch 1.7.1 build.
mamba install -y -c pytorch -c nvidia pytorch pytorch-cuda=11.8 
mamba install -y -c pytorch -c conda-forge pytorch=1.7.1 torchaudio=0.7.2 cudatoolkit=11.0

# sanity: confirm GPU is visible
python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

# --- runtime env (GPU) ---
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
print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY

# Launch
dora run \
  --clear \
  'dset.selections=[gwilliams2022]' \
  device=cuda \
  num_workers=4 \
  > logs/train_gpu_96h_%j.log 2>&1
