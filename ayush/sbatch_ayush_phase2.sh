#!/bin/bash
#SBATCH --job-name=ayush_phase2
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
# Log paths must be absolute: ./logs depends on cwd when you run sbatch.
# Keep prefix in sync with REPO below.
#SBATCH --output=/scratch/ak13124/attention_dilution/logs/ayush_phase2_%j.out
#SBATCH --error=/scratch/ak13124/attention_dilution/logs/ayush_phase2_%j.err

# Phase 2 only: context-length scaling sweep + its plots.
# Assumes Phase 1 has already produced results/<slug>/refusal_directions.pt
# (run sbatch_ayush_pipeline.sh at least once, or unskip just the extract
# stage in that file).
#
# This script reuses the venv created by sbatch_ayush_pipeline.sh if present;
# otherwise it creates it the same way.
#
# Submit from anywhere:
#   sbatch /scratch/ak13124/attention_dilution/ayush/sbatch_ayush_phase2.sh

set -euo pipefail

# --- edit these ---
SCRATCH="/scratch/ak13124"
SIF="${SCRATCH}/ubuntu-20.04.3.sif"
OVERLAY="${SCRATCH}/overlay-25GB-500K.ext3:ro"
REPO="${SCRATCH}/attention_dilution"
# Stage toggles for Phase 2 only.
RUN_SCALING_SWEEP=1
RUN_VISUALIZE_SCALING=1
# ------------------

mkdir -p "${REPO}/logs"

singularity exec --bind "${SCRATCH}" --nv \
  --overlay "${OVERLAY}" \
  "${SIF}" \
  /bin/bash -c "
set -euo pipefail

source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:\$PATH
export PATH=${SCRATCH}/tools/bin:\$PATH
export UV_CACHE_DIR=${SCRATCH}/.uv_cache

# Anaconda ToS accept (idempotent; needed once per overlay, harmless afterwards).
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    >/dev/null 2>&1 || true

# uv-managed Python interpreters on scratch so every job reuses them.
export UV_PYTHON_INSTALL_DIR=${SCRATCH}/.uv_python

# Keep model / dataset downloads on scratch so we don't blow through \$HOME.
export HF_HOME=${SCRATCH}/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=\${HF_HOME}/hub
export TRANSFORMERS_CACHE=\${HF_HOME}/hub
export TORCH_HOME=${SCRATCH}/.cache/torch
mkdir -p \"\${HF_HOME}\" \"\${TORCH_HOME}\"

cd \"${REPO}\"

echo \"=== preflight (phase 2) | REPO=\$(pwd) ===\"
ls -la

# Need Phase 1 output and the two Phase 2 scripts on disk.
missing=0
for f in requirements.txt \
         ayush/utils.py \
         ayush/sweep_context_scaling.py \
         ayush/visualize_scaling.py \
         results/qwen3-1.7b/refusal_directions.pt; do
  if [ ! -e \"\$f\" ]; then
    echo \"ERROR: missing \$(pwd)/\$f\" >&2
    missing=1
  fi
done
if [ \"\$missing\" = \"1\" ]; then
  echo >&2
  echo \"Phase 2 needs Phase 1 artefacts. If refusal_directions.pt is missing,\" >&2
  echo \"run sbatch_ayush_pipeline.sh first (or at least its extract stage).\" >&2
  exit 1
fi

# Reuse the venv from Phase 1 if present; create it the same way otherwise.
if [ ! -d .venv ]; then
  uv venv --python 3.11 .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install --python .venv/bin/python -r requirements.txt

echo \"=== ayush_phase2 | job \${SLURM_JOB_ID:-local} ===\"
echo \"Repo: \$(pwd)\"
echo \"CUDA: \${CUDA_VISIBLE_DEVICES:-unset}\"
python -c 'import torch; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"\")'

run_step() {
  local name=\"\$1\"; shift
  echo
  echo \"----- [\${name}] \$(date -u +%FT%TZ) -----\"
  time \"\$@\"
}

if [ \"${RUN_SCALING_SWEEP}\" = \"1\" ]; then
  run_step scaling_sweep      python ayush/sweep_context_scaling.py
fi
if [ \"${RUN_VISUALIZE_SCALING}\" = \"1\" ]; then
  run_step visualize_scaling  python ayush/visualize_scaling.py
fi

echo
echo \"=== done | scaling artefacts under: \${PWD}/results/qwen3-1.7b/scaling/ ===\"
ls -la results/qwen3-1.7b/scaling/ || true
"
