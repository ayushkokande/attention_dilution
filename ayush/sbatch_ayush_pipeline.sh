#!/bin/bash
#SBATCH --job-name=ayush_pipeline
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
# Log paths must be absolute: ./logs depends on cwd when you run sbatch.
# Keep prefix in sync with REPO below.
#SBATCH --output=/scratch/ak13124/attention_dilution/logs/ayush_pipeline_%j.out
#SBATCH --error=/scratch/ak13124/attention_dilution/logs/ayush_pipeline_%j.err

# Attention-dilution / refusal-mech pipeline: Qwen3-1.7B baseline refusal rate,
# refusal-direction extraction, ablate / add interventions, and circuit-tracer
# attribution graphs. Runs the scripts in ayush/ end-to-end on one A100.
#
# Edit SCRATCH / SIF / OVERLAY / REPO and the two #SBATCH log lines above.
# hf: huggingface-cli login in this env if any model / dataset is gated.
#
# This script lives in ayush/ and always runs from ${REPO} (see the cd below),
# so you can submit it from anywhere, e.g. from the repo root:
#   sbatch ayush/sbatch_ayush_pipeline.sh
# or from inside ayush/:
#   sbatch sbatch_ayush_pipeline.sh

set -euo pipefail

# --- edit these ---
SCRATCH="/scratch/ak13124"
SIF="${SCRATCH}/ubuntu-20.04.3.sif"
OVERLAY="${SCRATCH}/overlay-25GB-500K.ext3:ro"
REPO="${SCRATCH}/attention_dilution"
# Which stages to run: set to 0 to skip.
RUN_SANITY=1
RUN_EXTRACT=1
RUN_INTERVENE=1
RUN_ABLATE_LONG=1
RUN_VISUALIZE_DIR=1
RUN_VISUALIZE_STEER=1
RUN_TRACE=1
# Feature-level intervention requires a hand-curated
# results/<slug>/refusal_features.json, so it is off by default.
RUN_FEATURE_INTERVENE=0
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

# Accept Anaconda channel ToS non-interactively so conda does not abort with
# CondaToSNonInteractiveError under sbatch. Idempotent; once-per-overlay.
# Swallow failures so a missing plugin or already-accepted state is harmless.
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    >/dev/null 2>&1 || true

# uv provisions its own Python interpreters; keep its downloads on scratch so
# we don't blow through \$HOME and so subsequent jobs reuse the cache.
export UV_PYTHON_INSTALL_DIR=${SCRATCH}/.uv_python

# Keep model / dataset downloads on scratch so we don't blow through \$HOME.
export HF_HOME=${SCRATCH}/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=\${HF_HOME}/hub
export TRANSFORMERS_CACHE=\${HF_HOME}/hub
export TORCH_HOME=${SCRATCH}/.cache/torch
mkdir -p \"\${HF_HOME}\" \"\${TORCH_HOME}\"

cd \"${REPO}\"

echo \"=== preflight | REPO=\$(pwd) ===\"
ls -la

# Preflight: the pipeline needs the full repo laid out under \${REPO}, not
# just the ayush/ subdir. Fail fast with a clear message before we do the
# slow uv venv + dependency install.
missing=0
for f in requirements.txt ayush/utils.py ayush/sanity_check.py; do
  if [ ! -f \"\$f\" ]; then
    echo \"ERROR: missing \$(pwd)/\$f\" >&2
    missing=1
  fi
done
if [ \"\$missing\" = \"1\" ]; then
  echo >&2
  echo \"Sync the full repo to \${REPO} first, e.g. from your laptop:\" >&2
  echo \"  rsync -avz --exclude .venv --exclude __pycache__ --exclude results/ \\\\\" >&2
  echo \"    ./ greene:${REPO}/\" >&2
  exit 1
fi

# Create / reuse a venv backed by a uv-managed Python 3.11 (the overlay's
# miniconda does not ship python3.11). uv will download 3.11 into
# \${UV_PYTHON_INSTALL_DIR} on first run and reuse it afterwards.
if [ ! -d .venv ]; then
  uv venv --python 3.11 .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install --python .venv/bin/python -r requirements.txt

echo \"=== ayush_pipeline | job \${SLURM_JOB_ID:-local} ===\"
echo \"Repo: \$(pwd)\"
echo \"CUDA: \${CUDA_VISIBLE_DEVICES:-unset}\"
python -c 'import torch; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"\")'

run_step() {
  local name=\"\$1\"; shift
  echo
  echo \"----- [\${name}] \$(date -u +%FT%TZ) -----\"
  time \"\$@\"
}

if [ \"${RUN_SANITY}\" = \"1\" ]; then
  run_step sanity_check           python ayush/sanity_check.py
fi
if [ \"${RUN_EXTRACT}\" = \"1\" ]; then
  run_step extract_direction      python ayush/extract_refusal_direction.py
fi
if [ \"${RUN_INTERVENE}\" = \"1\" ]; then
  run_step intervene_refusal      python ayush/intervene_refusal.py
fi
if [ \"${RUN_ABLATE_LONG}\" = \"1\" ]; then
  run_step intervene_ablate_long  python ayush/intervene_ablate_long.py
fi
if [ \"${RUN_VISUALIZE_DIR}\" = \"1\" ]; then
  run_step visualize_direction    python ayush/visualize_refusal_direction.py
fi
if [ \"${RUN_TRACE}\" = \"1\" ]; then
  run_step trace_refusal_circuit  python ayush/trace_refusal_circuit.py
fi
if [ \"${RUN_FEATURE_INTERVENE}\" = \"1\" ]; then
  run_step intervene_features     python ayush/intervene_refusal_features.py
fi
if [ \"${RUN_VISUALIZE_STEER}\" = \"1\" ]; then
  # Run last so it can pick up the feature-ablation results if present.
  run_step visualize_steering     python ayush/visualize_steering.py
fi

echo
echo \"=== done | results under: \${PWD}/results/ ===\"
ls -la results/ || true
"
