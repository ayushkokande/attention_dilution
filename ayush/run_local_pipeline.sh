#!/bin/bash
# Local counterpart of sbatch_ayush_pipeline.sh for running the refusal-mech
# pipeline on a dev box (macOS / Apple Silicon MPS or a local CUDA GPU).
#
# No Slurm, no Singularity. Assumes the repo's .venv already has the pinned
# requirements installed.
#
# From the repo root:
#   bash ayush/run_local_pipeline.sh
# Optionally override stages via env:
#   RUN_TRACE=0 RUN_FEATURE_INTERVENE=0 bash ayush/run_local_pipeline.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Stage toggles (override via env). trace / feature-intervene default off
# locally because circuit-tracer is heavy and feature intervention needs a
# hand-curated refusal_features.json.
RUN_SANITY=${RUN_SANITY:-1}
RUN_EXTRACT=${RUN_EXTRACT:-0}
RUN_INTERVENE=${RUN_INTERVENE:-1}
RUN_ABLATE_LONG=${RUN_ABLATE_LONG:-1}
RUN_VISUALIZE_DIR=${RUN_VISUALIZE_DIR:-1}
RUN_TRACE=${RUN_TRACE:-0}
RUN_FEATURE_INTERVENE=${RUN_FEATURE_INTERVENE:-0}
RUN_VISUALIZE_STEER=${RUN_VISUALIZE_STEER:-1}

if [ ! -x .venv/bin/python ]; then
  echo "ERROR: .venv/bin/python not found. Create it first:" >&2
  echo "  python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

PY=".venv/bin/python"
export PYTHONUNBUFFERED=1
# Silence transformer_lens MPS warnings (they are noisy, not actionable here).
export TRANSFORMERLENS_ALLOW_MPS=1

mkdir -p logs

echo "=== ayush_pipeline (local) | pid $$ | $(date -u +%FT%TZ) ==="
echo "Repo: ${REPO_ROOT}"
${PY} -c 'import torch; print("torch", torch.__version__, "mps", torch.backends.mps.is_available(), "cuda", torch.cuda.is_available())'

run_step() {
  local name="$1"; shift
  echo
  echo "----- [${name}] $(date -u +%FT%TZ) -----"
  time "$@"
}

[ "${RUN_SANITY}" = "1" ]           && run_step sanity_check           "${PY}" ayush/sanity_check.py
[ "${RUN_EXTRACT}" = "1" ]          && run_step extract_direction      "${PY}" ayush/extract_refusal_direction.py
[ "${RUN_INTERVENE}" = "1" ]        && run_step intervene_refusal      "${PY}" ayush/intervene_refusal.py
[ "${RUN_ABLATE_LONG}" = "1" ]      && run_step intervene_ablate_long  "${PY}" ayush/intervene_ablate_long.py
[ "${RUN_VISUALIZE_DIR}" = "1" ]    && run_step visualize_direction    "${PY}" ayush/visualize_refusal_direction.py
[ "${RUN_TRACE}" = "1" ]            && run_step trace_refusal_circuit  "${PY}" ayush/trace_refusal_circuit.py
[ "${RUN_FEATURE_INTERVENE}" = "1" ] && run_step intervene_features     "${PY}" ayush/intervene_refusal_features.py
[ "${RUN_VISUALIZE_STEER}" = "1" ]  && run_step visualize_steering     "${PY}" ayush/visualize_steering.py

echo
echo "=== done | results under: ${REPO_ROOT}/results/ ==="
ls -la results/ || true
