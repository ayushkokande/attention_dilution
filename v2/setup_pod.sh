#!/bin/bash
# One-time pod setup: python deps + model downloads. Idempotent.
set -x
export HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=1
cd /workspace/ad_v2
pip install -q "vllm==0.11.0" 2>&1 | tail -2
pip install -q "transformers==4.57.3" "datasets<4.5" hf_transfer pandas scipy matplotlib websocket-client 2>&1 | tail -2
python -c "import vllm, transformers, torch; print('versions', vllm.__version__, transformers.__version__, torch.__version__)"
for m in Qwen/Qwen3-14B google/gemma-2-9b-it cais/HarmBench-Llama-2-13b-cls; do
  huggingface-cli download $m --exclude "*.pth" "original/*" >/dev/null 2>&1 && echo "downloaded $m" || echo "FAILED $m"
done
python - <<'PY'
from datasets import load_dataset
print(len(load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")), len(load_dataset("tatsu-lab/alpaca", split="train")))
PY
echo SETUP_DONE
