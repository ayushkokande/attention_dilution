#!/bin/bash
# Full overnight run on a 2-GPU pod. Logs in /workspace/logs.
cd /workspace/ad_v2/v2
( bash run_chain.sh 0 qwen3-14b ) > /workspace/logs/chain_gpu0.log 2>&1 &
( bash run_chain.sh 1 gemma2-9b extract
  code=$(curl -s -o /dev/null -w %{http_code} -H "Authorization: Bearer $HF_TOKEN" https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/resolve/main/config.json)
  if [ "$code" = 200 ]; then HF_HOME=/workspace/hf huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --exclude "original/*" >/dev/null 2>&1; bash run_chain.sh 1 llama31-8b extract; else echo "llama access: $code"; fi
) > /workspace/logs/chain_gpu1.log 2>&1 &
wait
echo ALL_DONE
