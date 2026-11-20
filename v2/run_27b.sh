#!/bin/bash
# Pod 1: Gemma-2-27B-it behaviour on the scale grid.
export HF_HOME=/workspace/hf AD_OUT=/workspace/results_v2 VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HUB_ENABLE_HF_TRANSFER=1
cd /workspace/ad_v2/v2; L=/workspace/logs
until grep -q SR0_DONE $L/run_sr0.log 2>/dev/null; do sleep 30; done
run() { name=$1; shift; [ -f $AD_OUT/.done_$name ] && { echo "skip $name"; return 0; }
  echo "=== $(date -u +%H:%M) $name"; "$@" > $L/$name.log 2>&1 && touch $AD_OUT/.done_$name && echo "ok $name" || echo "FAIL $name"; }
run dl_27b  huggingface-cli download google/gemma-2-27b-it --exclude "original/*"
run gen_scale_27b python generate.py --model gemma2-27b --variant scale --tp 2
run judge_scale_27b python judge.py --in $AD_OUT/gen_scale_gemma2-27b.jsonl
echo "=== $(date -u +%H:%M) R27_DONE"
