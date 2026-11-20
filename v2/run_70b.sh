#!/bin/bash
# Pod 2: Llama-3.1-70B-Instruct behaviour (scale grid), direction, preregistered held-out ablation.
export HF_HOME=/workspace/hf AD_OUT=/workspace/results_v2 VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HUB_ENABLE_HF_TRANSFER=1
cd /workspace/ad_v2/v2; L=/workspace/logs
run() { name=$1; shift; [ -f $AD_OUT/.done_$name ] && { echo "skip $name"; return 0; }
  echo "=== $(date -u +%H:%M) $name"; "$@" > $L/$name.log 2>&1 && touch $AD_OUT/.done_$name && echo "ok $name" || echo "FAIL $name"; }
run dl_70b       huggingface-cli download meta-llama/Llama-3.1-70B-Instruct --exclude "original/*"
run gen_scale_70b  python generate.py --model llama31-70b --variant scale --tp 2
run judge_scale_70b python judge.py --in $AD_OUT/gen_scale_llama31-70b.jsonl
run dir_70b      python extract_direction.py --model llama31-70b
run ablateho_70b python causal.py ablate --model llama31-70b --heldout --bs 16
run judge_ablateho_70b python judge.py --in $AD_OUT/gen_ablateho_llama31-70b.jsonl
echo "=== $(date -u +%H:%M) R70_DONE"
