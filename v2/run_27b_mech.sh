#!/bin/bash
# Pod 1 GPU1: Gemma-2-27B direction + preregistered held-out prompt-only ablation (Addendum 3).
export HF_HOME=/workspace/hf AD_OUT=/workspace/results_v2 VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=1
cd /workspace/ad_v2/v2; L=/workspace/logs
run() { name=$1; shift; [ -f $AD_OUT/.done_$name ] && { echo "skip $name"; return 0; }
  echo "=== $(date -u +%H:%M) $name"; "$@" > $L/$name.log 2>&1 && touch $AD_OUT/.done_$name && echo "ok $name" || echo "FAIL $name"; }
cp /workspace/ad_v2/results/v2/harmbench_text_all.csv $AD_OUT/ 2>/dev/null || true
run dir_27b python extract_direction.py --model gemma2-27b
run ablateho_27b python causal.py ablate --model gemma2-27b --heldout --bs 16
until grep -q SRSEG_DONE $L/run_srseg.log 2>/dev/null; do sleep 30; done
CUDA_VISIBLE_DEVICES=0 run judge_ablateho_27b python judge.py --in $AD_OUT/gen_ablateho_gemma2-27b.jsonl
echo "=== $(date -u +%H:%M) M27_DONE"
