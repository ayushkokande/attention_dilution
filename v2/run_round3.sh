#!/bin/bash
# Extra claims: reasoning mode (Qwen3 thinking), sampled decoding (3 models), prompt-only ablation on Gemma.
export HF_HOME=/workspace/hf AD_OUT=/workspace/results_v2 VLLM_WORKER_MULTIPROC_METHOD=spawn
cd /workspace/ad_v2/v2
L=/workspace/logs
run() { name=$1; shift; [ -f $AD_OUT/.done_$name ] && { echo "skip $name"; return 0; }
  echo "=== $(date -u +%H:%M) $name"; "$@" > $L/$name.log 2>&1 && touch $AD_OUT/.done_$name && echo "ok $name" || echo "FAIL $name"; }
(
  until grep -q GPU0_DONE $L/causal_gpu0.log 2>/dev/null; do sleep 30; done
  export CUDA_VISIBLE_DEVICES=0
  run gen_thinking_qwen python generate.py --model qwen3-14b --variant thinking
  run judge_thinking_qwen python judge.py --in $AD_OUT/gen_thinking_qwen3-14b.jsonl
  run ablate_gemma python causal.py ablate --model gemma2-9b --n 100 --bs 16
  run judge_ablate_gemma python judge.py --in $AD_OUT/gen_ablate_gemma2-9b.jsonl
  echo "=== $(date -u +%H:%M) R3_GPU0_DONE"
) > $L/r3_gpu0.log 2>&1 &
(
  until grep -q GPU1_DONE $L/causal_gpu1.log 2>/dev/null; do sleep 30; done
  export CUDA_VISIBLE_DEVICES=1
  for m in qwen3-14b gemma2-9b llama31-8b; do
    run gen_sample_$m python generate.py --model $m --variant sample
    run judge_sample_$m python judge.py --in $AD_OUT/gen_sample_$m.jsonl
  done
  echo "=== $(date -u +%H:%M) R3_GPU1_DONE"
) > $L/r3_gpu1.log 2>&1 &
wait
echo "=== $(date -u +%H:%M) R3_ALL_DONE"
