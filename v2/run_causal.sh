#!/bin/bash
# Causal-test round (E5-E9) on a 2-GPU pod. Assumes run_all.sh outputs in $AD_OUT.
export HF_HOME=/workspace/hf AD_OUT=/workspace/results_v2 VLLM_WORKER_MULTIPROC_METHOD=spawn
cd /workspace/ad_v2/v2
L=/workspace/logs
run() { name=$1; shift; [ -f $AD_OUT/.done_$name ] && { echo "skip $name"; return 0; }
  echo "=== $(date -u +%H:%M) $name"; "$@" > $L/$name.log 2>&1 && touch $AD_OUT/.done_$name && echo "ok $name" || echo "FAIL $name"; }
(
  export CUDA_VISIBLE_DEVICES=0
  run ablate_qwen  python causal.py ablate --model qwen3-14b --n 100 --bs 16
  run steer1_qwen  python causal.py steer1 --model qwen3-14b --n 64 --bs 8
  run head_qwen    python causal.py head   --model qwen3-14b --n 64 --bs 8
  for e in ablate steer1 head; do run judge_${e}_qwen python judge.py --in $AD_OUT/gen_${e}_qwen3-14b.jsonl; done
  echo "=== $(date -u +%H:%M) GPU0_DONE"
) > $L/causal_gpu0.log 2>&1 &
(
  export CUDA_VISIBLE_DEVICES=1
  for m in qwen3-14b gemma2-9b llama31-8b; do run prefill_$m python causal.py prefill --model $m; done
  run ablate_llama  python causal.py ablate --model llama31-8b --n 100 --bs 16
  run steer1_llama  python causal.py steer1 --model llama31-8b --n 64 --bs 8
  run steer1_gemma  python causal.py steer1 --model gemma2-9b --n 64 --bs 8
  for m in qwen3-14b gemma2-9b llama31-8b; do run judge_prefill_$m python judge.py --in $AD_OUT/gen_prefill_$m.jsonl; done
  for e in ablate steer1; do run judge_${e}_llama python judge.py --in $AD_OUT/gen_${e}_llama31-8b.jsonl; done
  run judge_steer1_gemma python judge.py --in $AD_OUT/gen_steer1_gemma2-9b.jsonl
  run sr_main python srjudge.py --in $AD_OUT/judged_qwen3-14b.jsonl --in $AD_OUT/judged_gemma2-9b.jsonl --in $AD_OUT/judged_llama31-8b.jsonl
  echo "=== $(date -u +%H:%M) GPU1_DONE"
) > $L/causal_gpu1.log 2>&1 &
wait
CUDA_VISIBLE_DEVICES=0 run sr_causal python srjudge.py $(for f in $AD_OUT/judged_ablate_*.jsonl $AD_OUT/judged_prefill_*.jsonl $AD_OUT/judged_steer1_*.jsonl $AD_OUT/judged_head_*.jsonl $AD_OUT/judged_steer_*.jsonl; do echo --in $f; done)
echo "=== $(date -u +%H:%M) CAUSAL_ALL_DONE"
