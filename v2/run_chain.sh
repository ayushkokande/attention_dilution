#!/bin/bash
# Usage: run_chain.sh <gpu> <model> [extract]
# Runs generate -> (extract) -> judge -> probe -> steer -> judge(steer) for one model on one GPU.
GPU=$1; M=$2; EXTRACT=$3
export CUDA_VISIBLE_DEVICES=$GPU HF_HOME=/workspace/hf AD_OUT=/workspace/results_v2 VLLM_WORKER_MULTIPROC_METHOD=spawn
mkdir -p $AD_OUT /workspace/logs; cd /workspace/ad_v2/v2
step() { name=$1; shift; [ -f $AD_OUT/.done_${M}_$name ] && { echo "skip $name"; return 0; }
  echo "=== $(date -u +%H:%M) $M $name"; "$@" > /workspace/logs/${M}_$name.log 2>&1 && touch $AD_OUT/.done_${M}_$name && echo "ok $name" || { echo "FAIL $name"; return 1; }; }
step gen python generate.py --model $M || exit 1
[ "$EXTRACT" = extract ] && { step dir python extract_direction.py --model $M || exit 1; }
step judge python judge.py --in $AD_OUT/gen_$M.jsonl || exit 1
step probe python probe.py --model $M
step steer python steer.py --model $M && step judge_steer python judge.py --in $AD_OUT/gen_steer_$M.jsonl
echo "=== $(date -u +%H:%M) $M CHAIN_DONE"
