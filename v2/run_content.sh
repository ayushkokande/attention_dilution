#!/bin/bash
# Content-type experiment; runs on GPU1 of pod 1 after round 3.
export HF_HOME=/workspace/hf AD_OUT=/workspace/results_v2 VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=1
cd /workspace/ad_v2/v2; L=/workspace/logs
until grep -q R3_GPU1_DONE $L/r3_gpu1.log 2>/dev/null; do sleep 30; done
for m in qwen3-14b llama31-8b gemma2-9b; do
  python generate.py --model $m --variant content > $L/gen_content_$m.log 2>&1 && \
  python judge.py --in $AD_OUT/gen_content_$m.jsonl > $L/judge_content_$m.log 2>&1 && echo "ok content $m" || echo "FAIL content $m"
done
echo CONTENT_DONE
