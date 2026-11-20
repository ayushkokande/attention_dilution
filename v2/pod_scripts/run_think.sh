#!/bin/bash
export HF_HOME=/workspace/hf AD_OUT=/workspace/results_v2 VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=0
L=/workspace/logs
until grep -q R3_GPU0_DONE $L/r3_gpu0.log 2>/dev/null; do sleep 30; done
sleep 20
cd /workspace/ad_v2/v2
python generate.py --model qwen3-14b --variant thinking > $L/gen_thinking_qwen2.log 2>&1 && \
python judge.py --in $AD_OUT/gen_thinking_qwen3-14b.jsonl > $L/judge_thinking_qwen2.log 2>&1 && echo ok_thinking || echo FAIL_thinking
echo THINK_DONE
