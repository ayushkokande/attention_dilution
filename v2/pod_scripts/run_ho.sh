#!/bin/bash
export HF_HOME=/workspace/hf AD_OUT=/workspace/results_v2
L=/workspace/logs
until grep -q SCALE_ALL_DONE /workspace/logs_scale.log; do sleep 30; done
cd /workspace/ad_v2 && git pull -q && cd v2
CUDA_VISIBLE_DEVICES=0 python causal.py ablate --model qwen3-14b --heldout --bs 16 > $L/ablateho_qwen.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python causal.py ablate --model llama31-8b --heldout --bs 16 > $L/ablateho_llama.log 2>&1 &
wait
CUDA_VISIBLE_DEVICES=0 python judge.py --in $AD_OUT/gen_ablateho_qwen3-14b.jsonl > $L/judge_ablateho_qwen.log 2>&1
CUDA_VISIBLE_DEVICES=0 python judge.py --in $AD_OUT/gen_ablateho_llama31-8b.jsonl > $L/judge_ablateho_llama.log 2>&1
echo HO_DONE
