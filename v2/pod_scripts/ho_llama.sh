#!/bin/bash
export HF_HOME=/workspace/hf AD_OUT=/workspace/results_v2 CUDA_VISIBLE_DEVICES=1
cd /workspace/ad_v2/v2
python causal.py ablate --model llama31-8b --heldout --bs 16 > /workspace/logs/ablateho_llama2.log 2>&1 && echo ok_gen
until grep -q HO_DONE /workspace/logs/run_ho.log; do sleep 20; done
python judge.py --in /workspace/results_v2/gen_ablateho_llama31-8b.jsonl > /workspace/logs/judge_ablateho_llama2.log 2>&1 && echo ok_judge
echo HO_LLAMA_DONE
