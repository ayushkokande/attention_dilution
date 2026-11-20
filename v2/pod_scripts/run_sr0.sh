#!/bin/bash
export HF_HOME=/workspace/hf AD_OUT=/workspace/results_v2 CUDA_VISIBLE_DEVICES=0
L=/workspace/logs; cd /workspace/ad_v2/v2
python srjudge.py --in $AD_OUT/judged_qwen3-14b.jsonl --in $AD_OUT/judged_gemma2-9b.jsonl --in $AD_OUT/judged_llama31-8b.jsonl --in $AD_OUT/judged_ablate_gemma2-9b.jsonl --in $AD_OUT/judged_thinking_qwen3-14b.jsonl > $L/sr_main0.log 2>&1 && echo ok_sr_main || echo FAIL_sr_main
echo SR0_DONE
