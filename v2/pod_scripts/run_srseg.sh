#!/bin/bash
export HF_HOME=/workspace/hf AD_OUT=/workspace/results_v2 CUDA_VISIBLE_DEVICES=0
L=/workspace/logs
until grep -q R27_DONE $L/run_27b.log 2>/dev/null; do sleep 30; done
cd /workspace/ad_v2 && git pull -q; cd v2
python srjudge.py --segment --in $AD_OUT/judged_qwen3-14b.jsonl --in $AD_OUT/judged_gemma2-9b.jsonl --in $AD_OUT/judged_llama31-8b.jsonl > $L/srseg.log 2>&1 && echo ok_srseg || echo FAIL_srseg
echo SRSEG_DONE
