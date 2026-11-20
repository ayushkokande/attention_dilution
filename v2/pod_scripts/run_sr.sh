#!/bin/bash
export HF_HOME=/workspace/hf AD_OUT=/workspace/results_v2 CUDA_VISIBLE_DEVICES=1
L=/workspace/logs
until grep -q CONTENT_DONE $L/run_content.log 2>/dev/null; do sleep 30; done
cd /workspace/ad_v2/v2
python srjudge.py --in $AD_OUT/judged_qwen3-14b.jsonl --in $AD_OUT/judged_gemma2-9b.jsonl --in $AD_OUT/judged_llama31-8b.jsonl > $L/sr_main2.log 2>&1 && echo ok_sr_main || echo FAIL_sr_main
echo SR_DONE
