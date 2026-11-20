#!/bin/bash
export HF_HOME=/workspace/hf AD_OUT=/workspace/results_v2 CUDA_VISIBLE_DEVICES=0
L=/workspace/logs
cd /workspace/ad_v2 && git pull -q; cd v2
pip install -q peft > $L/pip_peft.log 2>&1
python srjudge.py --in $AD_OUT/judged_ablateho_qwen3-14b.jsonl --in $AD_OUT/judged_ablateho_llama31-8b.jsonl --in $AD_OUT/judged_scale_qwen3-32b.jsonl > $L/sr_pod2.log 2>&1 && echo ok_sr2 || echo FAIL_sr2
echo SR2_DONE
