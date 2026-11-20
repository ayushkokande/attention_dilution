#!/bin/bash
# Second pod: Qwen3-32B (behaviour + direction + probes) and the framing comparison on all models.
export HF_HOME=/workspace/hf AD_OUT=/workspace/results_v2 VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p $AD_OUT /workspace/logs; cd /workspace/ad_v2/v2
L=/workspace/logs
run() { name=$1; shift; [ -f $AD_OUT/.done_$name ] && { echo "skip $name"; return 0; }
  echo "=== $(date -u +%H:%M) $name"; "$@" > $L/$name.log 2>&1 && touch $AD_OUT/.done_$name && echo "ok $name" || echo "FAIL $name"; }
pip install -q "vllm==0.11.0" > $L/pip.log 2>&1
pip install -q "transformers==4.57.3" "datasets<4.5" hf_transfer accelerate >> $L/pip.log 2>&1
huggingface-cli download Qwen/Qwen3-32B > $L/dl32.log 2>&1
( for m in Qwen/Qwen3-14B google/gemma-2-9b-it meta-llama/Llama-3.1-8B-Instruct cais/HarmBench-Llama-2-13b-cls; do
    huggingface-cli download $m --exclude "original/*" >> $L/dl.log 2>&1; done ) &
run gen_scale_32b  python generate.py --model qwen3-32b --variant scale --tp 2
run dir_32b        python extract_direction.py --model qwen3-32b
wait
run judge_scale_32b python judge.py --in $AD_OUT/gen_scale_qwen3-32b.jsonl
run probe_32b      python probe.py --model qwen3-32b --judged $AD_OUT/judged_scale_qwen3-32b.jsonl --n 200
run gen_framing_32b python generate.py --model qwen3-32b --variant framing --tp 2
for m in qwen3-14b gemma2-9b llama31-8b; do run gen_framing_$m python generate.py --model $m --variant framing --tp 2; done
for m in qwen3-32b qwen3-14b gemma2-9b llama31-8b; do run judge_framing_$m python judge.py --in $AD_OUT/gen_framing_$m.jsonl; done
echo "=== $(date -u +%H:%M) SCALE_ALL_DONE"
