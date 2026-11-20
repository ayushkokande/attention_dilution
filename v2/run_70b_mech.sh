#!/bin/bash
# Pod 1 (after Gemma-2-27B mech): Llama-3.1-70B direction + preregistered held-out ablation (Addendum, preds 6-7).
export HF_HOME=/workspace/hf AD_OUT=/workspace/results_v2 VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HUB_ENABLE_HF_TRANSFER=1
cd /workspace/ad_v2/v2; L=/workspace/logs
until grep -q M27_DONE $L/run_27b_mech.log 2>/dev/null; do sleep 30; done
cd /workspace/ad_v2 && git pull -q; cd v2
# free volume space: keep only the HarmBench judge
rm -rf /workspace/hf/hub/models--Qwen--Qwen3-14B /workspace/hf/hub/models--google--gemma-2-9b-it \
       /workspace/hf/hub/models--meta-llama--Llama-3.1-8B-Instruct /workspace/hf/hub/models--google--gemma-2-27b-it
run() { name=$1; shift; [ -f $AD_OUT/.done_$name ] && { echo "skip $name"; return 0; }
  echo "=== $(date -u +%H:%M) $name"; "$@" > $L/$name.log 2>&1 && touch $AD_OUT/.done_$name && echo "ok $name" || echo "FAIL $name"; }
run dl_70b       huggingface-cli download meta-llama/Llama-3.1-70B-Instruct --exclude "original/*"
run dir_70b      python extract_direction.py --model llama31-70b
run ablateho_70b python causal.py ablate --model llama31-70b --heldout --bs 16
run judge_ablateho_70b python judge.py --in $AD_OUT/gen_ablateho_llama31-70b.jsonl
echo "=== $(date -u +%H:%M) M70_DONE"
