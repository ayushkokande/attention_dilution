#!/bin/bash
# Pod 1, after the 70B ablation: rerun the Gemma-2-27B preregistered ablation (predictions 8-9) with the
# coherence-checked layer selection. The first attempt (degenerate generation) is kept with suffix _attempt1.
export HF_HOME=/workspace/hf AD_OUT=/workspace/results_v2 VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=0,1 HF_HUB_ENABLE_HF_TRANSFER=1
L=/workspace/logs
until grep -q M70_DONE $L/run_70b_mech.log 2>/dev/null; do sleep 30; done
cd /workspace/ad_v2 && git pull -q; cd v2
rm -rf /workspace/hf/hub/models--meta-llama--Llama-3.1-70B-Instruct
cd $AD_OUT
for f in gen_ablateho_gemma2-27b.jsonl judged_ablateho_gemma2-27b.jsonl dirs_gemma2-27b.pt; do [ -f $f ] && mv $f ${f%.*}_attempt1.${f##*.}; done
rm -f .done_dir_27b .done_ablateho_27b .done_judge_ablateho_27b
cd /workspace/ad_v2/v2
run() { name=$1; shift; [ -f $AD_OUT/.done_$name ] && { echo "skip $name"; return 0; }
  echo "=== $(date -u +%H:%M) $name"; "$@" > $L/$name.log 2>&1 && touch $AD_OUT/.done_$name && echo "ok $name" || echo "FAIL $name"; }
run dl_27b2 huggingface-cli download google/gemma-2-27b-it --exclude "original/*"
run dir_27b python extract_direction.py --model gemma2-27b
run ablateho_27b python causal.py ablate --model gemma2-27b --heldout --bs 16
run judge_ablateho_27b python judge.py --in $AD_OUT/gen_ablateho_gemma2-27b.jsonl
echo "=== $(date -u +%H:%M) M27B_DONE"
