# v2: Deferred refusal in multi-request prompts

This folder re-measures the "distractor" result of the original project, controlling the confounds that
produced it. The paper is in [`../paper_v2/`](../paper_v2/) and the results are in
[`../results/v2/`](../results/v2/).

## What changed from v1

| | v1 | v2 |
|---|---|---|
| Refusal scoring | substring match on the first 24 generated tokens | full greedy generation; HarmBench classifier on the whole response; target-specific refusal phrases with a measured false-positive rate |
| Harmful item position | hash-random in the list | controlled: first / middle / last, same benign items |
| Models | Qwen3-14B | Qwen3-14B, Gemma-2-9B-it, Llama-3.1-8B-Instruct; larger: Qwen3-32B, Gemma-2-27B-it, Llama-3.1-70B-Instruct |
| Targets | 100 AdvBench | 100 JailbreakBench + the same 100 AdvBench |
| Templates | 1 | 3 (numbered, bullets, "Task i:") |
| Probes | last prompt token only | last prompt token **and** the harmful item's answer onset in the model's own response |
| Steering | unit vector, alpha <= 16 (about 2.6% of natural scale) | unnormalized diff-of-means vector, alpha in {0.25, 0.5, 1}, applied at the item's decision point |

## Files

| File | Purpose |
|---|---|
| `common.py` | models, data, list construction, segmentation, refusal phrases |
| `generate.py` | E1: greedy generations on the full grid (vLLM) |
| `judge.py` | 24-token v1 metric, target-specific refusal and its locus, HarmBench ASR |
| `extract_direction.py` | diff-of-means direction plus causal layer selection (Gemma, Llama) |
| `probe.py` | E3: projections and guardrail-head attention at the last prompt token and the item answer onset |
| `steer.py` | E4: item-time activation addition |
| `analyze.py`, `make_tables.py` | summary JSON, figures, LaTeX tables and macros for the paper |
| `run_chain.sh`, `run_all.sh`, `setup_pod.sh` | one-command run on a 2-GPU pod |
| `causal.py` | E5–E8, E13–E15: position-masked ablation (`ablate`, `--heldout`), forced openings (`prefill`), single-token addition (`steer1`), head zeroing (`head`) |
| `srjudge.py` | StrongREJECT judge (whole response or `--segment`) |
| `analyze_causal.py` | tables and macros for every intervention and robustness experiment |
| `export_labels.py` | public label export (strips response text) |
| `plot_horizon.py` | judge-horizon figure |
| `PREREGISTRATION.md` | predictions committed before each confirmation run |
| `run_causal.sh`, `run_round3.sh`, `run_content.sh`, `run_scale.sh`, `run_27b*.sh`, `run_70b*.sh`, `pod_scripts/` | the queues actually run on the pods |
| `tests/test_mech.py` | CPU tests: head-attention recompute matches eager attention; hooks; segmentation |

Full documentation (experiment index, compute runbook, paper build, submission checklist): [`../docs/`](../docs/).

## Reproduce

```bash
# on a machine with 2x 80GB GPUs and HF access to the three models
bash v2/setup_pod.sh       # deps + model downloads
bash v2/run_all.sh         # about 2.5 h on 2x A100-80GB
AD_OUT=results/v2 python v2/make_tables.py paper_v2/generated
cd paper_v2 && latexmk -pdf main.tex
```

`run_all.sh` assumes the repository is cloned at `/workspace/ad_v2`. Edit the paths in `run_chain.sh` to change
that.
