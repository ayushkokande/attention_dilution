# Project documentation (v2: "Deferred Refusal")

Start here if you are picking this project up later: you, a co-author, a reviewer asking for a rerun, or
an AI assistant in a new session. Nothing below depends on any chat history.

| Document | Read it when you want to... |
|---|---|
| [`HISTORY.md`](HISTORY.md) | understand how the project went from the course report (v1) to the current paper, and why |
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | find any experiment: what it tests, the command, its output file, and where it appears in the paper |
| [`RUNBOOK.md`](RUNBOOK.md) | rent GPUs, run jobs on RunPod, move files, avoid the problems we hit, and control cost |
| [`PAPER.md`](PAPER.md) | build the two PDFs (preprint and STALA/NDSS), regenerate tables, and change numbers safely |
| [`SUBMISSION.md`](SUBMISSION.md) | submit: venues, deadlines, what is still missing, and the pre-submission checklist |
| [`../v2/PREREGISTRATION.md`](../v2/PREREGISTRATION.md) | see the predictions committed before each confirmation run, and their outcomes |

## The project in one paragraph

The original course project (v1, May 2026) concluded that putting a harmful request in a short numbered list
of benign tasks collapses Qwen3-14B's refusal. v2 shows that conclusion came from measuring refusal in the first
24 generated tokens with the harmful item at an uncontrolled position. With full responses judged, models refuse
the harmful item at the item itself, often hundreds of tokens in. The paper's main claim is therefore about
measurement: short-horizon judges misread multi-request prompts. The supporting mechanistic claim is that where a
model carries its refusal decision (prompt versus item) differs across models and matches where it refuses. This
was confirmed in preregistered held-out tests.

## Where things are

```
v2/                 experiment code (generation, judging, probes, causal tests, analysis); see v2/README.md
v2/pod_scripts/     the ad-hoc queue scripts actually run on the pods (provenance)
tools/              RunPod + Jupyter helpers (rp.py, jc.py, launch.py, deadman.sh); keys read from ~/.secrets
paper_v2/           LaTeX: main.tex (ICLR-style preprint), main_ndss.tex (STALA/NDSS two-column, anonymous)
results/v2/         public labels, probes, directions, summaries (no model response text)
docs/               this documentation
experiment_*/ Suraj/ paper/   v1 (the course project), kept unchanged
```

Private data, **not in git**: full judged files with response text (some responses are harmful). These are the
`judged_*.jsonl`, `*.sr.jsonl` and `*.srseg.jsonl` files. They were delivered to the authors as a zip and live
on the pods under `/workspace/results_v2/` while the pods exist. Every number in the paper can be recomputed
from them with `v2/analyze.py`, `v2/make_tables.py` and `v2/analyze_causal.py`. Most numbers can also be
recomputed from the public labels in `results/v2/`.
