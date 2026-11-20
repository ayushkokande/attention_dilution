# Attention Dilution

**A Mechanistic Analysis of Context-Length Jailbreaks and Activation Steering in LLMs**

Ayush Kokande, Suraj Mishra · New York University · May 2026

📄 **Paper:** [`Attention_Dilution_Submission.pdf`](Attention_Dilution_Submission.pdf) (LaTeX source in [`paper/`](paper/))

---

> **Update (Nov 2026): v2 re-measures this project and overturns its headline result.** With the harmful
> item's position controlled and full responses judged, models refuse the item at the item itself; the
> "collapse" came from a 24-token judge. The new paper ("Deferred Refusal: Short-Horizon Judges Misread Safety
> in Multi-Request Prompts") is in [`paper_v2/`](paper_v2/), code in [`v2/`](v2/), and full documentation in
> [`docs/`](docs/) (branch `v2-deferred-refusal`). The v1 description below is kept as the original record.

## TL;DR

Safety-tuned language models that reliably refuse a harmful request in isolation often comply with the
same request once it is buried in benign context. This failure mode is invisible to standard red-team
benchmarks, which test prompts on their own. We ask **why, mechanistically**, in **Qwen3-14B**.

The short answer: refusal breaks because the model's guardrail attention head **stops looking at the
harmful request**. The refusal representation itself is intact. Only certain prompt *structures* cause
this, not raw context length.

## Key findings

1. **A causal refusal direction.** Following [Arditi et al. (2024)](https://arxiv.org/abs/2406.11717), we
   extract a single refusal direction d̂ in the residual stream by difference-of-means. We select it at
   layer 36 with a causal-ablation criterion and validate it against four confound controls (verb
   class/length, style/vocabulary, topic, and harm-refusal vs. policy-refusal). These show d̂ encodes
   harmful *intent* rather than surface artefacts.
2. **Format, not length.** We sweep six "bloat" formats (prefix, suffix, sandwich, distractor,
   multi-turn, many-shot) at context lengths up to N = 4096 tokens:
   - Only the **distractor** (numbered-list) format collapses refusal, at just **~31 tokens**.
   - Four of the other five formats keep refusal at **≥ 87%** even at N = 4096. Suffix degrades
     partially, to 0.59.
3. **Attentional, not representational.** The dominant *Guardrail Head*, **L36H31**, was found by
   direct logit attribution. Its attention mass on the harmful tokens drops **6×** between N = 0 and
   N = 4096, while d̂ itself is unaltered.
4. **Why single-direction steering can't rescue it.** At N ≥ 512, no steering strength
   α ∈ {0, …, 16} restores refusal. Under dilution, the d̂-projection at the last token **flips
   sign**, which places the residual stream deep in the anti-refusal half-space.
5. **A two-head circuit.** Denoising path patching localizes the upstream lesion to a single
   mid-network head, **L20H10**. This defines the circuit **L20H10 → L36H31**, whose silencing under
   dilution causes the behavioural collapse.

![Distractor-format dense sweep on Qwen3-14B](Suraj/results_v3/fig_phase2_dense_distractor.png)

*Distractor format on Qwen3-14B. As bloat grows, the attention mass on the harmful span (H1, blue) and
the refusal rate (purple, dashed) collapse together. Meanwhile, the cosine between the harmful-span
residual and the refusal direction (H2b, green) stays flat, so the refusal representation is still
there and the model just stops attending to it.*

## Setup

| | |
|---|---|
| **Model** | `Qwen/Qwen3-14B` (40 blocks, 40 query heads / 8 KV heads, d_model = 5120), bf16, `enable_thinking=False` |
| **Harmful prompts** | AdvBench `harmful_behaviors.csv` (520 prompts) |
| **Harmless prompts** | Alpaca train split, empty-input rows after a content filter |
| **Refusal detector** | 18 canonical refusal substrings on the first 200 characters, after stripping any `<think>` block |
| **Splits** | Per-experiment row indices pinned in [`splits.json`](splits.json); train/eval disjointness enforced |
| **Hardware** | 2× A100-80GB (RunPod; earlier runs on NYU Greene via Slurm) |

Every cross-experiment constant lives in [`INVARIANTS.md`](INVARIANTS.md). That covers the model,
chat template, refusal detector, canonical layer and datasets.

## Repository layout

### Main pipeline: `experiment_1` … `experiment_9`

| Experiment | What it does | Entry point |
|---|---|---|
| [`experiment_1`](experiment_1/) | Baseline refusal rates (AdvBench vs. Alpaca); shared utilities | `baseline_benchmark.py` |
| [`experiment_2`](experiment_2/) | Per-layer refusal direction d̂ via difference-of-means and a causal-ablation layer sweep (canonical L36) | `refusal_direction.py` |
| [`experiment_3`](experiment_3/) | Guardrail heads via direct logit attribution onto d̂ | `guardrail_heads.py` |
| [`experiment_4`](experiment_4/) | Validity check 1: verb-class/length-matched pools; compares d̂ against the matched d̂* | `curate_matched_pools.py`, `compare_directions.py` |
| [`experiment_5`](experiment_5/) | Validity check 2: style/vocabulary (2×2 intent × "edgy" lexicon, AUCs) | `build_styled_pools.py`, `eval_validity.py` |
| [`experiment_6`](experiment_6/) | Validity check 3: topic decoupling (2×2 intent × topic, two-way ANOVA) | `build_topic_pools.py`, `eval_topic_decouple.py` |
| [`experiment_7`](experiment_7/) | Validity check 4: harm-refusal vs. policy-refusal AUC | `build_policy_pools.py`, `eval_policy_vs_harm.py` |
| [`experiment_8`](experiment_8/) | Two-arm context-length sweep (refusal intact vs. ablated), binary refusal rate vs. N | `context_sweep.py` |
| [`experiment_9`](experiment_9/) | Continuous projection of the last-token residual onto d̂ vs. N | `projection_sweep.py` |

Each experiment has a `sbatch_*.sh` launcher that documents its prerequisites and outputs.

### Six-phase pipeline: [`Suraj/`](Suraj/)

[`Suraj/experiment.py`](Suraj/experiment.py) is a single-script pipeline on Qwen3-14B. It covers the
format triage, dense distractor sweep, steering rescue, capability cost (MMLU/GSM8K), the 2×2
intact/ablated grid, attribution maps, and path patching / circuit tracing. Results and figures are in
[`Suraj/results_v3/`](Suraj/results_v3/). See [`MERGER.md`](MERGER.md) for how its phases map onto the
numbered experiments.

### Everything else

- [`results/qwen3-14b/`](results/qwen3-14b/): outputs of the numbered experiments (d̂ per layer, guardrail heads, validity battery, context sweep).
- [`results/qwen3-1.7b/`](results/qwen3-1.7b/): an earlier pilot on Qwen3-1.7B.
- [`paper/`](paper/): LaTeX source (`main.tex`, `refs.bib`, `figures/`).
- [`Project_Proposal_LLM_Reasoners.pdf`](Project_Proposal_LLM_Reasoners.pdf): the original project proposal.

## Reproducing

```bash
pip install -r requirements.txt
```

1. **Hardware.** Qwen3-14B in bf16 with hooks needs about 2× A100-80GB.
2. **Repo path.** The launchers default to `REPO=/workspace/attention_dilutionv2` (a RunPod network
   volume). Override `WORKSPACE` / `REPO` to point at your clone.
3. **Run order:**

   ```bash
   bash experiment_1/sbatch_baseline_qwen14b.sh            # baseline refusal rates
   bash experiment_2/sbatch_refusal_direction.sh           # d̂ for every layer + canonical layer (needed by 3, 8, 9)
   bash experiment_3/sbatch_guardrail_heads.sh             # guardrail heads
   bash experiment_4/sbatch_refusal_direction_matched.sh   # validity battery (4–7)
   bash experiment_5/sbatch_eval_validity.sh
   bash experiment_6/sbatch_topic_decouple.sh
   bash experiment_7/sbatch_policy_validity.sh
   bash experiment_8/sbatch_context_sweep.sh               # binary refusal vs. context length
   bash experiment_9/sbatch_projection_sweep.sh            # continuous d̂ projection vs. context length
   ```

4. **Six-phase pipeline** (see the script's docstring for all flags):

   ```bash
   python Suraj/experiment.py --phases 1,2_triage,2_dense
   python Suraj/experiment.py --phases 5,6 --resume
   ```

## Citation

```bibtex
@misc{kokande2026attentiondilution,
  title  = {Attention Dilution: A Mechanistic Analysis of Context-Length Jailbreaks and Activation Steering in LLMs},
  author = {Kokande, Ayush and Mishra, Suraj},
  year   = {2026},
  note   = {New York University},
  url    = {https://github.com/ayushkokande/attention_dilution}
}
```

## References

- Arditi et al. (2024). [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717).
- Anil et al. (2024). [Many-Shot Jailbreaking](https://www.anthropic.com/research/many-shot-jailbreaking).
- Full bibliography: [`paper/refs.bib`](paper/refs.bib).
