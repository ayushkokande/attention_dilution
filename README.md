# Attention dilution

A controlled study of how added benign context changes Qwen3's refusal behavior and residual activations.

The question is deliberately small: **when the same request follows more inert text, do refusal rates or projections onto a harmful-versus-harmless direction change?** A flat curve is a useful result. We do not assume in advance that attention normalization is the cause.

## Core experiments

| Stage | What it measures | Script |
| --- | --- | --- |
| Baseline | Phrase-refusal rates on AdvBench and filtered Alpaca | `experiment_1/baseline_benchmark.py` |
| Direction | Difference of class means at each block; select a layer on separate validation prompts | `experiment_2/refusal_direction.py` |
| Context | Intact vs. post-block directional ablation as inert prefix length increases | `experiment_8/context_sweep.py` |
| Projection | Residual projection at the last request token and at the chat-template readout token | `experiment_9/projection_sweep.py` |

The shared code lives in `attention_dilution/`. The numbered entry points remain usable. Head ranking, style/topic/policy analyses, steering, multi-format prompts, and circuit tracing are exploratory follow-ups rather than requirements for this study.

## Setup and runs

Use Python 3.11 or newer and an environment with enough memory for the selected model. Install a PyTorch build suitable for your hardware, then the core dependencies:

```bash
python -m pip install -r requirements.txt
```

Run from the repository root. These commands explicitly name their output directories:

```bash
python -m attention_dilution baseline --output-dir runs/baseline
python -m attention_dilution direction --output-dir runs/direction
python -m attention_dilution context --refusal-dir runs/direction --output-dir runs/context
python -m attention_dilution projection --refusal-dir runs/direction --output-dir runs/projection
```

All four default to `Qwen/Qwen3-14B`. For a cheap model smoke run, use the same `--model Qwen/Qwen3-1.7B` throughout, smaller pool sizes, and `--lengths 0 128` for the sweeps. Command help is available without loading model dependencies:

```bash
python -m attention_dilution direction --help
```

Without `--output-dir`, each command creates a separate timestamped run directory. Existing runs require `--resume`, which checks the saved settings, exact prompts, and direction artifact hashes. Sampling runs cannot resume because batching can change their random draws. The old shell launchers now forward arguments to these commands; they do not install environments or assume a particular cluster.

Each run contains:

- `run.json`: settings, prompt text, scoring configuration, and input artifact hashes.
- `environment.json`: resolved model/tokenizer revisions and package versions.
- Stage results, with incremental generation JSONL or projection checkpoints.

Use `--revision` to pin a model revision when repeating a study. Dataset inputs are recorded in the run manifest; changes to retrieved prompts prevent a resume from mixing old and new data.

## Measurement rules

The core ranges in `splits.json` separate direction training, layer selection, and held-out evaluation. The baseline covers the full datasets and is descriptive; it is not an independent held-out test. Direction extraction also checks for identical prompt strings across splits.

The behavioral metric is an **18-phrase heuristic over the first 200 characters**, with 256 generated tokens by default. A missing phrase is not a verified harmful answer. Empty outputs and generation errors are reported separately. Results include sample counts, truncation counts, and Wilson 95% intervals for phrase-refusal rates.

The context sweep adds one inert passage before a single request. It does not insert extra instructions or change the conversation structure. Input-template overhead and output tokens count toward the 32768-token default budget; a 32768-token filler alone is too long.

Directions are unit vectors extracted at block outputs. The ablation removes the selected direction at each block output. The projection sweep measures two distinct token positions and calls the base decoder directly, without allocating unnecessary vocabulary logits.

## Scope and historical results

The existing `results/`, `Suraj/results_v3/`, paper source, and PDFs are preserved. They were produced with older settings and have not been rerun by this refactor. In particular, the earlier harmless split in experiment 6 overlaps the expanded 256-prompt training pool.

The [review notes](docs/review.md) explain the code defects, misleading wording, and limitations of the earlier multi-request scoring and circuit interpretation. The [core settings](INVARIANTS.md) describe the maintained protocol. The old merge and phase plans are in `docs/history/`.

The present study supports claims about behavior and residual projections on this model and prompt set. Establishing a causal attention mechanism would additionally require target-response grading, position-matched controls, and interventions that recover the actual behavior on held-out prompts.

## Checks

The regression suite checks split separation, run isolation, token locations, result denominators, and hook cleanup. It runs without downloading model weights:

```bash
python -m unittest discover -s tests -v
```

Tiny decoder tests require PyTorch and are skipped when it is absent. CI installs a CPU build to run them. Full Qwen3 experiments require a separate model run; passing these checks does not validate or regenerate the historical results.

