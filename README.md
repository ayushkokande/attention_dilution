# refusal-mech

Mechanistic analysis of long-context jailbreaks via attention dilution.

The pipeline measures a model's baseline refusal behaviour, extracts a
*refusal direction* in the residual stream, validates it via ablation /
addition interventions, and then dives deeper with
[`circuit-tracer`](https://github.com/safety-research/circuit-tracer) to
identify and intervene on individual MLP transcoder features involved in
refusal.

The default model is **Qwen3-1.7B** (`Qwen/Qwen3-1.7B`); the matching
transcoder set is **`mwhanna/qwen3-1.7b-transcoders-lowl0`**.

## Setup

Use Python **3.11** (see `.python-version`). Create a virtual environment and
install dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Note that `circuit-tracer` pins `transformers<=4.57.3`, so this project does
the same. If you already have a newer `transformers` in the environment, the
pinned version will be downgraded.

## Pipeline

All scripts are idempotent and write to `results/<model_slug>/…` (e.g.
`results/qwen3-1.7b/`). Run from the repo root.

### 1. Sanity check (baseline refusal rates)

```bash
python ayush/sanity_check.py
```

Runs `N_PROMPTS=20` harmful AdvBench prompts and 20 harmless Alpaca prompts
through the model with `enable_thinking=False` and `MAX_NEW_TOKENS=256`.
Classifies each reply with a simple keyword heuristic on the first ~200 chars
after stripping any `<think>…</think>` block. You generally want:

- **Harmful refusal rate**: high (target ≥ 80%); Qwen3-1.7B usually exceeds
  this easily.
- **Harmless refusal rate**: near 0% (false-positive refusals).

Full per-prompt outputs land in `results/qwen3-1.7b/sanity_results.json`.

### 2. Extract the refusal direction

```bash
python ayush/extract_refusal_direction.py
```

Caches residual-stream activations at the last prompt token for the same
harmful/harmless split, computes the difference-of-means direction per layer,
and reports the top-5 layers by a standardised separation score. Outputs:

- `results/qwen3-1.7b/refusal_directions.pt`
- `results/qwen3-1.7b/activations.pt`

### 3. Intervene on the refusal direction

```bash
python ayush/intervene_refusal.py
python ayush/intervene_ablate_long.py   # harmful-ablation with longer generations
```

Ablates the direction across all layers on held-out harmful prompts (expect
refusal rate to drop) and adds it at the best layer on harmless prompts
(expect refusal rate to rise).

### 4. Visualize the steering direction

```bash
python ayush/visualize_refusal_direction.py
python ayush/visualize_steering.py
```

Produces separation-by-layer curves, KDE/histogram plots of the projection
onto the direction, a layer-by-layer cosine heatmap, a 2D PCA view at the
best layer, and a bar chart of refusal rates across intervention conditions
(including the circuit-tracer feature ablation if step 6 has been run).

### 5. Context-length scaling sweep (Phase 2)

```bash
python ayush/sweep_context_scaling.py
python ayush/visualize_scaling.py
```

`sweep_context_scaling.py` iterates over a held-out slice of AdvBench
harmful prompts and, for each bloat length
`N ∈ {0, 128, 512, 1k, 2k, 4k, 8k}` (Willowbrook preamble prepended to the
prompt), captures one hooked forward pass and one greedy generation. It
records, per (prompt, N):

- attention mass from the final input position to every `(layer, head)`
  across the harmful token span (attention dilution, H1),
- residual-stream projection onto `r̂^(ℓ)` at the last harmful token and at
  the final input position (representational dilution, H2),
- the greedy response and keyword-based refusal label (ASR).

Outputs land under `results/<slug>/scaling/`:
`sweep_rows.jsonl` (raw rows) and `summary.csv` (per-N aggregates).
`visualize_scaling.py` then emits `scaling_asr.png`,
`scaling_refusal_projection.png`, and `scaling_attn_dilution.png`
(per-head decay at the best refusal layer, with the top-K heads at N=0
highlighted as candidate guardrail heads).

### 6. Circuit-level attribution with `circuit-tracer`

```bash
python ayush/trace_refusal_circuit.py
```

Loads `Qwen/Qwen3-1.7B` with its pre-trained per-layer transcoders via
`ReplacementModel` and computes attribution graphs for:

1. a held-out harmful prompt the baseline model refuses (pulled from
   `intervention_results.json`), and
2. a matched harmless prompt.

Raw graphs are saved under `results/qwen3-1.7b/graphs/` and pruned JSON
graph files (for the web UI) under `results/qwen3-1.7b/graph_files/`. A
`graph_index.json` records which slug corresponds to which prompt.

Browse the graphs interactively:

```bash
python ayush/visualize_refusal_graph.py --port 8046
# then open http://localhost:8046/index.html
```

### 7. Feature-level intervention

After inspecting the graphs and picking candidate refusal features, record
them in `results/qwen3-1.7b/refusal_features.json`:

```json
{
  "features": [
    {"layer": 18, "pos_from_end": -1, "feature_idx": 12345, "label": "refusal_1"},
    {"layer": 20, "pos_from_end": -1, "feature_idx": 777, "label": "refusal_2"}
  ],
  "value": 0.0
}
```

Then run:

```bash
python ayush/intervene_refusal_features.py
```

This zeros each feature across the final prompt token and every generated
token, reruns the held-out harmful prompts, and writes
`intervention_feature_results.json`. Re-running
`ayush/visualize_steering.py` adds a "harmful / feature ablate" bar to the
summary chart.

## Project layout

```
ayush/
  utils.py                        # model + path constants, helpers
  sanity_check.py                 # baseline refusal rates
  extract_refusal_direction.py    # compute per-layer directions
  intervene_refusal.py            # ablate / add direction, short generations
  intervene_ablate_long.py        # ablate with longer generations
  visualize_refusal_direction.py  # projection + separation plots
  visualize_steering.py           # summary / cosine / PCA / bar-chart figures
  sweep_context_scaling.py        # Phase 2: context-length scaling sweep
  visualize_scaling.py            # Phase 2 plots (ASR / projection / attn dilution)
  trace_refusal_circuit.py        # circuit-tracer attribution graphs
  visualize_refusal_graph.py      # wrapper around circuit-tracer's server
  intervene_refusal_features.py   # zero specific transcoder features
results/<model_slug>/             # all artefacts keyed by model slug
results/<model_slug>/scaling/     # sweep_rows.jsonl + summary.csv + Phase 2 plots
```

## Switching models

`MODEL_NAME`, `MODEL_SLUG`, and `TRANSCODER_SET` live at the top of
`ayush/utils.py`. Switch all four (model, slug, transcoder repo, and — if
needed — `enable_thinking`) in one place.
