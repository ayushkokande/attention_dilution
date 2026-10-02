# Core experiment settings

The maintained study has four stages: baseline, direction extraction, inert-prefix context sweep, and residual projection sweep. The commands are in [README.md](README.md).

- Default model: `Qwen/Qwen3-14B`. Smoke runs may use another Qwen3 model, but direction and evaluation model revisions must match.
- Chat template: one user message; `enable_thinking=False` unless explicitly requested throughout a run.
- Refusal scoring: the same 18-phrase, first-200-character heuristic in `attention_dilution/shared.py`. A missing refusal phrase is not a verified harmful answer.
- Generation: 256 output tokens, greedy decoding by default. The same judge is used for layer selection and behavioral evaluation.
- Directions: difference of harmful and harmless means at each decoder block's output, at the last token of the full chat template.
- Intervention: remove a selected unit direction at each decoder block's output. This differs from the exploratory three-site TransformerLens intervention.
- Selected layer: read it from the freshly extracted `meta.json`; do not hard-code historical L36.
- Splits: read the `core` section of `splits.json`, validate its ranges, and compare the actual training, validation, and test prompt strings.
- Context: one unchanged inert filler passage, followed by the request. Default filler lengths are 0, 128, 512, 1024, 2048, 4096, 8192, and 16384.
- Budget: the full templated input plus the output allowance must fit inside the 32768-token default budget. 32768 filler tokens alone do not fit.
- Measurements: distinguish the last request token from the last chat-template token. Projection runs measure both.
- Outputs: new runs belong in `runs/`, with input prompts, settings, artifact hashes, and model/package versions. An explicit resume must match the saved configuration.
- Claims: report observed behavior and projections. These measurements alone do not establish a particular attention mechanism or a two-head circuit.

The existing `results/`, `Suraj/results_v3/`, and paper PDFs are historical artifacts. Their different pools, judges, interventions, and generation lengths are documented in [docs/review.md](docs/review.md). They must not be relabeled as results of this refactor.
