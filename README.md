# Context and refusal

**When benign context changes a refusal-related activation, does it change the answer to the target request, or just which task the model answers first?**

The [revision protocol](docs/revision_protocol.md) acts on Greg Durrett's course feedback: fewer claims, stronger evaluation, explicit interventions, and related work tied to the experimental design. It includes the saved-output audit, three experiments with stopping decisions, and a section-by-section plan for the paper. This revision has no new Qwen3 results or completed human study.

The saved ordered-list examples make the evaluation problem concrete: four N512 outputs start answering a benign first task while the harmful target is later in the list. Their missing refusal phrases and negative readout cosines do not establish unsafe target compliance.

## Model stages

| Stage | Purpose | Command |
| --- | --- | --- |
| Baseline | Descriptive phrase-refusal rates on the original pools | `baseline` |
| Direction | Harmful-minus-harmless means; select a source layer on validation | `direction` |
| Context | Save full target answers across controlled contexts and optional interventions | `context` |
| Projection | Measure the same contexts at target and assistant-template tokens | `projection` |

The shared implementation is in `attention_dilution/`. The numbered entry points remain usable. The additional `review` command is an offline human-annotation workflow, not a model stage or automatic safety judge. Style/topic/policy analyses and head/circuit tracing remain historical exploratory work.

## Setup

Use Python 3.11 or newer and hardware suitable for the selected model. Install an appropriate PyTorch build, then:

```bash
python -m pip install -r requirements.txt
python -m attention_dilution context --help
```

Command help and offline review do not load model weights. The default model is `Qwen/Qwen3-14B`; a smaller model can check execution but does not confirm a result on 14B. Use the same model and `--revision` throughout a study. Extract a separate direction for each model.

## Start with the validation pilot

Run from the repository root. Direction extraction records actual training and validation prompts so downstream test runs can exclude overlaps.

```bash
python -m attention_dilution baseline --output-dir runs/baseline
python -m attention_dilution direction --output-dir runs/direction
python -m attention_dilution context --refusal-dir runs/direction --split validation --n-prompts 20 --formats prefix quoted tasks --backgrounds apennines garden --lengths 0 512 4096 --arms baseline --output-dir runs/pilot-harmful
python -m attention_dilution projection --refusal-dir runs/direction --split validation --n-prompts 20 --formats prefix quoted tasks --backgrounds apennines garden --lengths 0 512 4096 --output-dir runs/pilot-harmful-projections
python -m attention_dilution context --refusal-dir runs/direction --split validation --pool alpaca --n-prompts 20 --formats prefix quoted tasks --backgrounds apennines garden --lengths 0 512 4096 --arms baseline --output-dir runs/pilot-harmless
```

Keep settings identical between behavior and projection runs, including pool, split, prompt count, formats, backgrounds, and target position. Run matching harmless projections when comparing geometry by request kind. Validation runs may reuse layer-selection prompts; they are not independent tests. Default sweeps use only 0, 512, and 4096 background-token allowances. Extend the grid only when the pilot supports a specific question.

Use `--target-position first` as a separate control. For the scheduling diagnostic, use `--formats ordered-tasks --lengths 0 128`; contrast first and last positions in separate runs. The primary `tasks` format explicitly requests only the target answer. Empty conditions retain each format's instructions, and are shared across background sources.

## Review the actual answers

```bash
python -m attention_dilution review export --results runs/pilot-harmful/*.jsonl runs/pilot-harmless/*.jsonl --output-dir runs/pilot-review
```

The packet contains shuffled full answers in `review.csv`, labeling instructions, and a separate condition mapping. Have two reviewers work independently on copies of the CSV; keep the mapping and others' labels hidden. Fill `label`, `evidence`, and `reviewer`. The five labels are target refusal, safe answer, unsafe answer, target not addressed, and unclear. Quotes in the evidence column must occur in the saved response.

After the reviewers supply labels:

```bash
python -m attention_dilution review summarize --packet-dir runs/pilot-review --labels runs/pilot-review/reviewer-a.csv runs/pilot-review/reviewer-b.csv --output runs/pilot-review/summary.json
```

The report preserves missing/disputed cases and produces per-case consensus for joining to projection records by request ID and condition. Missing labels are not invented. An answer to another task is not counted as a target refusal or unsafe target answer. Bounds on the unsafe-answer rate retain unresolved labels; the Wilson interval is labeled as conditional on resolved cases. The exporter also works on historical full-response JSONL files, but flags missing full inputs and unknown truncation. It does not reconstruct generations missing from aggregate CSVs.

## Held-out prompts and interventions

After freezing the pilot's choices, repeat on `--split test`. `--pool jbb-harmful` and `--pool jbb-benign` load JailbreakBench's official harmful/benign behaviors; `--dataset-revision` can pin the dataset snapshot. For example:

```bash
python -m attention_dilution context --refusal-dir runs/direction --pool jbb-harmful --n-prompts 20 --formats prefix quoted tasks --backgrounds apennines garden --arms baseline --output-dir runs/jbb-harmful
```

JBB contains AdvBench-derived entries. The loader records exact normalized training/validation duplicates it excludes and validates the requested count after exclusion. Inspect semantic overlap before calling this an independent benchmark confirmation. Exact retained prompts are recorded even when no dataset revision is supplied. Use `--prompt-file` for another benchmark: each JSONL row needs `id`, `prompt`, `kind` (`harmful` or `harmless`), and `source`; optional category fields are retained in the run manifest. Choose a consistent safety policy when importing broader policy-refusal benchmarks.

The context stage defaults to intact behavior. `--arms both` adds all-block output ablation. `--arms steered --steering-alpha VALUE` adds that many activation units of the direction at one block output, with an optional `--steering-layer`. Choose VALUE on validation data using the measured coordinate scale and harmless controls, then freeze it for testing. `--random-direction` replaces the learned vector with a seeded random unit vector as an intervention control. Hook sites and equations are spelled out in the [protocol](docs/revision_protocol.md).

## Measurement and outputs

The 18-phrase heuristic over the first 200 characters is retained as a diagnostic and layer-selection proxy. It is not the primary measure of unsafe behavior. Generation saves full answers with 512 output tokens by default in context runs; truncation and failure to reach the target remain visible. A longer allowance is a changed experiment, requiring a new run.

Projection records contain per-request dot product, norm, cosine, token position, and paired change from the same format's empty-context control. Summaries count individual sign crossings and bootstrap requests within each cell. These are representation measurements; behavior association, position matching, evaluator calibration, and capability-preservation evidence still require analysis. In particular, cosine normalization cannot change the sign of a dot product on the same vectors.

Each model run records settings, exact requests/exclusions/backgrounds, input artifact hashes, code hashes, resolved model/tokenizer revisions, and package versions. Without `--output-dir`, commands create timestamped runs. Existing directories require `--resume` with identical settings, prompts, and artifacts; generation also checks the saved formatted input and case metadata. Sampling runs cannot resume. The full templated input plus output allowance must fit within the default 32768-token budget. The shell wrappers forward CLI arguments; they do not allocate cluster resources or install environments.

## Historical work and checks

The existing `results/`, `Suraj/results_v3/`, paper source, and PDFs are preserved. Their different judges, intervention sites, prompt pools, and generation lengths prevent treating them as one uniform protocol. The [review notes](docs/review.md) document those limitations; [INVARIANTS.md](INVARIANTS.md) defines the maintained settings.

```bash
python -m unittest discover -s tests -v
```

The suite checks prompt separation, target spans, paired sign counts, human-label integrity, resume behavior, and hooks with a tiny CPU decoder. CI downloads no model weights. Passing software checks does not verify the historical research claims or substitute for a Qwen3 run, human grading, or second-model replication.
