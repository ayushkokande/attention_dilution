# refusal-mech

Mechanistic analysis of long-context jailbreaks via attention dilution.

## Setup

Use Python **3.11** (see `.python-version`). Create a virtual environment and install dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the sanity check

```bash
python src/sanity_check.py
```

## What the sanity check does

The script loads a small instruct model (`Qwen/Qwen2.5-1.5B-Instruct` by default), runs a fixed number of **harmful** prompts (from AdvBench via CSV) and **harmless** prompts (from `tatsu-lab/alpaca` with empty `input`), and classifies each completion with a simple **keyword-based refusal** heuristic on the start of the reply.

**How to read the output**

- **Harmful refusal rate**: fraction of harmful prompts where the model’s reply looks like a refusal. For a safety-aligned chat model you generally want this **high** (the script flags a target of **≥ 80%**).
- **Harmless refusal rate**: fraction of harmless prompts classified as refusal. You usually want this **near 0%**, since refusals on benign instructions indicate over-refusal or a brittle detector.

This is a **smoke test** for your environment and tokenizer/chat template, not a rigorous safety benchmark. Tune `MODEL_NAME`, `N_PROMPTS`, and `MAX_NEW_TOKENS` at the top of `src/sanity_check.py` as needed. Full per-prompt outputs are written to `results/sanity_results.json`.
