# Exploratory pipeline

`experiment.py` contains the earlier multi-format, steering, attribution, and activation-patching experiments. The checked-in `results_v3/` includes Qwen3-14B runs. The previous README described a different Qwen3-1.7B run and referenced figures that were absent from this directory.

This script is outside the maintained four-stage study in the [root README](../README.md). Its outputs remain available as historical evidence and ideas for follow-up experiments.

Before using its results in a new report:

- Generate and grade the response to the target request. The distractor format inserts the target among several requests, while the dense sweep generates only 24 tokens and judges the response prefix.
- Compare exactly the same prompt pools, generation length, phrase judge, model revision, and intervention sites.
- Treat node-patching recovery as a measurement of influence on the chosen residual projection. It does not by itself verify a specific edge between two attention heads.
- Measure attention during the relevant answer position and compare with position-matched controls before attributing a change to attention normalization.

The original planning document is preserved under `docs/history/Suraj_PLAN.md`. Dependencies for this script are separate in this directory. No existing result files have been recomputed by the core refactor.
