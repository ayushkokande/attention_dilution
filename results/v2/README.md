# v2 results (labels only)

This public folder holds labels, probe measurements and aggregates, but **no model response text**. Some
responses contain harmful content: they are kept privately, outside the repository. The labels are enough to
recompute every number in the paper except the segmentation-based fields, which are already stored.

| File | Contents |
|---|---|
| `labels_<model>.jsonl.gz` | one row per prompt cell: ids, design factors, `v1_24`, `item_ref`, `locus`, `upfront`, `seg_how`, `hb` (HarmBench), `outcome` |
| `labels_steer_<model>.jsonl` | item-time steering labels |
| `probe_<model>.jsonl` | per-layer projections and guardrail-head attention at three positions |
| `dirs_<model>.pt` | difference-of-means directions, norms, and the layer-selection sweep (Gemma, Llama) |
| `summary.json` | aggregated numbers with bootstrap CIs |
| `figures/` | behaviour and probe figures |

To rerun `v2/make_tables.py` you need the full `judged_*.jsonl` files, which are available from the authors.
