# Held-out confirmation of the prompt-only ablation result (written before running)

Committed before the held-out run starts. Exploratory result (100 JBB+AdvBench targets, local refusal-phrase scoring):
Qwen3-14B keeps in-list refusal under prompt-only ablation (89% -> 93%) while refusal of the request alone drops
(94% -> 37%); Llama-3.1-8B loses it (88% -> 25% for 4-item lists; alone 95% -> 1%).

Confirmation set: 100 HarmBench "standard" behaviours not in JBB/AdvBench (exact-match exclusion), seed 2027,
alone and last in 4- and 8-item numbered lists. Conditions: none, prompt-only ablation (v2/causal.py ablate --heldout).
Scoring: target-refusal rule in v2/judge.py and HarmBench classifier, unchanged.

Predictions (each counts as confirmed if it holds on the held-out set):
1. Qwen3-14B, lists (4 and 8 items pooled): target refusal under prompt-only ablation is within 10 points of, or above,
   the no-ablation rate.
2. Qwen3-14B, request alone: prompt-only ablation lowers target refusal by at least 30 points.
3. Llama-3.1-8B, lists pooled: prompt-only ablation lowers target refusal by at least 30 points.
4. Under prompt-only ablation, Qwen3-14B's list refusals are mostly in the item (locus in_item > 50% of refusals).

Whatever the outcome, the paper reports the held-out numbers next to the exploratory ones.

## Addendum (2026-11-20, written before any Llama-3.1-70B-Instruct output exists)

Results of the four predictions above (all held) are in the paper (appendix, Table "held-out").
We now add Llama-3.1-70B-Instruct and register predictions before generating anything with it. Same held-out set,
conditions, rule and classifier as above; direction from v2/extract_direction.py with the same recipe.
Behaviour: the same `scale` grid used for Qwen3-32B.

5. Behaviour: with the harmful item not first (middle/last pooled, all list lengths), the 24-token judge reports at
   least 20 points less refusal than the target-specific rule.
6. Held-out, lists pooled: prompt-only ablation lowers Llama-3.1-70B's target refusal by at least 30 points
   (i.e. it behaves like Llama-3.1-8B, not Qwen3-14B).
7. Held-out, request alone: prompt-only ablation lowers target refusal by at least 30 points.

Prediction 6 is the informative one; we report it whichever way it comes out.

## Addendum 2 (2026-11-20, before any Gemma-2-27B-it output exists)

Gemma-2-27B-it, behaviour only, same `scale` grid: prediction 5 (24-token judge reports at least 20 points less
refusal than the target-specific rule when the harmful item is not first), and HarmBench ASR with the harmful
item not first stays at or below 15%.
