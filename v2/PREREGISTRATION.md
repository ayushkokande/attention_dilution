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

Gemma-2-27B result (behaviour): both predictions above held (24-token 33% vs target-specific 95% refusal with the item
not first; ASR 1%). Unlike Gemma-2-9B, its list refusals are mostly up front (59% upfront, 36% in the item).

## Addendum 3 (2026-11-20, before any Gemma-2-27B direction or ablation run)

Our claim is that where a model refuses predicts where its decision is carried. Gemma-2-27B refuses mostly up
front, so on the same held-out set and conditions as above:

8. Held-out, lists pooled: prompt-only ablation lowers Gemma-2-27B's target refusal by at least 30 points.
9. Held-out, request alone: prompt-only ablation lowers target refusal by at least 30 points.

If 8 fails, the link between refusal locus and decision locus is weaker than we claim, and the paper says so.

## Addendum 4 (2026-11-20 ~16:15 UTC, after Llama-3.1-70B behaviour, before any 70B direction or ablation run)

Llama-3.1-70B behaviour: prediction 5 held (24-token 15% vs target-specific 74% refusal with the item not first;
ASR 8%). Unexpectedly, its list refusals are mostly at the item (74% in_item, 24% upfront), unlike Llama-3.1-8B
(mostly upfront). Prediction 6 (registered before any 70B output) assumed it would behave like its 8B sibling.
Our locus hypothesis ("where a model refuses predicts where it carries the decision") now predicts the opposite:

10. Held-out, lists pooled: prompt-only ablation lowers Llama-3.1-70B's target refusal by **less than 30 points**
    (it behaves like Qwen3-14B, not Llama-3.1-8B). Predictions 6 and 10 cannot both hold; prediction 6 stays as
    registered and both are reported. Prediction 7 (alone drops >= 30 points) is unchanged.

## Gemma-2-27B result and amendment to direction selection (2026-11-20 ~16:25 UTC, before the 70B direction run)

Predictions 8–9 (Gemma-2-27B) are **inconclusive, not confirmed**. The selected layer (41 of 46) satisfied the
selection rule because ablating it produced repetitive, incoherent text, which the refusal check scores as "not
refusing". Under prompt-only ablation, all 300 held-out responses ran to the length limit with repetitive text
(97% flagged by the degeneracy check below; 0% without ablation). Refusal and ASR both read 0%, which is
uninterpretable.

Amendment, applied before the Llama-3.1-70B direction is extracted: layer selection now also requires ≤ 10%
degenerate outputs (fewer than half of the word 4-grams distinct) on the harmful and harmless selection prompts
under ablation (`v2/extract_direction.py`). The same check finds 0% degenerate outputs in Llama-3.1-8B's
prompt-only-ablated held-out responses, so the earlier confirmations (1–4) are unaffected. Predictions 6, 7 and 10
for Llama-3.1-70B stand as registered.
