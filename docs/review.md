# Review and reduced scope

This review covers the default branch at commit `538a6e3446a93c47a3ac680ef5d19f649343b559`.

## Wording and evidence

Style cannot reliably establish who wrote a passage. The repository does contain wording that reads like generated project planning: repeated phase narratives, dramatic labels such as "killer comparison," and conclusions stated more strongly than the checks support. There is also a literal duplicate implementation in the baseline script.

| Earlier framing | More precise framing |
| --- | --- |
| The guardrail heads physically lose attention and cause refusal failure | Measured heads allocate less attention to the request span under particular prompt formats |
| The failure is attentional, not representational | Attention mass, residual projection, and output behavior change differently; their causal relationship needs a controlled test |
| A single upstream lesion defines a two-head circuit | Patching a head changes a downstream projection; an edge-specific circuit claim needs further interventions |
| No refusal marker means the jailbreak succeeded | No marker was found in the inspected response prefix; grade the target response before calling it compliance |
| Every validity check passes | Report each check, its sample construction, and failed preflight conditions separately |

The maintained README and experiment descriptions use the more limited framing. The earlier paper source and PDFs remain historical reports rather than rewritten or newly validated papers.

## Concrete findings

1. **Baseline did not compile.** Two versions of `baseline_benchmark.py` were concatenated, including a second `from __future__` import at line 251. Python raised a SyntaxError before either implementation could run. The refactor keeps one implementation.

2. **A claimed split fix was stale.** The committed direction metadata uses 256 harmless training prompts, corresponding to filtered Alpaca[0:256). Experiment 6's old filtered [200:250) slice is entirely inside it. Future experiment 6 defaults now use [600:650), and historical results retain an explicit warning. The experiment 7 loader previously did not apply the canonical Alpaca filter, so its [100:150) row labels could not be compared directly with the training indices. Its future loader uses the shared filter and [650:700). Experiment 5 now excludes harmless training strings before constructing its plain cell. Old pools and statistics must be rebuilt; changing code does not fix saved results retroactively. Matched directions still need their own exact-string overlap audit.

3. **The distractor metric may score the wrong task.** `Suraj/experiment.py` inserts the target among multiple requests to answer in order. The dense sweep generates only 24 tokens and checks the first 200 characters for refusal. If an earlier request is benign, its answer can look like a refusal failure before the model reaches the target. The current artifacts do not establish that every such output is harmful compliance. The core experiment uses one request after inert text. Multi-request experiments need longer outputs, target-specific grading, and benign task-coverage checks.

4. **Pipeline settings differed.** The numbered scripts used post-block ablation, while the exploratory TransformerLens script removes the direction at resid_pre, resid_mid, and resid_post. Judges, generation lengths, pools, and model-library versions also differed. The new core uses one judge and explicitly names the post-block intervention; older curves are not merged with new ones.

5. **Residual positions were mislabeled.** The last token of a formatted Qwen3 prompt includes the assistant/template suffix; it is not necessarily the last request token. The core projection stage measures and labels both positions.

6. **Forward-only runs created unnecessary logits.** Calling the full causal language model allocates vocabulary logits for every input token. Direction and projection measurements now call the base decoder and keep only selected residuals.

7. **Output reuse lacked a configuration check.** A changed model, direction, or decoding setting could be written into the same summary directory. New core runs use separate directories and fingerprint checks before resuming.

8. **The policy preflight failed.** The saved policy pool has a 44% phrase-refusal rate against a 90% gate. Its AUC still describes separation between those prompt pools, but it does not by itself isolate harmful refusal from another kind of refusal. This experiment is outside the core scope.

## What remains in the study

The follow-up in [revision_protocol.md](revision_protocol.md) supersedes the earlier prefix-only recommendation. Keep the four model stages, but use a small set of explicit-target contexts, full answer review, and paired representation measurements. An inert prefix remains a control. Target scheduling must be separated from target unsafe behavior before an intervention claim.

Greg Durrett's feedback identifies the sign change as a possible focus, conditional on systematic evidence. Calibrated steering and a second-model confirmation are conditional follow-ups, rather than a broad survey or automatic next phase. Remove the style/topic/policy battery and two-head interpretation from the main argument until their individual claims have adequate support.

The refactor does not claim a new experimental result. Historical curves remain available, and the new protocol needs fresh model runs before its outputs can replace them.
