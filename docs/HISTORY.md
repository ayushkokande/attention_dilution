# History and decisions

## v1: course project (May 2026)

"Attention Dilution: a mechanistic analysis of context-length jailbreaks" (Kokande & Mishra, NYU; graded
74/80 by Greg Durrett). Claims on Qwen3-14B:
- only the numbered-list ("distractor") format collapses refusal, at about 31 tokens;
- guardrail head L36H31 stops attending to the harmful request;
- the refusal-direction projection "flips sign";
- steering cannot restore refusal.

Code: `experiment_1..9/`, `Suraj/`. Report: `Attention_Dilution_Submission.pdf`.

Instructor feedback (Greg Durrett), as written in `Attention_Dilution_Graded.pdf` (page 1), and how v2
responds to each point. Re-read the original before claiming the paper "addresses Greg's feedback".

| Feedback (paraphrased; see the PDF for the exact wording) | v2 response | Status |
|---|---|---|
| Quantity over quality: for a real paper, one weak analysis undermines the argument | One core claim (judge horizon); mechanism only in support; side results moved to appendices; claims we could not support were retracted (planning, head necessity, v1 collapse) | done |
| Parts read as LLM-generated: several hypotheses analysed, conclusions asserted with confidence | Preregistered confirmations; controls that overturned our own readings; analysis decisions listed. **The prose must still be rewritten by the authors** | partly (authors) |
| Show prompt examples; ideally a human study | Prompt examples in the appendix; 120-response human audit prepared | audit pending (authors) |
| Which analyses were used in prior work? Risk of "rolling your own" | Methods follow published work: difference-of-means direction and ablation (Arditi et al.), HarmBench classifier, StrongREJECT, shallow-alignment prefill (Qi et al.), SequentialBreak-style framings | done |
| What exactly does "ablating d-hat" mean, and what degradation should we expect? | Defined explicitly as x ← x − (x·d̂)d̂ at every layer; expectations stated as preregistered predictions | done |
| Related work mostly 2024 or earlier | Related work updated with 2025–2026 papers (refusal dynamics during generation, planning probes, multi-turn and embedded jailbreaks) | done; authors should re-verify citations |
| Broader, more up-to-date prompts (e.g. JailbreakBench) | JailbreakBench + AdvBench; held-out HarmBench behaviours for confirmation | done |
| Beyond a single distractor format, model and setup; the "format-specific failure" conclusion was not supported | 3 list templates, 4 adversarial framings, 4 kinds of benign content, 6 models (8–70B), sampling and reasoning mode; the format-specific conclusion was dropped | done |
| The direction flip is interesting if shown systematically and examined more thoroughly | Measured on all list prompts and models; shown to be real but not a decision (same-request-alone control) | done |

## v2: what went wrong in v1, and what replaced it

Three v1 design choices produced the "collapse":
1. **24-token substring judge.** Refusals deferred to the item are invisible to it.
2. **Uncontrolled position of the harmful item.** It was hash-random in the list, so "list" was confounded
   with "harmful item not first".
3. **Steering with a unit vector at α ≤ 16.** That is about 2.6% of the natural scale of the
   difference-of-means vector, so the negative result was uninformative.

v2 controls position (first / middle / last), judges whole responses, and adds models, templates, judges
and causal tests. The paper's appendix "What went wrong in our earlier analysis" tells this story; keep it.
Reviewers respond well to it.

## Decisions made after seeing data

All of these are listed in the paper (appendix "Analysis decisions made after seeing data").

- **Refusal rule.** "Phrase anywhere in the response" gave 18% false positives on all-benign Gemma lists, so
  it became target-specific (preamble, the target's segment, or a skipped target). The rule was fixed before
  any HarmBench judging or mechanistic analysis.
- **Segmentation.** Added Task/Level/Question/bold-number markers and an up-front-refusal check, after nested
  numbered lists inside benign answers were mistaken for item markers.
- **Same-request-alone control.** Added after high last-token AUCs looked like "planning". It overturned that
  reading: the projection of the same request asked alone predicts the list outcome about as well.
- **Attention dilution.** Attention at the item's answer onset does fall with list length (0.027 to 0.014),
  but ASR does not rise with it. The paper states both.

## Confirmations (preregistered; see `v2/PREREGISTRATION.md`)

Each prediction was committed to git before its run started. Check the commit timestamps.

| Commit | Predictions | Outcome |
|---|---|---|
| 9006e74 | 1–4: Qwen3-14B keeps in-list refusal under prompt-only ablation; Llama-3.1-8B loses it (100 held-out HarmBench behaviours, seed 2027) | **all held** (Qwen lists 74→82%, alone 81→40%; Llama lists 74→12%) |
| 513da53 | 5–7: Llama-3.1-70B shows the judge gap (5) and behaves like Llama-3.1-8B under prompt-only ablation (6, 7) | 5 **held** (24-tok 15% vs 74%; ASR 8%); 6 **failed**; 7 **held** |
| b49248e | Gemma-2-27B shows the judge gap, ASR ≤ 15% | **held** (24-tok 33% vs 95% target refusal; ASR 1%) |
| f2f0a7f | 8–9: Gemma-2-27B (refuses mostly up front) loses list refusal under prompt-only ablation | **inconclusive (twice)**: prompt-only ablation broke generation at layer 41 (97% repetitive) and again at layer 21 after the coherence check was added. A naive reading (99% → 0% refusal) would have falsely confirmed it. |
| 5d85a48 | 10: Llama-3.1-70B refuses at the item (unlike the 8B), so the locus hypothesis predicts it keeps list refusal under prompt-only ablation; this conflicts with family-based prediction 6 | **10 held, 6 failed, 7 held**: lists 66→72%, alone 83→46% (ASR 16→43%) |

## Interpretation guardrails (what the paper does *not* claim)

- It does not claim lists are a jailbreak. Adversarial framings drive embedded-prompt attack success (E11).
- It does not claim the last-token read-out is a "plan". The same-request control shows it reflects the
  request.
- It does not claim prompt-only ablation works on every model. On Gemma-2-27B it breaks generation, so that
  test is inconclusive. Always check ablated outputs for coherence (`degenerate()` in `extract_direction.py`).
- It does not claim L36H31 is necessary for refusal. Zeroing it and three control heads changes nothing.
- For Qwen3-14B, ablating d̂ removes refusal *phrasing* but rarely produces harmful content (ASR ≤ 7%). The
  mechanism claims are about where refusal is decided, not where harm is permitted.
- The refusal phrase list misses some warning-style refusals ("Note: this is illegal…"). Refusal rates are
  therefore lower bounds; HarmBench ASR is the conservative safety number.
