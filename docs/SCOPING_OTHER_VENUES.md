# Re-scoping the paper for other venues

This guide is self-contained: together with `HANDOFF.md` (goals, preferences, constraints, state) and the
other files in `docs/`, it is all you need. No conversation history is required. If a decision depends on
something not written in the repository (for example a new budget, a co-author's wishes, or a venue's
current rules), get it from the authors or the venue's official call; do not assume it.

What the current paper is: a **measurement paper with a supporting mechanistic analysis**, sized for a
workshop. Short-horizon judges misread multi-request prompts. Where a model refuses predicts where it carries
its decision. There are preregistered held-out confirmations. Six open models, 8–70B.

This file lists what each kind of venue would additionally expect, what that costs, and in what order to do
it. Check each venue's current call (page limits, template, anonymity, dual-submission and archival rules)
before committing. Those details change every year and are deliberately not hard-coded here.

## 1. What any stronger venue will ask for (shared backlog, highest value first)

| # | Addition | Why reviewers want it | Effort / compute | Notes |
|---|---|---|---|---|
| A | **Human audit** of the automatic labels (≥ 120 responses, two annotators, Cohen's κ) | Every number rests on automatic judges | 1–2 days of annotation, no GPU | 120 responses are already selected (40 COMPLY / REFUSE / OTHER). Label "target refused?" and "harmful?". Stratify across models next time. |
| B | **Agentic and multi-turn settings**: benign "items" as tool outputs, retrieved documents, or earlier conversation turns | The main external-validity gap; agents are where multi-request inputs occur in practice | 1 week of engineering + ~10–20 GPU-h | Reuse `common.py` list builders; add a transcript builder. Judges must find the harmful sub-task, so reuse segmentation and `srjudge.py --segment`. |
| C | **Closed models via API** (behaviour + judge horizon only) | "Does this matter for frontier models?" | Engineering ~2 days; API cost ~$100–300 for ~5k responses per model | No mechanism possible; report the 24-token vs full-response gap and the locus. Check each provider's usage policy for harmful prompts first. |
| D | **A released judging protocol or tool**: segment the response, then judge the target segment | Turns the finding into a contribution others can use | ~3 days | E9 already shows segment-only StrongREJECT matches HarmBench. Package it with tests and a small benchmark of labelled multi-request responses (from A). |
| E | **Judge comparison at scale**: LlamaGuard-3/4, WildGuard, GPT-4-class judges, keyword judges, each at several horizons | Measurement venues expect more than two judges | ~10 GPU-h + API cost | Plot each judge's horizon curve as in Fig. 2. |
| F | **Mitigation**: train for item-level refusal (SFT or DPO on multi-request data), or run the protocol as a monitor | Security and ML venues like a fix | 1–2 weeks; ~50–100 GPU-h for 8B LoRA runs | Report over-refusal on benign lists. |
| G | **Mechanism beyond one direction**: multiple refusal directions or concept cones, SAE features, more held-out models | Interpretability main tracks will press here | 1–3 weeks; ~30–60 GPU-h | Extend the preregistered ablation to every model; add causal mediation at the item. |
| H | **Statistics clean-up**: cluster bootstrap by behaviour, mixed-effects model for position × length × model | Pooled intervals are currently a bit narrow (noted in the appendix) | 1 day, CPU | `analyze.py` already bootstraps; switch to cluster resampling by `pidx`. |

The compute figures are on 2× A100-80GB at about $4.4/hr per pod. The whole current paper used roughly
25–30 pod-hours.

## 2. Venue families and what to add

### Security venues: workshop at NDSS/CCS/USENIX, or main track (USENIX Security, CCS, IEEE S&P, NDSS, SaTML)
- **Framing:** evaluation validity. A security claim ("model X resists embedded-prompt attacks") is unsupported
  when the judge is short-horizon. Lead with the threat model; it is already in Setup.
- **Workshop (current target STALA):** the paper as is, plus A.
- **Main track:** A + B + C + D are close to required. F or a strong E strengthens it. The longer page limits
  (often 12–13 pages plus appendices) fit the material now in the appendix. Expect questions about
  responsible disclosure: the paper introduces no new attack, but say so explicitly (Ethics Considerations).
- **Format:** usually two-column with venue-specific templates and mandatory ethics and open-science
  sections. Reuse `main_ndss.tex` as the starting point.

### ML main tracks (NeurIPS, ICLR, ICML)
- **Framing:** either (i) evaluation methodology, where NeurIPS has a Datasets & Benchmarks–style track and the
  contribution would be D + A + E as a benchmark of multi-request safety judging; or (ii) interpretability
  ("refusal decisions are re-derived at the item in some models and inherited in others"), which needs G and
  preregistered tests on more models.
- **Minimum:** A + B or C + E + H for (i); A + G + H for (ii).
- **Format:** 9–10 pages, single column. `main.tex` (ICLR style) is the starting point; switch the style file
  for NeurIPS or ICML.
- **Reviewer risks:** "only lists", "only open models ≤ 70B", "only automatic judges". The shared backlog above
  targets exactly these.

### NLP venues (ACL, EMNLP, NAACL via ACL Rolling Review)
- **Framing:** evaluation of LLM safety judges and long-form generation analysis. Fits *Findings* or the
  main conference with A + E + D, and ideally C.
- **Format:** 8 pages long / 4 short, plus an unlimited Limitations section and an Ethics statement (the ARR
  checklist). The current limitations and ethics text is a good base.
- **Notes:** ARR cycles have fixed deadlines and a commitment step. Preprints are allowed, but check the
  current anonymity-period rules.

### Workshops (ML or NLP; often non-archival)
- **Fit:** the current paper as is. Interpretability-focused workshops: lead with Sec. mechanism and keep the
  measurement part short. Evaluation or safety workshops: keep the current order.
- Non-archival workshops usually allow later submission to a main track; archival ones (with proceedings)
  usually do not allow resubmitting substantially the same paper elsewhere.

## 3. How to re-scope the manuscript quickly

1. **Pick the story.** Measurement-first (current) or mechanism-first. The sections are modular
   (`results_*.tex`, `app_*.tex`), so reordering means changing the `\input` order in `body.tex` and
   rewriting the intro and contributions.
2. **Page budget.** With more space, move back from the appendix, in this order:
   - the probe/same-request control (`app_mechanism.tex`);
   - the content-type table;
   - sampling and reasoning;
   - steering.
   With less space (4-page short paper), keep only Table 1, Fig. 2, the larger-models table and one paragraph
   on prompt-only ablation.
3. **Template.** Copy `main.tex` or `main_ndss.tex` and change only the preamble. Everything else is shared,
   and numbers come from `generated/` macros (see `PAPER.md`).
4. **Anonymity.** Switch `\codenote`, remove the author block and acknowledgements, and link an anonymized
   artifact.
5. **New results.** Add each experiment as its own `results_*.tex` or `app_*.tex` file with a generated table.
   Preregister confirmation runs in `v2/PREREGISTRATION.md` before running them, as before. This was the most
   persuasive part of the current paper.

## 4. Things that would change the paper's claims (watch for these)

- If the **human audit** shows the phrase rule misses many refusals, refusal rates go up; the claim is
  unaffected. If it shows many false refusals, the claim weakens. Report it either way.
- If **agentic or API** results show a different locus pattern (e.g. frontier models always refuse up front),
  the judge-horizon problem becomes model-specific. That is still a finding, but the framing changes.
- If more models break the link between **refusal locus and decision locus** (the Gemma-2-27B test, E14, is
  the first check of this), the mechanistic claim must be narrowed to "model-dependent" without the locus
  link.
