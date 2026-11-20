# Building and editing the paper

Two versions share every section and generated file:

| File | Format | Use |
|---|---|---|
| `paper_v2/main.tex` | ICLR 2026 style, one column, author names | arXiv preprint; ICLR 2027 workshops (comment out `\iclrfinalcopy` to anonymize) |
| `paper_v2/main_ndss.tex` | NDSS 2027 template (IEEEtran conference, two columns, US letter), anonymous, STALA block | STALA 2027 @ NDSS submission |

```bash
cd paper_v2
for t in main main_ndss; do pdflatex $t; bibtex $t; pdflatex $t; pdflatex $t; done
```

IEEEtran comes from `texlive-publishers`. Page limits:
- STALA: 8 pages excluding references and appendices. The main text currently ends on page 6–7.
- ICLR-style: 8 pages; the main text ends on page 8.

## Layout

```
sections/abstract.tex
sections/body.tex              intro, contributions, related work, setup (incl. threat model), \input of:
  results_behaviour.tex        Sec. "Short-horizon judges misread..."  (Fig. 1, Table 1)
  results_horizon.tex          Sec. "How far must a judge read..."      (Fig. 2)
  results_robust.tex           larger models (Table), StrongREJECT, framings, pointer to app:robust
  results_mechanism.tex        prompt-only ablation (+held-out, Gemma), prefill, summary of read-outs -> app:mech
  discussion.tex               recommendations (protocol) and limitations
sections/statements.tex        Ethics Considerations + Acknowledgements/Generative-AI disclosure (edit before submitting)
sections/appendix.tex          v1 retrospective, prompts, judging, validity, tables, \input of:
  app_robust.tex               sampling, reasoning mode, benign content
  app_mechanism.tex            probe table, sign flip, same-request control, guardrail head, item-time steering
                               ... then "Analysis decisions made after seeing data" (+ held-out table)
results_steering.tex           not included (older text, kept for reference)
```

The two versions differ only in their preambles. `\codenote` (where the v1 code lives) names the repository
folders in `main.tex` and says "anonymized artifact" in `main_ndss.tex`. In the NDSS version every `table` and
`figure` becomes double-column.

## Numbers come from macros. Do not type them in.

Almost every number is a macro generated from results:
- `generated/macros.tex` comes from `v2/make_tables.py` (main grid, probes, horizon).
- `generated/macros_causal.tex` comes from `v2/analyze_causal.py` (all interventions and robustness).

To refresh after new results:

```bash
AD_OUT=<private results dir> python v2/make_tables.py paper_v2/generated
AD_OUT=<private results dir> python v2/analyze_causal.py paper_v2/generated
git checkout results/v2/summary.json   # if AD_OUT pointed at results/v2 (analyze.py side effect)
```

`generated/fallback.tex` defines any macro a section uses but no generator produced, as a bold **??**. After
compiling, `pdftotext main.pdf - | grep -c '??'` should print 0. If it doesn't, a result is missing or a macro
name changed.

Macro name families:
- `Abl{Model}{Alone|Four|Eight}{None|Prompt|Gen|All}{Ref|ASR}`: prompt-only ablation (E5)
- `Ho{Model}{Alone|Four|Eight|Lists}{None|Prompt}{Ref|ASR}`, `HoQwenInItemShare`: held-out (E13–15)
- `Pf*`, `SOne*`, `Head*`: prefill, single-token steering, head zeroing
- `SR*`, `Seg*`: StrongREJECT, whole response vs segment
- `Samp*`, `Think*`, `Ct*`, `Fr*`: sampling, reasoning, content, framing
- `Big*` (Qwen3-32B), `GemBig*` (Gemma-2-27B), `Huge*` (Llama-3.1-70B)

A few numbers in the prose are still typed in by hand, for example some steering and probe values in
`app_mechanism.tex` and the head-zeroing numbers. If those results change, search for them.

## Before submission

See `SUBMISSION.md`. In short: rewrite the prose in your own voice; edit `statements.tex`; check every
citation in `refs.bib` (entries were verified on 2026-11-20, but check again); make sure there are no
`??` and no "pending" strings; and check that the anonymized PDF contains no names (`pdftotext
main_ndss.pdf - | grep -i -E 'kokande|mishra|nyu|github'`).
