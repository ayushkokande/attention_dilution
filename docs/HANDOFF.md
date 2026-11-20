# Handoff: everything a new collaborator or AI session needs (no chat history required)

This file records the goals, preferences, constraints and state that were agreed in conversation while v2
was built (19–20 Nov 2026). Together with the other files in `docs/`, it should be enough to continue,
re-scope or submit the work without access to that conversation. If something you need is not written down
in the repository, ask the authors; do not infer it.

## People and roles

- **Authors:** Ayush Kokande and Suraj Mishra (NYU). Suraj remains a co-author on v2.
- **Instructor:** Greg Durrett graded v1 (74/80). His written feedback is on page 1 of
  `Attention_Dilution_Graded.pdf`; `HISTORY.md` maps each point to v2. The authors plan to send him the
  draft before submitting.
- v2's code, experiments, analysis and first-draft text were produced with an AI assistant (Claude). The
  authors review results, rewrite the prose, run the human audit, and are responsible for the content. The
  paper's AI-use statement (`paper_v2/sections/statements.tex`) must reflect this accurately.

## Goals and preferences the authors stated

- **Direction:** AI safety + interpretability. Measurement-first: the judge-horizon finding is the core,
  and mechanism supports it.
- **Ambition:** a workshop paper is acceptable ("get it out of the way"). Primary target: **STALA 2027 @ NDSS**
  (deadline 11 Dec 2026). Backup: ICLR 2027 workshops. See `SUBMISSION.md`; for other venues, see
  `SCOPING_OTHER_VENUES.md`.
- **Results:** keep all results, but keep the main text focused; side results belong in the appendices.
- **Honesty over spin.** Report preregistered outcomes whichever way they go, keep the "what went wrong in
  our earlier analysis" appendix, and do not overclaim (see "Interpretation guardrails" in `HISTORY.md`).
- **Larger models are welcome**, even for a workshop paper (Qwen3-32B, Gemma-2-27B and Llama-3.1-70B were
  added for this reason).
- **Everything documented** so future work does not depend on chat history (this folder).

## Constraints

- **Compute:** a personal RunPod account, pay-as-you-go. About 2× A100-80GB at $4.38/hr per pod. Balance was
  about $98 on the morning of 20 Nov 2026. Always set an auto-stop (`tools/deadman.sh`) and stop pods as soon
  as results are downloaded.
- **Model access:** HF token with Llama-3.1 (8B, 70B) and Gemma-2 (9B, 27B) access. Llama-3.3-70B is not
  approved.
- **Safety of data:** model responses can be harmful. Keep them out of git and out of any public link;
  publish labels only (`v2/export_labels.py`).
- **Credentials:** never commit them. Keys live in `~/.secrets/` on the controller and `/root/.rp/key` on
  pods. The authors should revoke or rotate the RunPod keys and HF token used during the Nov 2026 runs.

## State at handoff (update this section when things change)

- **Branch:** `v2-deferred-refusal` on GitHub (`ayushkokande/attention_dilution`). Not yet merged to `main`.
- **Paper:** `paper_v2/main.tex` (preprint) and `paper_v2/main_ndss.tex` (STALA, anonymous). Both compile;
  numbers come from `paper_v2/generated/`.
- **Finished experiments:** E1–E13 and E16 (Qwen3-32B, Gemma-2-27B behaviour). See `EXPERIMENTS.md`.
- **In flight on 20 Nov 2026:**
  - E14 (Gemma-2-27B preregistered ablation);
  - E15 (Llama-3.1-70B behaviour + preregistered ablation).

  If their results are not in `paper_v2/generated/` (no `HoGemmaBig*` or `HoLlamaHuge*` macros, or
  `\HUGESENTENCE` still empty), they were not completed. Either rerun them (`v2/run_70b.sh`,
  `v2/run_27b_mech.sh`, `v2/run_70b_mech.sh`; ~4–6 pod-hours) or remove the 70B/27B ablation mentions and
  keep the behaviour results.
- **Private data:** full judged files and the human-audit sheet (`human_audit_qwen3-14b.csv`, 120 rows) were
  delivered to the authors as zips (`attention_dilution_v2_fulltext*.zip`). They also exist on the pods under
  `/workspace/results_v2/` until the pods are terminated. Analysis scripts expect them in a directory passed as
  `AD_OUT`.
- **Authors' open tasks:** see the checklist in `SUBMISSION.md` (human audit, rewrite, AI statement,
  citation check, anonymized artifact, Greg's review, credentials, pods).

## How to resume work in a new AI session

Give the session this repository (branch `v2-deferred-refusal`) and point it to `docs/README.md` and this
file. Tell it the current goal (e.g. "prepare the STALA submission" or "re-scope for venue X"), any change in
budget or ambition, and whether compute is available. It should then:
1. read `HISTORY.md` (claims and guardrails) and `EXPERIMENTS.md` (what exists);
2. for new runs, follow `RUNBOOK.md` and preregister confirmation predictions in `v2/PREREGISTRATION.md`
   before running them;
3. regenerate tables and macros rather than typing numbers (`PAPER.md`);
4. update the "State at handoff" section above when done.
