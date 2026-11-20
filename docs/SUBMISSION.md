# Submission plan and checklist

## Venues (checked 2026-11-20)

1. **STALA 2027, the Workshop on Security Testing and Assurance for LLMs and Agents, at NDSS** (primary).
   - **Deadline:** 11 Dec 2026, 23:59 AoE, at https://stala27.hotcrp.com/. Notification 1 Feb 2027,
     camera-ready 19 Feb 2027, workshop 22 Mar 2027.
   - **Format:** full papers up to 8 pages excluding references and appendices (short papers 4); NDSS 2027
     template, two columns, US letter, Times ≥ 10 pt. Double-blind.
   - **Requirements:** a clear threat model; NDSS policies on ethics (optional Ethics Considerations section
     right before the references) and generative-AI disclosure (name the sections and tools when whole
     sections, tables or figures were AI-generated).
   - **Archival:** papers appear in NDSS proceedings.
   - Use `paper_v2/main_ndss.tex`.
2. **ICLR 2027 workshops** on safety, trustworthy ML or interpretability (backup, usually non-archival).
   - Deadlines are typically early February; check the workshop list when it is announced.
   - Use `paper_v2/main.tex` with `\iclrfinalcopy` commented out.
3. **Longer term:** an ACL Rolling Review cycle (full paper), after adding human labels and agentic or
   multi-turn transcripts.

Before relying on these dates, check them against the official pages; they can change.

## Still to do (authors)

- [ ] **Human audit.** 120 prepared responses (40 each COMPLY/REFUSE/OTHER, Qwen3-14B lists). Label whether
      the target was refused and whether the response is harmful; report agreement with the automatic labels.
      Even 60 labelled responses remove the main reviewer objection.
- [ ] **Rewrite the prose in your own voice.** The draft was written with an AI assistant. Keep claims and
      numbers; change the wording.
- [ ] **Edit `paper_v2/sections/statements.tex`** so the generative-AI disclosure matches what you actually
      did.
- [ ] **Verify every reference** in `refs.bib`: authors, year, venue. Remove any you have not read.
- [ ] **Anonymized artifact link** (e.g. anonymous.4open.science), because the GitHub repo shows your names.
      Exclude response text.
- [ ] **Get feedback** from Greg Durrett (and from Suraj, who is a co-author) before submitting.
- [ ] **Final compile:** no `??`, no "pending", no names in `main_ndss.pdf`, main text ≤ 8 pages.
- [ ] **Credentials and pods:** revoke or rotate the RunPod key(s) and HF token used during the runs;
      terminate the pods (`python tools/rp.py list`, then terminate in the RunPod console).

## Likely reviewer questions and where the answers are

| Question | Answer in the paper |
|---|---|
| Is the refusal phrase rule reliable? | Measured false-positive rate on benign lists; target-specific rule; StrongREJECT and HarmBench agreement; human audit (pending) |
| Is this just a weak judge? | It is a judge-horizon problem: Fig. 2 shows how far a judge must read; whole-response judges have a mirror problem (StrongREJECT credits benign answers; fixed by segment-only scoring) |
| Does it hold at scale and for other decoding? | Larger-models table (32B, 27B, 70B); sampling and reasoning mode (appendix) |
| Are lists a jailbreak? | No: framings drive attack success (framing table); list ASR stays low |
| Are the mechanistic claims post hoc? | Preregistered held-out tests (`v2/PREREGISTRATION.md`, commit timestamps) |
| What changed from your course report? | Appendix "What went wrong in our earlier analysis" |
