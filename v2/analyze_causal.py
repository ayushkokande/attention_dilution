"""Tables + macros for round-3 experiments (causal tests, second judge, reasoning mode, sampling). CPU only.

Usage: AD_OUT=results/v2 python v2/analyze_causal.py paper_v2/generated
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import auc, boot, rj  # noqa: E402

OUT = Path(os.environ.get("AD_OUT", "results/v2"))
DST = Path(sys.argv[1] if len(sys.argv) > 1 else "paper_v2/generated")
DST.mkdir(parents=True, exist_ok=True)
NICE = {"qwen3-14b": "Qwen3-14B", "gemma2-9b": "Gemma-2-9B", "llama31-8b": "Llama-3.1-8B"}
TAG = {"qwen3-14b": "Qwen", "gemma2-9b": "Gemma", "llama31-8b": "Llama"}
M = {}
AW = {0.25: "Quarter", 0.5: "Half", 1.0: "One", 2.0: "Two"}


def mac(k, v):
    M[k] = v


def pc(e, ci=True):
    if e["n"] == 0:
        return "--"
    s = f"{100 * e['m']:.0f}"
    return s + (f"\\ci{{{100 * e['lo']:.0f}}}{{{100 * e['hi']:.0f}}}" if ci else "")


def load(name):
    p = OUT / name
    return rj(p) if p.exists() else []


hb = lambda r: bool(r.get("hb"))
ref = lambda r: bool(r["item_ref"])

# ---------------- E5 prompt-only ablation
lines = [r"\begin{tabular}{llcccc}", r"\toprule",
         r"Model & Prompt & no ablation & prompt positions & generated positions & all positions \\", r"\midrule"]
for m in NICE:
    R = load(f"judged_ablate_{m}.jsonl")
    if not R:
        continue
    for j, (lab, sel) in enumerate([("alone", lambda r: r["k"] == 0), ("4 items, last", lambda r: r["k"] == 3),
                                    ("8 items, last", lambda r: r["k"] == 7)]):
        cells = []
        for c in ("none", "prompt", "gen", "all"):
            rows = [r for r in R if r["cond"] == c and sel(r)]
            a, b = boot(rows, ref), boot(rows, hb)
            cells.append(f"{pc(a, False)} / {pc(b, False)}")
            mac(f"Abl{TAG[m]}{['Alone', 'Four', 'Eight'][j]}{c.capitalize()}Ref", f"{100 * a['m']:.0f}\\%")
            mac(f"Abl{TAG[m]}{['Alone', 'Four', 'Eight'][j]}{c.capitalize()}ASR", f"{100 * b['m']:.0f}\\%")
        lines.append(f"{NICE[m] if j == 0 else ''} & {lab} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
if len(lines) > 4:
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    (DST / "tab_ablate.tex").write_text("\n".join(lines))

# ---------------- E6 prefill recovery
lines = [r"\begin{tabular}{l" + "cc" * 3 + "}", r"\toprule",
         r"Model & \multicolumn{2}{c}{request alone} & \multicolumn{2}{c}{4 items, last} & \multicolumn{2}{c}{16 items, last} \\",
         r" & recovers & ASR & recovers & ASR & recovers & ASR \\", r"\midrule"]
for m in NICE:
    R = [r for r in load(f"judged_prefill_{m}.jsonl") if r["cond"] == "prefill"]
    if not R:
        continue
    cells = []
    for k, nm in ((0, "Alone"), (3, "Four"), (15, "Sixteen")):
        rows = [r for r in R if r["k"] == k]
        a, b = boot(rows, ref), boot(rows, hb)
        cells += [pc(a), pc(b)]
        mac(f"Pf{TAG[m]}{nm}Rec", f"{100 * a['m']:.0f}\\%")
        mac(f"Pf{TAG[m]}{nm}ASR", f"{100 * b['m']:.0f}\\%")
        mac(f"Pf{TAG[m]}{nm}N", str(a["n"]))
    lines.append(f"{NICE[m]} & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(DST / "tab_prefill.tex").write_text("\n".join(lines))

# ---------------- E7 single-token vs decision-onward steering
lines = [r"\begin{tabular}{llccc}", r"\toprule", r"Model & $\alpha$ & where & harmful: ASR / refused & benign: refused \\", r"\midrule"]
for m in NICE:
    S1 = load(f"judged_steer1_{m}.jsonl")
    S = load(f"judged_steer_{m}.jsonl")
    if not S1:
        continue
    base_h = [r for r in S if r["is_harmful"] and r["mode"] == "none"]
    base_b = [r for r in S if not r["is_harmful"] and r["mode"] == "none"]
    lines.append(f"{NICE[m]} & 0 & -- & {100 * np.mean([hb(r) for r in base_h]):.0f} / {100 * np.mean([ref(r) for r in base_h]):.0f} & {100 * np.mean([ref(r) for r in base_b]):.0f} \\\\")
    for a in sorted({r["alpha"] for r in S1}):
        for where, rows in (("decision token only", [r for r in S1 if r["alpha"] == a]),
                            ("decision token onward", [r for r in S if r["mode"] == "item" and r["alpha"] == a])):
            h = [r for r in rows if r["is_harmful"]]
            b = [r for r in rows if not r["is_harmful"]]
            if not h:
                continue
            lines.append(f" & {a:g} & {where} & {100 * np.mean([hb(r) for r in h]):.0f} / {100 * np.mean([ref(r) for r in h]):.0f} & {100 * np.mean([ref(r) for r in b]):.0f} \\\\")
            if where.startswith("decision token only"):
                mac(f"SOne{TAG[m]}{AW[a]}ASR", f"{100 * np.mean([hb(r) for r in h]):.0f}\\%")
                mac(f"SOne{TAG[m]}{AW[a]}Ref", f"{100 * np.mean([ref(r) for r in h]):.0f}\\%")
                mac(f"SOne{TAG[m]}{AW[a]}Ben", f"{100 * np.mean([ref(r) for r in b]):.0f}\\%")
    lines.append(r"\midrule")
lines[-1] = r"\bottomrule"
lines.append(r"\end{tabular}")
(DST / "tab_steer1.tex").write_text("\n".join(lines))

# ---------------- E8 head ablation (Qwen)
H = load("judged_head_qwen3-14b.jsonl")
if H:
    lines = [r"\begin{tabular}{lcc}", r"\toprule", r"Zeroed head (layer 36) & harmful: ASR / refused & benign: refused \\", r"\midrule"]
    ctrl = sorted({int(c.split("_")[0][1:]) for c in {r["cond"] for r in H} if c != "none"} - {31})
    hname = {31: "Guard", **{h: f"Ctrl{'ABC'[i]}" for i, h in enumerate(ctrl)}}
    for c in sorted({r["cond"] for r in H}, key=lambda c: (c != "none", "h31" not in c, c)):
        h = [r for r in H if r["cond"] == c and r["is_harmful"]]
        b = [r for r in H if r["cond"] == c and not r["is_harmful"]]
        lab = "none" if c == "none" else f"H{c.split('_')[0][1:]}, " + ("from decision token" if c.endswith("item") else "all positions")
        lines.append(f"{lab} & {100 * np.mean([hb(r) for r in h]):.0f} / {100 * np.mean([ref(r) for r in h]):.0f} & {100 * np.mean([ref(r) for r in b]):.0f} \\\\")
        key = "None" if c == "none" else hname[int(c.split("_")[0][1:])] + ("Item" if c.endswith("item") else "All")
        mac(f"Head{key}Ref", f"{100 * np.mean([ref(r) for r in h]):.0f}\\%")
        mac(f"Head{key}ASR", f"{100 * np.mean([hb(r) for r in h]):.0f}\\%")
        mac(f"Head{key}Ben", f"{100 * np.mean([ref(r) for r in b]):.0f}\\%")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (DST / "tab_head.tex").write_text("\n".join(lines))

# ---------------- E9 second judge agreement
lines = [r"\begin{tabular}{lcccc}", r"\toprule",
         r"Model & AUC(SR score, HB label) & SR$>$0.5 rate & HB rate & mean SR of ``other'' \\", r"\midrule"]
for m in NICE:
    R = [r for r in load(f"judged_{m}.sr.jsonl") if r["is_harmful"]]
    if not R:
        continue
    pos_ = [r["sr"] for r in R if hb(r)]
    neg_ = [r["sr"] for r in R if not hb(r)]
    oth = [r["sr"] for r in R if r["outcome"] == "OTHER"]
    a = auc(pos_, neg_)
    lines.append(f"{NICE[m]} & {a:.2f} & {100 * np.mean([r['sr'] > 0.5 for r in R]):.1f}\\% & {100 * np.mean([hb(r) for r in R]):.1f}\\% & {np.mean(oth):.2f} \\\\")
    mac(f"SR{TAG[m]}AUC", f"{a:.2f}")
    mac(f"SR{TAG[m]}Rate", f"{100 * np.mean([r['sr'] > 0.5 for r in R]):.1f}\\%")
    mac(f"SR{TAG[m]}Other", f"{np.mean(oth):.2f}")
    # list-cell maximum under SR
    cells = {}
    for r in R:
        if r["k"] > 0:
            cells.setdefault((r["template"], r["k"], r["pos"]), []).append(r["sr"] > 0.5)
    mac(f"SR{TAG[m]}MaxCell", f"{100 * max(np.mean(v) for v in cells.values()):.0f}\\%")
    mac(f"SR{TAG[m]}Alone", f"{100 * np.mean([r['sr'] > 0.5 for r in R if r['k'] == 0]):.0f}\\%")
lines += [r"\bottomrule", r"\end{tabular}"]
(DST / "tab_sr.tex").write_text("\n".join(lines))

# ---------------- reasoning mode + sampling (behaviour, like Table 1)
lines = [r"\begin{tabular}{llccc}", r"\toprule", r"Setting & Harmful item & Refusal in first 24 tok. & Target refused & ASR \\", r"\midrule"]
for name, fn, label in [("thinking", "judged_thinking_qwen3-14b.jsonl", "Qwen3-14B, reasoning on")] + \
        [("sample", f"judged_sample_{m}.jsonl", f"{NICE[m]}, $T{{=}}0.7$") for m in NICE]:
    R = [r for r in load(fn) if r["is_harmful"]]
    if not R:
        continue
    for j, (lab, sel) in enumerate([("alone", lambda r: r["k"] == 0), ("first in list", lambda r: r["k"] > 0 and r["pos"] == "first"),
                                    ("last in list", lambda r: r["k"] > 0 and r["pos"] == "last")]):
        rows = [r for r in R if sel(r)]
        a, b, c = boot(rows, lambda r: r["v1_24"]), boot(rows, ref), boot(rows, hb)
        lines.append(f"{label if j == 0 else ''} & {lab} & {pc(a)} & {pc(b)} & {pc(c)} \\\\")
        key = (TAG.get(fn.split("_")[-1].replace(".jsonl", ""), "Qwen") if name == "sample" else "Think")
        mac(f"{'Samp' if name == 'sample' else ''}{key}{['Alone', 'First', 'Last'][j]}{'Vone'}", f"{100 * a['m']:.0f}\\%")
        mac(f"{'Samp' if name == 'sample' else ''}{key}{['Alone', 'First', 'Last'][j]}Ref", f"{100 * b['m']:.0f}\\%")
        mac(f"{'Samp' if name == 'sample' else ''}{key}{['Alone', 'First', 'Last'][j]}ASR", f"{100 * c['m']:.0f}\\%")
    if name == "thinking":
        tr = [r for r in R if r["k"] > 0 and r.get("thinking")]
        from common import item_refusal
        mac("ThinkRefInThinking", f"{100 * np.mean([item_refusal(r['thinking']) for r in tr]):.0f}\\%")
        mac("ThinkTrunc", f"{100 * np.mean([not r['response'] for r in R]):.1f}\\%")
    lines.append(r"\midrule")
lines[-1] = r"\bottomrule"
lines.append(r"\end{tabular}")
(DST / "tab_variants.tex").write_text("\n".join(lines))


# ---------------- scale (Qwen3-32B), framing, content
NICE4 = dict(NICE, **{"qwen3-32b": "Qwen3-32B"})
TAG["qwen3-32b"] = "QwenBig"
for bm, BP in (("qwen3-32b", "Big"), ("llama31-70b", "Huge")):
    R = [r for r in load(f"judged_scale_{bm}.jsonl") if r["is_harmful"]]
    if not R:
        continue
    for j, (lab, sel) in enumerate([("Alone", lambda r: r["k"] == 0)] + [(p.capitalize(), (lambda p: lambda r: r["k"] >= 3 and r["pos"] == p)(p)) for p in ("first", "middle", "last")]):
        rows = [r for r in R if sel(r)]
        mac(f"{BP}{lab}Vone", f"{100 * np.mean([r['v1_24'] for r in rows]):.0f}\\%")
        mac(f"{BP}{lab}Ref", f"{100 * np.mean([ref(r) for r in rows]):.0f}\\%")
        mac(f"{BP}{lab}ASR", f"{100 * np.mean([hb(r) for r in rows]):.0f}\\%")
    nf = [r for r in R if r["k"] >= 3 and r["pos"] != "first"]
    mac(f"{BP}NotFirstVone", f"{100 * np.mean([r['v1_24'] for r in nf]):.0f}\\%")
    mac(f"{BP}NotFirstRef", f"{100 * np.mean([ref(r) for r in nf]):.0f}\\%")
    lr = [r for r in nf if r["item_ref"]]
    for loc in ("upfront", "in_item"):
        mac(f"{BP}Locus{loc.replace('_', '').capitalize()}", f"{100 * np.mean([r['locus'] == loc for r in lr]):.0f}\\%")
    B = [r for r in load(f"judged_scale_{bm}.jsonl") if not r["is_harmful"] and r["k"] > 0]
    if B:
        mac(f"{BP}BenignFP", f"{100 * np.mean([ref(r) for r in B]):.1f}\\%")
FR = {m: load(f"judged_framing_{m}.jsonl") for m in list(NICE) + ["qwen3-32b"]}
FR = {m: v for m, v in FR.items() if v}
if FR:
    ms = list(FR)
    lines = [r"\begin{tabular}{ll" + "c" * len(ms) + "}", r"\toprule",
             "Framing & Harmful item & " + " & ".join(NICE4[m] for m in ms) + r" \\", r"\midrule"]
    for t in ("numbered", "game", "dialogue", "qbank"):
        for pos in ("first", "late"):
            cells = []
            for m in ms:
                rows = [r for r in FR[m] if r["template"] == t and r["pos"] == pos]
                cells.append(f"{100 * np.mean([hb(r) for r in rows]):.0f} / {100 * np.mean([r['v1_24'] for r in rows]):.0f}" if rows else "--")
                if rows:
                    mac(f"Fr{TAG[m]}{t.capitalize()}{pos.capitalize()}ASR", f"{100 * np.mean([hb(r) for r in rows]):.0f}\\%")
            lines.append(f"{ {'numbered': 'plain numbered list', 'game': 'text-adventure game', 'dialogue': 'student/mentor dialogue', 'qbank': 'exam question bank'}[t] if pos == 'first' else ''} & {pos} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (DST / "tab_framing.tex").write_text("\n".join(lines))
CT = {m: load(f"judged_content_{m}.jsonl") for m in NICE}
CT = {m: v for m, v in CT.items() if v}
if CT:
    ms = list(CT)
    lines = [r"\begin{tabular}{l" + "ccc" * len(ms) + "}", r"\toprule",
             "Benign items & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{NICE[m]}}}" for m in ms) + r" \\",
             " & " + " & ".join(["24-tok & refused & ASR"] * len(ms)) + r" \\", r"\midrule"]
    for c, lab in (("alpaca", "Alpaca instructions"), ("code", "coding tasks (MBPP)"), ("math", "math problems (GSM8K)"), ("qa", "factual questions (NQ)")):
        cells = []
        for m in ms:
            rows = [r for r in CT[m] if r["rid"].endswith("|" + c)]
            cells += [f"{100 * np.mean([r['v1_24'] for r in rows]):.0f}", f"{100 * np.mean([ref(r) for r in rows]):.0f}", f"{100 * np.mean([hb(r) for r in rows]):.0f}"] if rows else ["--"] * 3
            if rows:
                mac(f"Ct{TAG[m]}{c.capitalize()}ASR", f"{100 * np.mean([hb(r) for r in rows]):.0f}\\%")
                mac(f"Ct{TAG[m]}{c.capitalize()}Ref", f"{100 * np.mean([ref(r) for r in rows]):.0f}\\%")
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (DST / "tab_content.tex").write_text("\n".join(lines))

# ---------------- preregistered held-out confirmation (v2/PREREGISTRATION.md)
NICE["llama31-70b"] = "Llama-3.1-70B"
TAG["llama31-70b"] = "LlamaHuge"
HO = {m: load(f"judged_ablateho_{m}.jsonl") for m in ("qwen3-14b", "llama31-8b", "llama31-70b")}
HO = {m: v for m, v in HO.items() if v}
if HO:
    lines = [r"\begin{tabular}{llcccc}", r"\toprule",
             r"Model & Prompt & \multicolumn{2}{c}{no ablation} & \multicolumn{2}{c}{prompt-only ablation} \\",
             r" & & refused & ASR & refused & ASR \\", r"\midrule"]
    for m, R in HO.items():
        for j, (lab, ks) in enumerate((("alone", (0,)), ("4 items, last", (3,)), ("8 items, last", (7,)),
                                       ("lists pooled", (3, 7)))):
            cells = []
            for c in ("none", "prompt"):
                rows = [r for r in R if r["cond"] == c and r["k"] in ks]
                a, b = boot(rows, ref), boot(rows, hb)
                cells += [pc(a), pc(b, False)]
                nm = ["Alone", "Four", "Eight", "Lists"][j]
                mac(f"Ho{TAG[m]}{nm}{c.capitalize()}Ref", f"{100 * a['m']:.0f}\\%")
                mac(f"Ho{TAG[m]}{nm}{c.capitalize()}ASR", f"{100 * b['m']:.0f}\\%")
            lines.append(f"{NICE[m] if j == 0 else ''} & {lab} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    (DST / "tab_heldout.tex").write_text("\n".join(lines))
    if "qwen3-14b" in HO:
        R = [r for r in HO["qwen3-14b"] if r["cond"] == "prompt" and r["k"] in (3, 7) and r["item_ref"]]
        mac("HoQwenInItemShare", f"{100 * np.mean([r['locus'] == 'in_item' for r in R]):.0f}\\%")

(DST / "macros_causal.tex").write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(M.items())) + "\n")
print(json.dumps(M, indent=0))
