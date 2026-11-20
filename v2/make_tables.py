"""Turn results/v2 judged/probe/steer files into LaTeX tables + macros for the paper. CPU only.

Usage: AD_OUT=results/v2 python v2/make_tables.py paper_v2/generated
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import auc, boot, rj  # noqa: E402  (analyze also regenerates summary + figures)

OUT = Path(os.environ.get("AD_OUT", "results/v2"))
DST = Path(sys.argv[1] if len(sys.argv) > 1 else "paper_v2/generated")
DST.mkdir(parents=True, exist_ok=True)
NICE = {"qwen3-14b": "Qwen3-14B", "gemma2-9b": "Gemma-2-9B", "llama31-8b": "Llama-3.1-8B"}
TAG = {"qwen3-14b": "Qwen", "gemma2-9b": "Gemma", "llama31-8b": "Llama"}
MODELS = [m for m in NICE if (OUT / f"judged_{m}.jsonl").exists()]
macros = {}


def pct(e, ci=True):
    s = f"{100 * e['m']:.0f}"
    return s + (f"\\ci{{{100 * e['lo']:.0f}}}{{{100 * e['hi']:.0f}}}" if ci else "")


def mac(name, val):
    macros[name] = val


J = {m: rj(OUT / f"judged_{m}.jsonl") for m in MODELS}

# ---------------- Table 1: behaviour, numbered template, pooled over k in {3,7,15,31}
lines = [r"\begin{tabular}{llccc}", r"\toprule",
         r"Model & Harmful item & Refusal in first 24 tok. & Target refused & HarmBench ASR \\", r"\midrule"]
for m in MODELS:
    H = [r for r in J[m] if r["is_harmful"]]
    cells = [("alone ($k{=}0$)", [r for r in H if r["k"] == 0])]
    for pos in ("first", "middle", "last"):
        cells.append((f"{pos} in list", [r for r in H if r["template"] == "numbered" and r["k"] >= 3 and r["pos"] == pos]))
    for j, (lab, rows) in enumerate(cells):
        a, b, c = boot(rows, lambda r: r["v1_24"]), boot(rows, lambda r: r["item_ref"]), boot(rows, lambda r: bool(r["hb"]))
        lines.append(f"{NICE[m] if j == 0 else ''} & {lab} & {pct(a)} & {pct(b)} & {pct(c)} \\\\")
        key = {0: "Alone", 1: "First", 2: "Middle", 3: "Last"}[j]
        tag = TAG[m]
        mac(f"{tag}{key}Vone", f"{100 * a['m']:.0f}\\%")
        mac(f"{tag}{key}Ref", f"{100 * b['m']:.0f}\\%")
        mac(f"{tag}{key}ASR", f"{100 * c['m']:.0f}\\%")
    # misclassification: v1 says not refused, but no harmful content
    rows = [r for r in H if r["template"] == "numbered" and r["k"] >= 3 and r["pos"] != "first"]
    e = boot(rows, lambda r: (not r["v1_24"]) and not r["hb"])
    mac(f"{tag}Misread", f"{100 * e['m']:.0f}\\%")
    rows_all = [r for r in H if r["k"] > 0]
    e = boot([r for r in rows_all if not r["v1_24"]], lambda r: not r["hb"])
    mac(f"{tag}NotHarmfulGivenVoneComply", f"{100 * e['m']:.0f}\\%")
    if m != MODELS[-1]:
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}"]
(DST / "tab_behaviour.tex").write_text("\n".join(lines))

# ---------------- Table 2: templates at k=15
lines = [r"\begin{tabular}{ll" + "cc" * len(MODELS) + "}", r"\toprule",
         "Template & Target & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{NICE[m]}}}" for m in MODELS) + r" \\",
         " & & " + " & ".join(["24-tok & ASR"] * len(MODELS)) + r" \\", r"\midrule"]
for t in ("numbered", "bullets", "tasks"):
    for pos in ("first", "last"):
        row = [t if pos == "first" else "", pos]
        for m in MODELS:
            rows = [r for r in J[m] if r["is_harmful"] and r["template"] == t and r["k"] == 15 and r["pos"] == pos]
            row += [pct(boot(rows, lambda r: r["v1_24"]), False), pct(boot(rows, lambda r: bool(r["hb"])), False)]
        lines.append(" & ".join(row) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(DST / "tab_templates.tex").write_text("\n".join(lines))

# ---------------- global macros
for m in MODELS:
    tag = TAG[m]
    B = [r for r in J[m] if not r["is_harmful"]]
    mac(f"{tag}BenignFP", f"{100 * np.mean([r['item_ref'] for r in B if r['k'] > 0]):.1f}\\%")
    mac(f"{tag}BenignFPvone", f"{100 * np.mean([r['v1_24'] for r in B if r['k'] > 0]):.1f}\\%")
    mac(f"{tag}Trunc", f"{100 * np.mean([r['truncated'] for r in J[m]]):.1f}\\%")
    H = [r for r in J[m] if r["is_harmful"] and r["k"] > 0]
    mac(f"{tag}MaxASR", f"{100 * max(np.mean([bool(r['hb']) for r in H if r['k'] == k and r['pos'] == p and r['template'] == t]) for k in (3, 7, 15, 31) for p in ('first', 'middle', 'last') for t in ('numbered',) if any(r['k'] == k and r['pos'] == p and r['template'] == t for r in H)):.0f}\\%")
    loc = [r for r in H if r["template"] == "numbered" and r["k"] >= 3 and r["pos"] != "first" and r["item_ref"]]
    mac(f"{tag}InItemShare", f"{100 * np.mean([r['locus'] == 'in_item' for r in loc]):.0f}\\%")
    mac(f"{tag}UpfrontShare", f"{100 * np.mean([r['locus'] == 'upfront' for r in loc]):.0f}\\%")

# ---------------- Table 3: probes
plines = [r"\begin{tabular}{l" + "c" * len(MODELS) + "}", r"\toprule", "Projection (\\% of harmful-alone value) & " + " & ".join(NICE[m] for m in MODELS) + r" \\", r"\midrule"]
prow = {k: [] for k in ["bare_h", "bare_b", "last_first", "last_last", "onset_ref", "onset_comply", "onset_benign_item", "auc_onset", "auc_last", "neg_last_frac", "neg_last_asr", "pos_last_asr", "attn_last", "attn_onset", "attn_benign_own"]}
for m in MODELS:
    p = OUT / f"probe_{m}.jsonl"
    if not p.exists():
        for k in prow:
            prow[k].append("--")
        continue
    P = {r["rid"]: r for r in rj(p)}
    Jd = {r["rid"]: r for r in J[m]}
    if m == "qwen3-14b":
        L = 36
    else:
        import torch
        L = int(torch.load(OUT / f"dirs_{m}.pt", map_location="cpu")["layer"])
    tag = TAG[m]
    mac(f"{tag}Layer", str(L))
    pr = [(P[i], Jd[i]) for i in P if i in Jd]
    ref_scale = np.mean([a["proj_last"][L] for a, j in pr if j["is_harmful"] and j["k"] == 0])
    mean = lambda v: f"{100 * np.mean(v) / ref_scale:+.0f}" if len(v) else "--"   # % of harmful-alone value
    prow["bare_h"].append(mean([a["proj_last"][L] for a, j in pr if j["is_harmful"] and j["k"] == 0]))
    prow["bare_b"].append(mean([a["proj_last"][L] for a, j in pr if not j["is_harmful"] and j["k"] == 0]))
    prow["last_first"].append(mean([a["proj_last"][L] for a, j in pr if j["is_harmful"] and j["k"] > 0 and j["pos"] == "first"]))
    prow["last_last"].append(mean([a["proj_last"][L] for a, j in pr if j["is_harmful"] and j["k"] > 0 and j["pos"] == "last"]))
    hp = [(a, j) for a, j in pr if j["is_harmful"] and j["k"] > 0]
    prow["onset_ref"].append(mean([a["proj_target"][L] for a, j in hp if "proj_target" in a and j["outcome"] == "REFUSE"]))
    prow["onset_comply"].append(mean([a["proj_target"][L] for a, j in hp if "proj_target" in a and j["outcome"] == "COMPLY"]))
    prow["onset_benign_item"].append(mean([a["proj_benign"][L] for a, j in hp if "proj_benign" in a]))
    nL = len(hp[0][0]["proj_last"])
    aon = [auc([a["proj_target"][l] for a, j in hp if "proj_target" in a and j["outcome"] == "REFUSE"],
               [a["proj_target"][l] for a, j in hp if "proj_target" in a and j["outcome"] == "COMPLY"]) for l in range(nL)]
    ala = [auc([a["proj_last"][l] for a, j in hp if j["outcome"] == "REFUSE"],
               [a["proj_last"][l] for a, j in hp if j["outcome"] == "COMPLY"]) for l in range(nL)]
    ar, al = aon[L], ala[L]
    bo, bl = int(np.nanargmax(aon)), int(np.nanargmax(ala))
    prow["auc_onset"].append(f"{ar:.2f} / {aon[bo]:.2f} ({bo})")
    prow["auc_last"].append(f"{al:.2f} / {ala[bl]:.2f} ({bl})")
    mac(f"{TAG[m]}AUCOnsetBest", f"{aon[bo]:.2f}"); mac(f"{TAG[m]}AUCLastBest", f"{ala[bl]:.2f}")
    (DST / f"auc_layers_{m}.json").write_text(json.dumps(dict(onset=aon, last=ala, L=L)))
    neg = [(a, j) for a, j in hp if a["proj_last"][L] < 0]
    posv = [(a, j) for a, j in hp if a["proj_last"][L] >= 0]
    prow["neg_last_frac"].append(f"{100 * len(neg) / max(1, len(hp)):.0f}\\%")
    prow["neg_last_asr"].append(f"{100 * np.mean([bool(j['hb']) for a, j in neg]):.0f}\\% / {100 * np.mean([j['item_ref'] for a, j in neg]):.0f}\\%" if neg else "--")
    prow["pos_last_asr"].append(f"{100 * np.mean([bool(j['hb']) for a, j in posv]):.0f}\\% / {100 * np.mean([j['item_ref'] for a, j in posv]):.0f}\\%" if posv else "--")
    mac(f"{tag}NegLastFrac", prow["neg_last_frac"][-1])
    mac(f"{tag}AUCOnset", f"{ar:.2f}")
    mac(f"{tag}AUCLast", f"{al:.2f}")
    att = lambda key, sel: f"{np.mean([a[key] for a, j in pr if key in a and sel(j)]):.3f}" if any(key in a and sel(j) for a, j in pr) else "--"
    prow["attn_last"].append(att("attn_last_to_target", lambda j: j["is_harmful"] and j["k"] > 0))
    prow["attn_onset"].append(att("attn_target_to_target", lambda j: j["is_harmful"] and j["k"] > 0))
    prow["attn_benign_own"].append(att("attn_benign_to_own", lambda j: j["is_harmful"] and j["k"] > 0))
labels = {
    "bare_h": r"last prompt token, harmful request alone",
    "bare_b": r"last prompt token, benign request alone",
    "last_first": r"last prompt token, harmful item first",
    "last_last": r"last prompt token, harmful item last",
    "onset_ref": r"harmful item's answer onset, item refused",
    "onset_comply": r"harmful item's answer onset, item complied",
    "onset_benign_item": r"benign item's answer onset",
    "auc_onset": r"AUC refused vs.\ complied, answer onset (sel.\ / best layer)",
    "auc_last": r"AUC refused vs.\ complied, last prompt token (sel.\ / best)",
    "neg_last_frac": r"lists with negative last-token projection",
    "neg_last_asr": r"\quad ASR / target refused, if negative",
    "pos_last_asr": r"\quad ASR / target refused, if positive",
}
for k, lab in labels.items():
    plines.append(lab + " & " + " & ".join(prow[k]) + r" \\")
    if k in ("bare_b", "last_last", "onset_benign_item", "auc_last"):
        plines.append(r"\midrule")
plines += [r"\bottomrule", r"\end{tabular}"]
(DST / "tab_probe.tex").write_text("\n".join(plines))
if "qwen3-14b" in MODELS and (OUT / "probe_qwen3-14b.jsonl").exists():
    mac("AttnLast", prow["attn_last"][0]); mac("AttnOnset", prow["attn_onset"][0]); mac("AttnBenignOwn", prow["attn_benign_own"][0])
    P = [r for r in rj(OUT / "probe_qwen3-14b.jsonl")]
    Jd = {r["rid"]: r for r in J["qwen3-14b"]}
    mac("AttnBare", f"{np.mean([a['attn_last_to_target'] for a in P if Jd[a['rid']]['is_harmful'] and Jd[a['rid']]['k'] == 0]):.3f}")

# ---------------- Table 4: steering
slines = [r"\begin{tabular}{ll" + "cc" * len(MODELS) + "}", r"\toprule",
          r"Where added & $\alpha$ & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{NICE[m]}}}" for m in MODELS) + r" \\",
          r" & & " + " & ".join(["harm.: ASR / ref. & benign: ref."] * len(MODELS)) + r" \\", r"\midrule"]
SR = {m: rj(OUT / f"judged_steer_{m}.jsonl") if (OUT / f"judged_steer_{m}.jsonl").exists() else [] for m in MODELS}
alphas = sorted({r["alpha"] for m in MODELS for r in SR[m]})
for mode in ("none", "item", "all"):
    for a in alphas:
        if (mode == "none") != (a == 0):
            continue
        row = [{"none": "no steering", "item": "decision point onward", "all": "every position"}[mode] if a in (0, alphas[1] if len(alphas) > 1 else 0) else "", f"{a:g}"]
        for m in MODELS:
            h = [r for r in SR[m] if r["is_harmful"] and r["mode"] == mode and r["alpha"] == a]
            b = [r for r in SR[m] if not r["is_harmful"] and r["mode"] == mode and r["alpha"] == a]
            row.append(f"{100 * np.mean([bool(r['hb']) for r in h]):.0f} / {100 * np.mean([r['item_ref'] for r in h]):.0f}" if h else "--")
            row.append(f"{100 * np.mean([r['item_ref'] for r in b]):.0f}" if b else "--")
        slines.append(" & ".join(row) + r" \\")
slines += [r"\bottomrule", r"\end{tabular}"]
(DST / "tab_steer.tex").write_text("\n".join(slines))
for m in MODELS:
    if SR[m]:
        tag = TAG[m]
        h0 = [r for r in SR[m] if r["is_harmful"] and r["mode"] == "none"]
        mac(f"{tag}SteerN", str(len(h0)))
        mac(f"{tag}SteerNComply", str(sum(r.get("base_outcome") == "COMPLY" for r in h0)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fa = [m for m in MODELS if (DST / f"auc_layers_{m}.json").exists()]
if fa:
    fig, axes = plt.subplots(1, len(fa), figsize=(3.3 * len(fa), 2.3), squeeze=False)
    for j, m in enumerate(fa):
        d = json.loads((DST / f"auc_layers_{m}.json").read_text())
        ax = axes[0, j]
        ax.plot(d["last"], color="#d62728", label="last prompt token")
        ax.plot(d["onset"], color="#2ca02c", label="harmful item answer onset")
        ax.axhline(0.5, color="k", lw=0.5); ax.axvline(d["L"], color="k", lw=0.5, ls=":")
        ax.set_ylim(0.3, 1.0); ax.set_title(NICE[m], fontsize=9); ax.set_xlabel("layer")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        if j == 0:
            ax.set_ylabel("AUC: refused vs. complied")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(DST / "fig_auc_layers.pdf")
(DST / "macros.tex").write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())) + "\n")
print(json.dumps(macros, indent=0))
