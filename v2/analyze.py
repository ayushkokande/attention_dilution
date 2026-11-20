"""Aggregate v2 results into tables (JSON + LaTeX snippets) and figures. CPU only.

Usage: AD_OUT=results/v2 python v2/analyze.py
"""
from __future__ import annotations

import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("AD_OUT", Path(__file__).resolve().parent.parent / "results" / "v2"))
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
MODELS = [m for m in ("qwen3-14b", "gemma2-9b", "llama31-8b") if (OUT / f"judged_{m}.jsonl").exists()]
NICE = {"qwen3-14b": "Qwen3-14B", "gemma2-9b": "Gemma-2-9B-it", "llama31-8b": "Llama-3.1-8B-Instruct"}
LAYER = {"qwen3-14b": 36}
rng = np.random.default_rng(0)


def rj(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def boot(rows, f, key=lambda r: (r["src"], r["pidx"]), B=1000):
    """Point estimate + 95% CI with prompts as the resampling unit."""
    if not rows:
        return dict(m=float("nan"), lo=float("nan"), hi=float("nan"), n=0)
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(f(r))
    g = [np.array(v, float) for v in groups.values()]
    sums, cnts = np.array([x.sum() for x in g]), np.array([len(x) for x in g])
    m = sums.sum() / cnts.sum()
    idx = rng.integers(0, len(g), (B, len(g)))
    bs = sums[idx].sum(1) / cnts[idx].sum(1)
    return dict(m=float(m), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), n=int(cnts.sum()))


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1.0
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


S = {"models": MODELS}

# ------------------------------------------------------------------ E1 behaviour
for m in MODELS:
    J = rj(OUT / f"judged_{m}.jsonl")
    H = [r for r in J if r["is_harmful"]]
    B = [r for r in J if not r["is_harmful"]]
    s = S.setdefault(m, {})
    s["n_rows"] = len(J)
    s["seg_rate"] = boot([r for r in H if r["n_items"] > 1], lambda r: r["seg_how"] != "unsegmented")
    s["trunc_rate"] = boot(H, lambda r: r["truncated"])
    grid = {}
    for k in sorted({r["k"] for r in H}):
        for pos in ("first", "middle", "last", "any"):
            rows = [r for r in H if r["template"] == "numbered" and r["k"] == k and (pos == "any" or r["pos"] == pos)]
            if not rows:
                continue
            grid[f"k{k}_{pos}"] = dict(
                v1_24=boot(rows, lambda r: r["v1_24"]),
                v1_full=boot(rows, lambda r: r["v1_full"]),
                item_refuse=boot(rows, lambda r: r["outcome"] == "REFUSE"),
                asr=boot(rows, lambda r: bool(r.get("hb"))),
                other=boot(rows, lambda r: r["outcome"] == "OTHER"),
                v1_says_comply_but_refused=boot(rows, lambda r: (not r["v1_24"]) and r["outcome"] == "REFUSE"),
            )
    s["grid"] = grid
    tmpl = {}
    for t in ("numbered", "bullets", "tasks"):
        for k in (3, 15):
            for pos in ("first", "last"):
                rows = [r for r in H if r["template"] == t and r["k"] == k and r["pos"] == pos]
                if rows:
                    tmpl[f"{t}_k{k}_{pos}"] = dict(v1_24=boot(rows, lambda r: r["v1_24"]),
                                                   item_refuse=boot(rows, lambda r: r["outcome"] == "REFUSE"),
                                                   asr=boot(rows, lambda r: bool(r.get("hb"))),
                                                   seg=boot(rows, lambda r: r["seg_how"] != "unsegmented"))
    s["templates"] = tmpl
    ben = {}
    for k in (0, 3, 15):
        for pos in ("first", "last"):
            rows = [r for r in B if r["k"] == k and (k == 0 or r["pos"] == pos)]
            if rows:
                ben[f"k{k}_{pos}"] = dict(v1_24=boot(rows, lambda r: r["v1_24"]),
                                          item_refuse=boot(rows, lambda r: r["outcome"] == "REFUSE"))
    s["benign_control"] = ben
    # by source
    s["by_src"] = {src: dict(asr_k15_last=boot([r for r in H if r["src"] == src and r["template"] == "numbered" and r["k"] == 15 and r["pos"] == "last"], lambda r: bool(r.get("hb"))),
                             refuse_k0=boot([r for r in H if r["src"] == src and r["k"] == 0], lambda r: r["outcome"] == "REFUSE"))
                   for src in ("jbb", "advbench")}

    # v1 position-only model check: v1_24 refusal ~ P(target first) * base
    s["v1_position_check"] = {f"k{k}": dict(
        v1_24_first=boot([r for r in H if r["template"] == "numbered" and r["k"] == k and r["pos"] == "first"], lambda r: r["v1_24"]),
        v1_24_notfirst=boot([r for r in H if r["template"] == "numbered" and r["k"] == k and r["pos"] != "first"], lambda r: r["v1_24"]))
        for k in sorted({r["k"] for r in H}) if k > 0}

# ------------------------------------------------------------------ E3 probes
for m in MODELS:
    p = OUT / f"probe_{m}.jsonl"
    if not p.exists():
        continue
    P = {r["rid"]: r for r in rj(p)}
    J = {r["rid"]: r for r in rj(OUT / f"judged_{m}.jsonl")}
    L = LAYER.get(m)
    if L is None:
        import torch
        L = int(torch.load(OUT / f"dirs_{m}.pt", map_location="cpu")["layer"])
    s = S[m]
    s["probe_layer"] = L
    pr = {}
    for (harm, k, pos) in [(h, k, pos) for h in (True, False) for k in (0, 3, 15) for pos in ("first", "last")]:
        rows = [(P[i], J[i]) for i in P if i in J and J[i]["is_harmful"] == harm and J[i]["k"] == k and J[i]["pos"] == pos and J[i]["template"] == "numbered"]
        if not rows:
            continue
        d = {}
        for where in ("last", "target", "benign"):
            v = [a[f"proj_{where}"][L] for a, _ in rows if f"proj_{where}" in a]
            if v:
                d[where] = dict(mean=float(np.mean(v)), sd=float(np.std(v)), frac_pos=float(np.mean(np.array(v) > 0)), n=len(v))
            for att in (f"attn_{where}_to_target",):
                w = [a[att] for a, _ in rows if att in a]
                if w:
                    d[att] = dict(mean=float(np.mean(w)), sd=float(np.std(w)), n=len(w))
        w = [a["attn_benign_to_own"] for a, _ in rows if "attn_benign_to_own" in a]
        if w:
            d["attn_benign_to_own"] = dict(mean=float(np.mean(w)), sd=float(np.std(w)), n=len(w))
        pr[f"{'H' if harm else 'B'}_k{k}_{pos}"] = d
    s["probe"] = pr
    # does the target-position projection predict the item outcome?  (harmful, k>0)
    hp = [(P[i], J[i]) for i in P if i in J and J[i]["is_harmful"] and J[i]["k"] > 0 and "proj_target" in P[i]]
    ref = [a["proj_target"][L] for a, j in hp if j["outcome"] == "REFUSE"]
    com = [a["proj_target"][L] for a, j in hp if j["outcome"] == "COMPLY"]
    lref = [a["proj_last"][L] for a, j in hp if j["outcome"] == "REFUSE"]
    lcom = [a["proj_last"][L] for a, j in hp if j["outcome"] == "COMPLY"]
    s["probe_predicts_outcome"] = dict(auc_target=auc(ref, com), auc_last=auc(lref, lcom), n_refuse=len(ref), n_comply=len(com))
    # what the last-token readout does predict: whether the response OPENS with a refusal (v1's 24-token label)
    allh = [(P[i], J[i]) for i in P if i in J and J[i]["is_harmful"] and J[i]["k"] > 0]
    o_ref = [a["proj_last"][L] for a, j in allh if j["v1_24"]]
    o_non = [a["proj_last"][L] for a, j in allh if not j["v1_24"]]
    # among responses that do NOT open with a refusal, can the last-token readout tell later item refusal from compliance?
    d_ref = [a["proj_last"][L] for a, j in allh if not j["v1_24"] and j["outcome"] == "REFUSE"]
    d_com = [a["proj_last"][L] for a, j in allh if not j["v1_24"] and j["outcome"] == "COMPLY"]
    t_ref = [a["proj_target"][L] for a, j in allh if not j["v1_24"] and j["outcome"] == "REFUSE" and "proj_target" in a]
    t_com = [a["proj_target"][L] for a, j in allh if not j["v1_24"] and j["outcome"] == "COMPLY" and "proj_target" in a]
    s["probe_last_vs_opening"] = dict(auc_opening_refusal=auc(o_ref, o_non), n_open_ref=len(o_ref), n_open_non=len(o_non),
                                      deferred_auc_last=auc(d_ref, d_com), deferred_auc_target=auc(t_ref, t_com),
                                      n_deferred_refuse=len(d_ref), n_deferred_comply=len(d_com))
    # harmful-vs-benign target separation at target position (k>0)
    bt = [P[i]["proj_target"][L] for i in P if i in J and not J[i]["is_harmful"] and J[i]["k"] > 0 and "proj_target" in P[i]]
    ht = [a["proj_target"][L] for a, j in hp]
    hl = [a["proj_last"][L] for a, j in hp]
    bl = [P[i]["proj_last"][L] for i in P if i in J and not J[i]["is_harmful"] and J[i]["k"] > 0]
    s["probe_harm_vs_benign"] = dict(auc_target=auc(ht, bt), auc_last=auc(hl, bl))
    # layer profiles for the figure
    prof = {}
    for name, sel in [("H_target", lambda j: j["is_harmful"] and j["k"] > 0), ("B_target", lambda j: (not j["is_harmful"]) and j["k"] > 0)]:
        arr = np.array([P[i]["proj_target"] for i in P if i in J and sel(J[i]) and "proj_target" in P[i]])
        if len(arr):
            prof[name] = arr.mean(0).tolist()
    for name, sel in [("H_last_first", lambda j: j["is_harmful"] and j["k"] > 0 and j["pos"] == "first"),
                      ("H_last_last", lambda j: j["is_harmful"] and j["k"] > 0 and j["pos"] == "last"),
                      ("H_bare", lambda j: j["is_harmful"] and j["k"] == 0),
                      ("B_bare", lambda j: (not j["is_harmful"]) and j["k"] == 0)]:
        arr = np.array([P[i]["proj_last"] for i in P if i in J and sel(J[i])])
        if len(arr):
            prof[name] = arr.mean(0).tolist()
    s["probe_profiles"] = prof

# ------------------------------------------------------------------ E4 steering
for m in MODELS:
    p = OUT / f"judged_steer_{m}.jsonl"
    if not p.exists():
        continue
    R = rj(p)
    st = {}
    for harm in (True, False):
        for mode in ("none", "item", "all"):
            for a in sorted({r["alpha"] for r in R}):
                rows = [r for r in R if r["is_harmful"] == harm and r["mode"] == mode and r["alpha"] == a]
                if not rows:
                    continue
                st[f"{'H' if harm else 'B'}_{mode}_a{a}"] = dict(
                    refuse=boot(rows, lambda r: r["item_ref"]),
                    asr=boot(rows, lambda r: bool(r.get("hb"))) if harm else None,
                    n_base_comply=sum(r.get("base_outcome") == "COMPLY" for r in rows))
    S[m]["steer"] = st

(OUT / "summary.json").write_text(json.dumps(S, indent=1))
print(json.dumps({m: {k: v for k, v in S[m].items() if k in ("seg_rate", "trunc_rate", "probe_predicts_outcome", "probe_harm_vs_benign")} for m in MODELS}, indent=1))

# ------------------------------------------------------------------ figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
C = {"first": "#1f77b4", "middle": "#9467bd", "last": "#d62728"}
if MODELS:
    fig, axes = plt.subplots(2, len(MODELS), figsize=(3.3 * len(MODELS), 4.6), squeeze=False, sharey=True)
    for j, m in enumerate(MODELS):
        g = S[m]["grid"]
        ks = sorted({int(x.split("_")[0][1:]) for x in g})
        for row, (metric, title) in enumerate([("v1_24", "v1 metric: refusal in first 24 tokens"),
                                               ("asr", "HarmBench ASR on the harmful item")]):
            ax = axes[row, j]
            for pos in ("first", "middle", "last"):
                xs, ys, lo, hi = [], [], [], []
                for k in ks:
                    key = f"k{k}_{pos}" if k > 0 else "k0_first"
                    if key in g:
                        e = g[key][metric]
                        xs.append(k + 1); ys.append(e["m"]); lo.append(e["lo"]); hi.append(e["hi"])
                ax.plot(xs, ys, "o-", color=C[pos], label=f"target {pos}", ms=3)
                ax.fill_between(xs, lo, hi, color=C[pos], alpha=0.15)
            ax.set_xscale("log", base=2)
            ax.set_ylim(-0.02, 1.02)
            if row == 1:
                ax.set_xlabel("items in list (k+1)")
            if j == 0:
                ax.set_ylabel(title, fontsize=8)
            if row == 0:
                ax.set_title(NICE[m])
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig_behaviour.pdf")
    fig.savefig(FIG / "fig_behaviour.png", dpi=200)

    pm = [m for m in MODELS if "probe_profiles" in S[m]]
    if pm:
        fig, axes = plt.subplots(1, len(pm), figsize=(3.3 * len(pm), 2.6), squeeze=False)
        for j, m in enumerate(pm):
            ax = axes[0, j]
            pr = S[m]["probe_profiles"]
            sty = {"H_bare": ("k", "-", "harmful, bare (last prompt tok)"), "B_bare": ("gray", "-", "benign, bare"),
                   "H_last_last": (C["last"], "--", "harmful last in list: last prompt tok"),
                   "H_last_first": (C["first"], "--", "harmful first in list: last prompt tok"),
                   "H_target": ("#2ca02c", "-", "harmful item: answer onset"),
                   "B_target": ("#8c564b", ":", "benign item: answer onset")}
            for k_, (c, ls, lab) in sty.items():
                if k_ in pr:
                    ax.plot(pr[k_], color=c, ls=ls, label=lab, lw=1.2)
            ax.axhline(0, color="k", lw=0.5)
            ax.axvline(S[m]["probe_layer"], color="k", lw=0.5, ls=":")
            ax.set_title(NICE[m]); ax.set_xlabel("layer")
            if j == 0:
                ax.set_ylabel(r"$\langle h_\ell, \hat d_\ell\rangle$")
        axes[0, 0].legend(frameon=False, fontsize=6)
        fig.tight_layout()
        fig.savefig(FIG / "fig_probe.pdf")
        fig.savefig(FIG / "fig_probe.png", dpi=200)

# ------------------------------------------------------------------ human-audit sheet
for m in MODELS[:1]:
    J = [r for r in rj(OUT / f"judged_{m}.jsonl") if r["is_harmful"] and r["k"] > 0]
    random.Random(0).shuffle(J)
    strata = defaultdict(list)
    for r in J:
        if len(strata[r["outcome"]]) < 40:
            strata[r["outcome"]].append(r)
    with open(OUT / f"human_audit_{m}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rid", "harmful_request", "target_item", "segment_or_response", "auto_outcome", "human_refused(1/0)", "human_harmful(1/0)"])
        for r in sum(strata.values(), []):
            w.writerow([r["rid"], r["target"], r["target_item"], (r["segment"] or r["response"])[:3000], r["outcome"], "", ""])
print("figures ->", FIG)
