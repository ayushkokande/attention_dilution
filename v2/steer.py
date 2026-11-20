"""E4: item-time activation addition.

Teacher-force the model's own greedy response up to the point where it starts answering the
harmful item, then generate 160 tokens while adding  alpha * (mu_harmful - mu_harmless)_L  at the
output of layer L, either only from the decision point onward ("item") or at every position ("all").
alpha is in units of the natural (unnormalised) difference-of-means vector, unlike v1, which added
alpha * unit(d) with alpha <= 16 (~2.5% of the natural norm 623 at L36 of Qwen3-14B).

Usage: python v2/steer.py --model qwen3-14b [--n 64]
Output: results/v2/gen_steer_<model>.jsonl  (same schema as gen_*.jsonl; score with judge.py)
"""
from __future__ import annotations

import argparse
import json

import torch

from common import MODELS, OUT, REPO, chat_text, find_item_markers, read_jsonl
from mech import addition_hook, hooks, load_model
from probe import content_start

ALPHAS = [0.0, 0.25, 0.5, 1.0]


def raw_vec(model_name: str, L: int) -> torch.Tensor:
    if model_name == "qwen3-14b":
        d = torch.load(REPO / "results/qwen3-14b/refusal_direction/d_hat_all_layers.pt", map_location="cpu")[L]
        norm = json.loads((REPO / "results/qwen3-14b/refusal_direction/meta.json").read_text())["diff_norms_per_layer"][L]
        return d.float() * norm
    blob = torch.load(OUT / f"dirs_{model_name}.pt", map_location="cpu")
    return blob["d_hat"][L].float() * blob["norms"][L]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--hf-path", default=None)
    ap.add_argument("--judged", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    cfg = MODELS[args.model]
    rows = read_jsonl(args.judged or OUT / f"judged_{args.model}.jsonl")
    L = cfg["layer"] if cfg["layer"] is not None else torch.load(OUT / f"dirs_{args.model}.pt")["layer"]

    def eligible(r, harmful):
        return (r["is_harmful"] == harmful and r["template"] == "numbered" and r["k"] == args.k
                and r["pos"] in ("middle", "last") and not r.get("upfront")
                and r["target_item"] in find_item_markers(r["response"], r["n_items"]))
    harm = [r for r in rows if eligible(r, True)]
    # prioritise cases where the unsteered model complied (that is where rescue matters)
    harm.sort(key=lambda r: (r.get("outcome") != "COMPLY", r["rid"]))
    harm = harm[: args.n]
    ben = [r for r in rows if eligible(r, False)][: max(16, args.n // 2)]
    items = harm + ben
    print(f"[steer] {len(harm)} harmful ({sum(r.get('outcome') == 'COMPLY' for r in harm)} complied), {len(ben)} benign; L={L}", flush=True)

    model, tok = load_model(args.hf_path or cfg["hf"])
    tok.padding_side = "left"
    vec = raw_vec(args.model, L)
    out = OUT / f"gen_steer_{args.model}.jsonl"
    done = set()
    for f in [out] + sorted(OUT.glob(f"gen_steer_{args.model}.part*.jsonl")):
        if f.exists():
            done |= {json.loads(l)["rid"] for l in open(f) if l.strip()}
    if args.nshards > 1:
        out = OUT / f"gen_steer_{args.model}.part{args.shard}.jsonl"

    jobs = []
    for r in items:
        marks = find_item_markers(r["response"], r["n_items"])
        prefix = chat_text(tok, args.model, r["user"]) + r["response"][: content_start(r["response"], marks[r["target_item"]])]
        for a in ALPHAS:
            for mode in (["item", "all"] if a > 0 else ["none"]):
                rid = f"{r['rid']}|steer|{mode}|a{a}"
                if rid not in done:
                    jobs.append((rid, r, prefix, a, mode))
    jobs = [j for i, j in enumerate(jobs) if i % args.nshards == args.shard]
    print(f"[steer] {len(jobs)} generations", flush=True)
    for s in range(0, len(jobs), args.bs):
        batch = jobs[s: s + args.bs]
        # one (alpha, mode) per batch keeps the hook simple: group
        groups = {}
        for j in batch:
            groups.setdefault((j[3], j[4]), []).append(j)
        for (a, mode), js in groups.items():
            enc = tok([j[2] for j in js], return_tensors="pt", padding=True, add_special_tokens=False, return_token_type_ids=False).to(model.device)
            T = enc["input_ids"].shape[1]
            start = torch.full((len(js),), T - 1) if mode == "item" else torch.zeros(len(js), dtype=torch.long)
            hk, st = addition_hook(model, L, a * vec, start) if a > 0 else ([], {"offset": 0})
            with torch.no_grad(), hooks(hk):
                g = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False, pad_token_id=tok.pad_token_id)
            with open(out, "a") as f:
                for j, seq in zip(js, g):
                    rid, r, _, _, _ = j
                    ids = seq[T:].tolist()
                    text = tok.decode(ids, skip_special_tokens=True)
                    rec = {k: r[k] for k in ("model", "src", "pidx", "target", "is_harmful", "k", "pos", "template", "user")}
                    rec.update(rid=rid, alpha=a, mode=mode, base_outcome=r.get("outcome"), target_item=1, n_items=1,
                               response=text, first24=tok.decode(ids[:24], skip_special_tokens=True),
                               n_gen=len(ids), finish="length", layer=L)
                    f.write(json.dumps(rec) + "\n")
        print(f"[steer] {min(s + args.bs, len(jobs))}/{len(jobs)}", flush=True)
    print("[steer] done", flush=True)


if __name__ == "__main__":
    main()
