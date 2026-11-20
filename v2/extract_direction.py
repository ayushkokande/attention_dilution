"""Difference-of-means refusal direction for a new model (same recipe as v1 experiment_2).

d_l = normalize(mean_harmful(h_l[last]) - mean_harmless(h_l[last]))   on splits.json harmful_train/harmless_train
Layer selection: directional ablation of d_l at every layer output on 32 held-out harmful prompts
(splits.json harmful_baseline[:32], disjoint from all E1 prompts); pick the layer with the lowest
post-ablation refusal subject to <=5% refusal on 32 harmless prompts; ties -> deeper layer.

Usage: python v2/extract_direction.py --model llama31-8b
Output: results/v2/dirs_<model>.pt  {d_hat [L,d], norms [L], layer, sweep}
"""
from __future__ import annotations

import argparse

import torch

from common import MODELS, OUT, _load_splits, chat_text, item_refusal
from mech import ablation_hooks, hooks, load_model


@torch.no_grad()
def last_token_acts(model, tok, model_name, prompts, bs=16):
    tok.padding_side = "left"
    acc = None
    for s in range(0, len(prompts), bs):
        texts = [chat_text(tok, model_name, p) for p in prompts[s:s + bs]]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False, return_token_type_ids=False).to(model.device)
        o = model(**enc, output_hidden_states=True, use_cache=False)
        h = torch.stack([x[:, -1].float() for x in o.hidden_states[1:]], 1).sum(0)  # [L, d]
        acc = h if acc is None else acc + h
    return acc / len(prompts)


@torch.no_grad()
def refusal_rate(model, tok, model_name, prompts, hk, bs=16, max_new=64):
    tok.padding_side = "left"
    n = 0
    for s in range(0, len(prompts), bs):
        texts = [chat_text(tok, model_name, p) for p in prompts[s:s + bs]]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False, return_token_type_ids=False).to(model.device)
        with hooks(hk):
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for seq in g:
            n += item_refusal(tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True))
    return n / len(prompts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--hf-path", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    sp = _load_splits()
    model, tok = load_model(args.hf_path or MODELS[args.model]["hf"])
    ntr = 8 if args.smoke else 256
    mu_h = last_token_acts(model, tok, args.model, sp["harmful_train"][:ntr])
    mu_b = last_token_acts(model, tok, args.model, sp["harmless_train"][:ntr])
    diff = mu_h - mu_b
    norms = diff.norm(dim=-1)
    d_hat = diff / norms.unsqueeze(-1)
    L = d_hat.shape[0]
    nsel = 4 if args.smoke else 32
    hsel, bsel = sp["harmful_baseline"][:nsel], sp["harmless_baseline"][:nsel]
    base = refusal_rate(model, tok, args.model, hsel, [])
    sweep = []
    cands = range(int(0.25 * L), int(0.9 * L) + 1, 2)
    for l in cands:
        hk = ablation_hooks(model, d_hat[l].to(model.device))
        rh = refusal_rate(model, tok, args.model, hsel, hk)
        rb = refusal_rate(model, tok, args.model, bsel, hk)
        sweep.append(dict(layer=l, harmful_post=rh, harmless_post=rb))
        print(f"[dir] layer {l}: harmful {rh:.2f} harmless {rb:.2f} (intact harmful {base:.2f})", flush=True)
    ok = [s for s in sweep if s["harmless_post"] <= 0.05] or sweep
    best = min(ok, key=lambda s: (s["harmful_post"], -s["layer"]))
    torch.save(dict(d_hat=d_hat.cpu(), norms=norms.cpu(), layer=best["layer"], sweep=sweep, intact=base),
               OUT / f"dirs_{args.model}.pt")
    print(f"[dir] selected layer {best['layer']}", flush=True)


if __name__ == "__main__":
    main()
