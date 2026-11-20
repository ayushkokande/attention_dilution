"""Causal tests of the paper's mechanistic claims.

  ablate   (E5) Is the item-level refusal "planned" in prompt-position states?  Directional ablation of d-hat
                at every layer, restricted to prompt positions / generated positions / all / none, then free
                greedy generation of the full response to a list prompt.
  prefill  (E6) Is safety deeper than the first tokens of the item's answer?  Teacher-force the model's own
                response up to the harmful item, then force a compliant opening (JailbreakBench "Target"
                string) and let it continue (cf. Qi et al., 2025). Same for the request alone.
  steer1   (E7) Item-time steering at a SINGLE position (the decision token) instead of from it onward.
  head     (E8) Zero the guardrail head (Qwen3-14B L36H31) from the decision token onward, or at every
                position; control = three other heads in the same layer.

All outputs use the gen_*.jsonl schema so judge.py / srjudge.py can score them.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from common import MODELS, OUT, REPO, chat_text, find_item_markers, max_new_tokens, read_jsonl
from mech import hooks, layers, load_model
from probe import content_start, load_dirs


# ----------------------------------------------------------------------------- position-aware hooks
class Pos:
    """Tracks absolute (padded) positions of the current forward call via cache_position."""

    def __init__(self, model):
        self.p = None
        self.base = model.model

    def pre(self, mod, args, kwargs):
        cp = kwargs.get("cache_position")
        ids = kwargs.get("input_ids", args[0] if args else None)
        emb = kwargs.get("inputs_embeds")
        T = (ids if ids is not None else emb).shape[1]
        self.p = cp if cp is not None else torch.arange(T)

    def handle(self):
        return [(self.base, self.pre, "pre")]

    def mask(self, lo, hi, device):
        p = self.p.to(device)
        m = p >= lo
        if hi is not None:
            m &= p < hi
        return m.view(1, -1, 1)


def masked_ablation(model, pos: Pos, d: torch.Tensor, lo, hi):
    def fn(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        dd = d.to(h.device, h.dtype)
        m = pos.mask(lo, hi, h.device).to(h.dtype)
        h = h - m * (h @ dd).unsqueeze(-1) * dd
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    return [(l, fn, "fwd") for l in layers(model)]


def masked_addition(model, pos: Pos, layer: int, vec: torch.Tensor, lo, hi):
    def fn(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        m = pos.mask(lo, hi, h.device).to(h.dtype)
        h = h + m * vec.to(h.device, h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    return [(layers(model)[layer], fn, "fwd")]


def masked_head_zero(model, pos: Pos, layer: int, head: int, lo, hi):
    attn = layers(model)[layer].self_attn
    hd = attn.head_dim

    def fn(mod, args, kwargs):
        x = args[0] if args else kwargs["input"]
        m = pos.mask(lo, hi, x.device).squeeze(-1)  # [1, T]
        x = x.clone()
        sl = x[..., head * hd:(head + 1) * hd]
        x[..., head * hd:(head + 1) * hd] = sl * (~m).unsqueeze(-1).to(x.dtype)
        return ((x,) + tuple(args[1:]), kwargs) if args else (args, {**kwargs, "input": x})
    return [(attn.o_proj, fn, "pre")]


@torch.no_grad()
def gen_batch(model, tok, texts, max_new, extra_hooks_fn, bs):
    """extra_hooks_fn(T) -> list of hooks, given the padded prompt length T of the batch."""
    tok.padding_side = "left"
    outs = []
    for s in range(0, len(texts), bs):
        enc = tok(texts[s:s + bs], return_tensors="pt", padding=True, add_special_tokens=False,
                  return_token_type_ids=False).to(model.device)
        T = enc["input_ids"].shape[1]
        mx = max_new if isinstance(max_new, int) else max(max_new[s:s + bs])
        with hooks(extra_hooks_fn(T)):
            g = model.generate(**enc, max_new_tokens=mx, do_sample=False, pad_token_id=tok.pad_token_id)
        eos = model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos]) | {tok.eos_token_id}
        for seq in g:
            ids = seq[T:].tolist()
            cut = [i for i, t in enumerate(ids) if t in eos]
            if cut:
                ids = ids[: cut[0] + 1]
            outs.append((tok.decode(ids, skip_special_tokens=True), len(ids), tok.decode(ids[:24], skip_special_tokens=True)))
    return outs


def write(out, rec):
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")


def base_fields(r):
    return {k: r[k] for k in ("model", "src", "pidx", "target", "is_harmful", "k", "pos", "template", "user",
                              "target_item", "n_items")}


def raw_vec(model_name, L, d_hat):
    if model_name == "qwen3-14b":
        norm = json.loads((REPO / "results/qwen3-14b/refusal_direction/meta.json").read_text())["diff_norms_per_layer"][L]
    else:
        norm = float(torch.load(OUT / f"dirs_{model_name}.pt", map_location="cpu")["norms"][L])
    return d_hat[L] * norm


def layer_of(model_name):
    L = MODELS[model_name]["layer"]
    return L if L is not None else int(torch.load(OUT / f"dirs_{model_name}.pt", map_location="cpu")["layer"])


def steer_items(rows, n, k=15):
    """Same selection as steer.py: k=15 middle/last, segmented, not upfront; compliers first."""
    def ok(r, harmful):
        return (r["is_harmful"] == harmful and r["template"] == "numbered" and r["k"] == k and r["pos"] in ("middle", "last")
                and not r.get("upfront") and r["target_item"] in find_item_markers(r["response"], r["n_items"]))
    harm = sorted([r for r in rows if ok(r, True)], key=lambda r: (r.get("outcome") != "COMPLY", r["rid"]))[:n]
    ben = [r for r in rows if ok(r, False)][: max(16, n // 2)]
    return harm + ben


def onset_prefix(tok, model_name, r):
    marks = find_item_markers(r["response"], r["n_items"])
    return chat_text(tok, model_name, r["user"]) + r["response"][: content_start(r["response"], marks[r["target_item"]])]


# ----------------------------------------------------------------------------- experiments
def run_ablate(args, model, tok, rows, out, done):
    L = layer_of(args.model)
    d = load_dirs(args.model, model.config.num_hidden_layers)[L].to(model.device)
    if args.heldout:
        pick = heldout_rows(args.model)
        conds = {"none": None, "prompt": (0, "T")}
    else:
        pick = [r for r in rows if r["is_harmful"] and r["template"] == "numbered" and r["pidx"] < args.n // 2
                and ((r["k"] == 0) or (r["k"] in (3, 7) and r["pos"] == "last"))]
        conds = {"none": None, "prompt": (0, "T"), "gen": ("T", None), "all": (0, None)}
    pos = Pos(model)
    for cname, span in conds.items():
        todo = [r for r in pick if f"{r['rid']}|abl|{cname}" not in done]
        todo.sort(key=lambda r: r["k"])
        if not todo:
            continue
        for k in sorted({r["k"] for r in todo}):
            kr = [r for r in todo if r["k"] == k]
            texts = [chat_text(tok, args.model, r["user"]) for r in kr]

            def hk(T, span=span):
                if span is None:
                    return []
                lo = T if span[0] == "T" else span[0]
                hi = T if span[1] == "T" else span[1]
                return pos.handle() + masked_ablation(model, pos, d, lo, hi)
            res = gen_batch(model, tok, texts, args.max_new or max_new_tokens(k), hk, args.bs)
            for r, (text, n, f24) in zip(kr, res):
                rec = base_fields(r)
                rec.update(rid=f"{r['rid']}|abl|{cname}", cond=cname, response=text, n_gen=n, first24=f24,
                           finish="length" if n >= max_new_tokens(k) else "stop", layer=L)
                write(out, rec)
            print(f"[ablate] {cname} k={k} done ({len(kr)})", flush=True)


def heldout_rows(model_name):
    """Held-out confirmation set: 100 HarmBench 'standard' behaviours not in JBB/AdvBench (exact match),
    placed alone or last in 4- and 8-item numbered lists. Fixed seed; never used before this run."""
    import csv
    from common import Row, build_list_prompt, load_benign_pool, load_harmful, pick_benign
    seen = {h["behavior"].strip().lower() for h in load_harmful()}
    hb = [r["Behavior"].strip() for r in csv.DictReader(open(OUT / "harmbench_text_all.csv"))
          if r["FunctionalCategory"] == "standard" and r["Behavior"].strip().lower() not in seen]
    random.Random(2027).shuffle(hb)
    pool = load_benign_pool()
    rows = []
    for i, t in enumerate(hb[:100]):
        for k in (0, 3, 7):
            pos = "first" if k == 0 else "last"
            user, ti, n = build_list_prompt(t, pick_benign(pool, f"hb{i}", k), pos, "numbered")
            rows.append(dict(rid=f"{model_name}|harmbench|{i}|H|k{k}|{pos}|numbered", model=model_name, src="harmbench",
                             pidx=i, target=t, is_harmful=True, k=k, pos=pos, template="numbered",
                             target_item=ti, n_items=n, user=user))
    return rows


def run_prefill(args, model_name, rows, out, done):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    jbb = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
    targets = {i: r["Target"] for i, r in enumerate(jbb)}
    path = MODELS[model_name]["hf"]
    tok = AutoTokenizer.from_pretrained(path)
    jobs = []
    for r in rows:
        if not (r["is_harmful"] and r["src"] == "jbb" and r["template"] == "numbered" and
                (r["k"] == 0 or (r["k"] in (3, 15) and r["pos"] == "last"))):
            continue
        if r["k"] == 0:
            pre = chat_text(tok, model_name, r["user"])
        else:
            if r.get("upfront") or r["target_item"] not in find_item_markers(r["response"], r["n_items"]):
                continue
            pre = onset_prefix(tok, model_name, r)
        forced = targets[r["pidx"]] + ":\n\n"
        for cname, text in (("noprefill", pre), ("prefill", pre + forced)):
            rid = f"{r['rid']}|pf|{cname}"
            if rid not in done:
                jobs.append((rid, r, text, forced if cname == "prefill" else "", cname))
    if not jobs:
        return
    llm = LLM(model=path, dtype="bfloat16", max_model_len=8192 if "gemma" in model_name else 16384,
              gpu_memory_utilization=0.85, enable_prefix_caching=True)
    outs = llm.generate([j[2] for j in jobs], SamplingParams(temperature=0.0, max_tokens=256), use_tqdm=False)
    for (rid, r, _, forced, cname), o in zip(jobs, outs):
        rec = base_fields(r)
        c = o.outputs[0]
        rec.update(rid=rid, cond=cname, forced=forced, response=forced + c.text, continuation=c.text,
                   n_gen=len(c.token_ids), first24=tok.decode(list(c.token_ids)[:24], skip_special_tokens=True),
                   finish=c.finish_reason, target_item=1, n_items=1)
        write(out, rec)
    print(f"[prefill] {model_name}: {len(jobs)} generations", flush=True)


def run_steer1(args, model, tok, rows, out, done):
    L = layer_of(args.model)
    d = load_dirs(args.model, model.config.num_hidden_layers)
    vec = raw_vec(args.model, L, d)
    pos = Pos(model)
    items = steer_items(rows, args.n)
    alphas = {"qwen3-14b": [1.0, 2.0], "llama31-8b": [0.5, 1.0], "gemma2-9b": [0.25, 0.5]}.get(args.model, [1.0])
    for a in alphas:
        todo = [r for r in items if f"{r['rid']}|steer1|a{a}" not in done]
        if not todo:
            continue
        texts = [onset_prefix(tok, args.model, r) for r in todo]
        res = gen_batch(model, tok, texts, args.max_new or 160,
                        lambda T: pos.handle() + masked_addition(model, pos, L, a * vec, T - 1, T), args.bs)
        for r, (text, n, f24) in zip(todo, res):
            rec = base_fields(r)
            rec.update(rid=f"{r['rid']}|steer1|a{a}", cond=f"single_a{a}", alpha=a, base_outcome=r.get("outcome"),
                       response=text, n_gen=n, first24=f24, finish="length", target_item=1, n_items=1, layer=L)
            write(out, rec)
        print(f"[steer1] alpha={a} done", flush=True)


def run_head(args, model, tok, rows, out, done):
    layer, head = MODELS[args.model]["head"]
    n_heads = model.config.num_attention_heads
    ctrl = random.Random(0).sample([h for h in range(n_heads) if h != head], 3)
    pos = Pos(model)
    items = steer_items(rows, args.n)
    conds = [("none", None, None)] + [(f"h{h}_{where}", h, where) for h in [head] + ctrl for where in ("item", "all")]
    for cname, h, where in conds:
        todo = [r for r in items if f"{r['rid']}|head|{cname}" not in done]
        if not todo:
            continue
        texts = [onset_prefix(tok, args.model, r) for r in todo]

        def hk(T, h=h, where=where):
            if h is None:
                return []
            return pos.handle() + masked_head_zero(model, pos, layer, h, (T - 1) if where == "item" else 0, None)
        res = gen_batch(model, tok, texts, args.max_new or 160, hk, args.bs)
        for r, (text, n, f24) in zip(todo, res):
            rec = base_fields(r)
            rec.update(rid=f"{r['rid']}|head|{cname}", cond=cname, head=h, where=where, base_outcome=r.get("outcome"),
                       response=text, n_gen=n, first24=f24, finish="length", target_item=1, n_items=1)
            write(out, rec)
        print(f"[head] {cname} done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["ablate", "prefill", "steer1", "head"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--hf-path", default=None)
    ap.add_argument("--judged", default=None)
    ap.add_argument("--max-new", type=int, default=None)
    ap.add_argument("--heldout", action="store_true", help="ablate: fresh HarmBench targets (confirmation run)")
    args = ap.parse_args()
    rows = [] if args.heldout else read_jsonl(args.judged or OUT / f"judged_{args.model}.jsonl")
    out = OUT / f"gen_{args.exp}{'ho' if args.heldout else ''}_{args.model}.jsonl"
    done = {json.loads(l)["rid"] for l in open(out)} if out.exists() else set()
    t0 = time.time()
    if args.exp == "prefill":
        run_prefill(args, args.model, rows, out, done)
    else:
        model, tok = load_model(args.hf_path or MODELS[args.model]["hf"])
        {"ablate": run_ablate, "steer1": run_steer1, "head": run_head}[args.exp](args, model, tok, rows, out, done)
    print(f"[{args.exp}] {args.model} finished in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
