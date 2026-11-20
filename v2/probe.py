"""E3: where is the refusal decision read out in list prompts?

For each judged generation we run ONE forward pass over  [chat prompt + the model's own greedy
response, cut shortly after the target item's answer begins]  and record, at three positions:
  last     : final prompt token (the position v1 and Arditi et al. probe)
  target   : first content token of the response segment answering the TARGET item
  benign   : first content token of a response segment answering a BENIGN item
  * projection of the residual stream onto d-hat_l for every layer l
  * for the guardrail head (Qwen3-14B L36H31): attention mass onto the request text of
    (a) the target item and (b) the item whose answer begins at that position.

Usage: python v2/probe.py --model qwen3-14b [--n 200] [--hf-path ...]
Output: results/v2/probe_<model>.jsonl
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from common import MODELS, OUT, _MARK, chat_text, find_item_markers, read_jsonl
from mech import HeadAttnRecorder, char_to_token, hooks, load_model, span_tokens


def content_start(resp: str, mark_off: int) -> int:
    m = _MARK.match(resp, mark_off)
    j = m.end() if m else mark_off
    while j < len(resp) and resp[j] in " *_\t:":
        j += 1
    return j


def load_dirs(model_name: str, n_layers: int) -> torch.Tensor:
    if model_name == "qwen3-14b":
        from common import REPO
        d = torch.load(REPO / "results/qwen3-14b/refusal_direction/d_hat_all_layers.pt", map_location="cpu")
    else:
        d = torch.load(OUT / f"dirs_{model_name}.pt", map_location="cpu")["d_hat"]
    assert d.shape[0] == n_layers
    return d.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--hf-path", default=None)
    ap.add_argument("--judged", default=None)
    args = ap.parse_args()
    cfg = MODELS[args.model]
    rows = read_jsonl(args.judged or OUT / f"judged_{args.model}.jsonl")
    sel = [r for r in rows if r["template"] == "numbered" and r["k"] in (0, 3, 15)
           and r["pos"] in ("first", "last")]
    # balance: up to n per (is_harmful, k, pos)
    cells = {}
    for r in sel:
        cells.setdefault((r["is_harmful"], r["k"], r["pos"]), []).append(r)
    sel = [r for v in cells.values() for r in v[: args.n]]
    out = OUT / f"probe_{args.model}.jsonl"
    done = {json.loads(l)["rid"] for l in open(out)} if out.exists() else set()
    sel = [r for r in sel if r["rid"] not in done]
    print(f"[probe] {len(sel)} rows", flush=True)

    model, tok = load_model(args.hf_path or cfg["hf"])
    L = model.config.num_hidden_layers
    D = load_dirs(args.model, L).to(model.device)
    rec = HeadAttnRecorder(model, *cfg["head"]) if cfg.get("head") else None
    t0 = time.time()
    for n_done, r in enumerate(sel):
        prompt = chat_text(tok, args.model, r["user"])
        resp = r["response"]
        marks = find_item_markers(resp, r["n_items"]) if r["n_items"] > 1 else {}
        tgt = r["target_item"]
        pos_chars = {}
        if r["n_items"] == 1:
            cut = min(len(resp), 400)
        else:
            if tgt in marks:
                pos_chars["target"] = content_start(resp, marks[tgt])
            ben = [i for i in marks if i != tgt]
            if ben:
                b = min(ben, key=lambda i: abs(i - tgt))
                pos_chars["benign"] = content_start(resp, marks[b])
                pos_chars["_benign_item"] = b
            cut = min(len(resp), max([v for k_, v in pos_chars.items() if not k_.startswith("_")] + [0]) + 300)
        full = prompt + resp[:cut]
        enc = tok(full, return_offsets_mapping=True, add_special_tokens=False)
        offs = enc["offset_mapping"]
        ids = torch.tensor([enc["input_ids"]], device=model.device)
        n_prompt = len(tok(prompt, add_special_tokens=False)["input_ids"])
        P = {"last": n_prompt - 1}
        for name in ("target", "benign"):
            if name in pos_chars:
                P[name] = char_to_token(offs, len(prompt) + pos_chars[name])
        # request-text spans (in the prompt) for the target item and the benign comparison item
        user_off = prompt.find(r["user"])
        lines = r["user"].split("\n")
        item_lines = [ln for ln in lines if ln.strip()][1:] if r["n_items"] > 1 else [r["user"]]

        def req_span(item_no):
            if r["n_items"] == 1:
                c0 = user_off
                return span_tokens(offs, c0, c0 + len(r["user"]))
            if item_no == tgt or len(item_lines) != r["n_items"]:
                txt = r["target"] if item_no == tgt else ""
                if not txt:
                    return []
            else:
                txt = item_lines[item_no - 1]
            c0 = user_off + r["user"].find(txt)
            return span_tokens(offs, c0, c0 + len(txt))

        with torch.no_grad(), hooks(rec.handle() if rec else []):
            o = model(ids, output_hidden_states=True, use_cache=False)
        res = {"rid": r["rid"], "positions": P, "n_tokens": ids.shape[1]}
        for name, p in P.items():
            hs = torch.stack([o.hidden_states[l + 1][0, p] for l in range(L)]).float()
            res[f"proj_{name}"] = (hs * D).sum(-1).tolist()
            res[f"cos_{name}"] = torch.nn.functional.cosine_similarity(hs, D, dim=-1).tolist()
        if rec is not None:
            qpos = list(P.values())
            pats = rec.pattern(qpos)
            tspan = req_span(tgt)
            for (name, p), pat in zip(P.items(), pats):
                res[f"attn_{name}_to_target"] = float(pat[tspan].sum())
                if name == "benign":
                    res["attn_benign_to_own"] = float(pat[req_span(pos_chars["_benign_item"])].sum())
        with open(out, "a") as f:
            f.write(json.dumps(res) + "\n")
        if n_done % 50 == 0:
            print(f"[probe] {n_done}/{len(sel)} {time.time() - t0:.0f}s", flush=True)
        del o
    print("[probe] done", flush=True)


if __name__ == "__main__":
    main()
