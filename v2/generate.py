"""E1: full-length greedy generations on the list grid (vLLM; HF fallback for CPU smoke tests).

Usage: python v2/generate.py --model qwen3-14b [--smoke] [--backend vllm|hf]
Output: results/v2/gen_<model>.jsonl  (one row per prompt; resumable)
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

from common import MODELS, OUT, build_grid, chat_text, max_new_tokens, read_jsonl

SEED_BASE = 1234


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    ap.add_argument("--hf-path", default=None, help="override weights path (smoke tests)")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--variant", default=None, choices=[None, "thinking", "sample"],
                    help="thinking: Qwen3 with reasoning on; sample: T=0.7/top-p 0.95, 3 seeds. Reduced grid.")
    args = ap.parse_args()

    tag = f"_{args.variant}" if args.variant else ""
    out = OUT / f"gen{tag}_{args.model}{'_smoke' if args.smoke else ''}.jsonl"
    rows = build_grid(args.model, smoke=args.smoke)
    if args.variant:   # reduced grid: numbered template, k in {0,3,15}, first/last
        rows = [r for r in rows if r.template == "numbered" and r.k in (0, 3, 15) and r.pos in ("first", "last")]
    if args.variant == "thinking":
        MODELS[args.model] = dict(MODELS[args.model], chat_kwargs={"enable_thinking": True})
    done = {r["rid"] for r in read_jsonl(out)} if out.exists() else set()
    todo = [r for r in rows if r.rid not in done]
    print(f"[gen] {args.model}: {len(rows)} rows, {len(todo)} to do -> {out}", flush=True)
    if not todo:
        return
    path = args.hf_path or MODELS[args.model]["hf"]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    prompts = [chat_text(tok, args.model, r.user) for r in todo]

    t0 = time.time()
    if args.backend == "vllm":
        from vllm import LLM, SamplingParams
        max_len = 8192 if "gemma" in args.model else 16384
        llm = LLM(model=path, dtype="bfloat16", tensor_parallel_size=args.tp,
                  max_model_len=max_len, gpu_memory_utilization=0.90, enable_prefix_caching=True)
        plen = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
        # chunk so partial progress is saved
        CH = 512
        for s in range(0, len(todo), CH):
            chunk, cp, pl = todo[s:s + CH], prompts[s:s + CH], plen[s:s + CH]
            extra = 4096 if args.variant == "thinking" else 0
            if args.variant == "sample":
                sps = [SamplingParams(temperature=0.7, top_p=0.95, n=3, seed=SEED_BASE + i,
                                      max_tokens=max(16, min(max_new_tokens(r.k), max_len - n - 8)))
                       for i, (r, n) in enumerate(zip(chunk, pl))]
            else:
                sps = [SamplingParams(temperature=0.0, max_tokens=max(16, min(max_new_tokens(r.k) + extra, max_len - n - 8)))
                       for r, n in zip(chunk, pl)]
            outs = llm.generate(cp, sps, use_tqdm=False)
            with open(out, "a") as f:
                for r, p, o in zip(chunk, cp, outs):
                    for si, c in enumerate(o.outputs):
                        ids = list(c.token_ids)
                        rec = asdict(r)
                        text, thinking = c.text, None
                        if "</think>" in text:          # thinking variant: judge only the visible answer
                            thinking, text = text.split("</think>", 1)
                            text = text.lstrip()
                        elif args.variant == "thinking":  # ran out of budget inside the reasoning
                            thinking, text = text, ""
                        f24 = tok.decode(tok(text, add_special_tokens=False)["input_ids"][:24]) if thinking is not None \
                            else tok.decode(ids[:24], skip_special_tokens=True)
                        if args.variant == "sample":
                            rec["rid"] = f"{r.rid}|s{si}"
                        rec.update(response=text, thinking=thinking, n_gen=len(ids), finish=c.finish_reason,
                                   first24=f24, n_prompt=len(o.prompt_token_ids), variant=args.variant, sample=si)
                        f.write(json.dumps(rec) + "\n")
            print(f"[gen] {s + len(chunk)}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
    else:
        import torch
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                                                     device_map="auto" if torch.cuda.is_available() else None)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        for r, p in zip(todo, prompts):
            enc = tok(p, return_tensors="pt", add_special_tokens=False, return_token_type_ids=False).to(model.device)
            mx = 48 if args.smoke else max_new_tokens(r.k)
            g = model.generate(**enc, max_new_tokens=mx, do_sample=False, pad_token_id=tok.pad_token_id)
            ids = g[0, enc["input_ids"].shape[1]:].tolist()
            rec = asdict(r)
            rec.update(response=tok.decode(ids, skip_special_tokens=True), n_gen=len(ids),
                       finish="length" if len(ids) >= mx else "stop",
                       first24=tok.decode(ids[:24], skip_special_tokens=True),
                       n_prompt=enc["input_ids"].shape[1])
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
    print(f"[gen] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
