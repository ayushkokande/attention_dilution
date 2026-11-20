"""Model/hook utilities shared by extract_direction.py, probe.py, steer.py (HF transformers)."""
from __future__ import annotations

import contextlib
from typing import Dict, List, Optional, Tuple

import torch


def load_model(path: str, attn: str = "sdpa"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16 if cuda else torch.float32,
        device_map="auto" if cuda else None, attn_implementation=("eager" if "gemma" in path.lower() else attn))
    model.eval()
    return model, tok


def layers(model):
    return model.model.layers


@contextlib.contextmanager
def hooks(handles_fns):
    hs = []
    try:
        for mod, fn, kind in handles_fns:
            if kind == "fwd":
                hs.append(mod.register_forward_hook(fn))
            else:
                hs.append(mod.register_forward_pre_hook(fn, with_kwargs=True))
        yield
    finally:
        for h in hs:
            h.remove()


def ablation_hooks(model, d: torch.Tensor):
    """Directional ablation: x <- x - (x . d) d at the output of every decoder layer (d unit)."""
    def fn(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        dd = d.to(h.device, h.dtype)
        h = h - (h @ dd).unsqueeze(-1) * dd
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    return [(l, fn, "fwd") for l in layers(model)]


def addition_hook(model, layer: int, vec: torch.Tensor, start_pos):
    """Activation addition at the output of `layer`, for (padded) positions >= start_pos.

    start_pos: int or LongTensor[B] (per-row, in left-padded coordinates).
    Works during generation with a KV cache: we track the absolute position of the
    first token in each forward call. Reset state["offset"]=0 before each generate().
    """
    state = {"offset": 0}

    def fn(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        B, T = h.shape[0], h.shape[1]
        pos = torch.arange(state["offset"], state["offset"] + T, device=h.device).view(1, T)
        state["offset"] += T
        sp = torch.as_tensor(start_pos, device=h.device).view(-1, 1)
        m = (pos >= sp).to(h.dtype).unsqueeze(-1)      # [B or 1, T, 1]
        h = h + m * vec.to(h.device, h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    return [(layers(model)[layer], fn, "fwd")], state


class HeadAttnRecorder:
    """Recompute one head's attention pattern for a few query positions without materialising
    the full [H, T, T] tensor. Matches eager attention (verified in tests/test_mech.py)."""

    def __init__(self, model, layer: int, head: int):
        self.attn = layers(model)[layer].self_attn
        self.head = head
        self.cap = {}

    def pre_hook(self, mod, args, kwargs):
        hs = kwargs.get("hidden_states", args[0] if args else None)
        self.cap["h"] = hs.detach()
        self.cap["pe"] = kwargs.get("position_embeddings", args[1] if len(args) > 1 else None)

    def handle(self):
        return [(self.attn, self.pre_hook, "pre")]

    @torch.no_grad()
    def pattern(self, qpos: List[int]) -> torch.Tensor:
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        a, h = self.attn, self.cap["h"]          # h: [1, T, d]
        hd = a.head_dim
        q = a.q_proj(h).view(1, h.shape[1], -1, hd)
        k = a.k_proj(h).view(1, h.shape[1], -1, hd)
        if hasattr(a, "q_norm"):
            q, k = a.q_norm(q), a.k_norm(k)
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        cos, sin = self.cap["pe"]
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        n_rep = q.shape[1] // k.shape[1]
        qh = q[0, self.head]                      # [T, hd]
        kh = k[0, self.head // n_rep]             # [T, hd]
        out = []
        for p in qpos:
            s = (qh[p] @ kh[: p + 1].T).float() * a.scaling
            out.append(torch.softmax(s, -1))
        return out


def char_to_token(offsets: List[Tuple[int, int]], char_pos: int) -> int:
    for i, (s, e) in enumerate(offsets):
        if s <= char_pos < e or (s >= char_pos and e > s):
            return i
    return len(offsets) - 1


def span_tokens(offsets, c0: int, c1: int) -> List[int]:
    return [i for i, (s, e) in enumerate(offsets) if e > c0 and s < c1 and e > s]
