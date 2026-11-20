"""CPU tests: head-attention recompute matches eager attention; hooks behave; segmentation works."""
import sys, pathlib, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from transformers import Qwen3Config, Qwen3ForCausalLM, LlamaConfig, LlamaForCausalLM
from mech import HeadAttnRecorder, hooks, ablation_hooks, addition_hook
from common import segment, find_item_markers, build_list_prompt, v1_refusal, item_refusal

torch.manual_seed(0)
def tiny(kind):
    if kind == "qwen":
        cfg = Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                          num_attention_heads=8, num_key_value_heads=2, head_dim=16, max_position_embeddings=256)
        m = Qwen3ForCausalLM(cfg)
    else:
        cfg = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                          num_attention_heads=8, num_key_value_heads=2, max_position_embeddings=256)
        m = LlamaForCausalLM(cfg)
    m.config._attn_implementation = "eager"
    return m.eval()

for kind in ("qwen", "llama"):
    m = tiny(kind)
    ids = torch.randint(0, 128, (1, 40))
    rec = HeadAttnRecorder(m, 1, 5)
    with torch.no_grad(), hooks(rec.handle()):
        o = m(ids, output_attentions=True)
    pats = rec.pattern([10, 39])
    ref = o.attentions[1][0, 5]
    for p, pat in zip([10, 39], pats):
        err = (pat - ref[p, :p + 1]).abs().max().item()
        assert err < 1e-5, (kind, p, err)
    # ablation: projection onto d after each layer ~ 0
    d = torch.randn(64); d = d / d.norm()
    cap = []
    capfn = lambda mod, i, o: cap.append((o[0] if isinstance(o, tuple) else o).detach())
    with torch.no_grad(), hooks(ablation_hooks(m, d) + [(l, capfn, "fwd") for l in m.model.layers]):
        m(ids)
    assert max(float((h[0] @ d).abs().max()) for h in cap) < 1e-4, [float((h[0] @ d).abs().max()) for h in cap]
    # addition: only positions >= start are changed (layer 0 output)
    with torch.no_grad():
        cap.clear(); c1 = [(m.model.layers[0], capfn, "fwd")]
        with hooks(c1): m(ids)
        base = cap[-1]
        hk, st = addition_hook(m, 0, 3 * d, torch.tensor([20]))
        with hooks(hk + c1):
            m(ids)
        ste = cap[-1]
    delta = (ste - base)[0]
    assert delta[:20].abs().max() < 1e-5 and (delta[20:] @ d - 3).abs().max() < 1e-3
    # addition during generate with KV cache: offsets advance
    hk, st = addition_hook(m, 0, 0 * d, 0)
    with torch.no_grad(), hooks(hk):
        m.generate(ids, max_new_tokens=5, do_sample=False)
    assert st["offset"] == 44, st  # 40 prefill + 4 decode forwards
    print(kind, "mech ok")

# segmentation
resp = ("Sure! Here are the answers.\n\n1. **Poem:** Roses are red.\n\n2. **How to pick a lock**: I can't help with that.\n\n"
        "3. The capital of France is Paris.")
seg, how = segment(resp, 3, 2, "numbered")
assert how == "marker" and "can't help" in seg and "Paris" not in seg, (seg, how)
assert item_refusal(seg) and not item_refusal(segment(resp, 3, 3, "numbered")[0])
resp2 = "### Task 1: Summary\nblah\n### Task 2: Steps\nStep 1. do x\nStep 2. do y\n### Task 3: Haiku\nleaves"
m2 = find_item_markers(resp2, 3)
assert set(m2) == {1, 2, 3}, m2
assert "Step 2" in segment(resp2, 3, 2, "tasks")[0]
u, ti, n = build_list_prompt("HARM", ["a", "b", "c"], "middle", "numbered")
assert ti == 3 and n == 4 and "3. HARM" in u, u
u, ti, n = build_list_prompt("HARM", ["a", "b", "c"], "last", "tasks")
assert ti == 4 and "Task 4: HARM" in u
assert v1_refusal("I'm sorry, but") and not v1_refusal("1. Here is a poem")
print("segmentation ok")

# ---- causal.py position-masked hooks
from causal import Pos, masked_ablation, masked_head_zero, masked_addition
for kind in ("qwen", "llama"):
    m = tiny(kind)
    ids = torch.randint(0, 128, (1, 40))
    d = torch.randn(64); d = d / d.norm()
    pos = Pos(m); cap = []
    capfn = lambda mod, i, o: cap.append((o[0] if isinstance(o, tuple) else o).detach())
    with torch.no_grad(), hooks(pos.handle() + masked_ablation(m, pos, d, 0, 20) + [(l, capfn, "fwd") for l in m.model.layers]):
        m(ids)
    pr = torch.stack([(h[0] @ d).abs() for h in cap])  # [L, T]
    assert pr[:, :20].max() < 1e-4 and pr[:, 20:].min() > 1e-6, (pr[:, :20].max(), pr[:, 20:].min())
    # head zero: o_proj input slice zero only at positions >= 25
    seen = []
    o = m.model.layers[1].self_attn.o_proj
    rec = lambda mod, args, kwargs: seen.append(args[0].detach().clone())
    with torch.no_grad(), hooks(pos.handle() + masked_head_zero(m, pos, 1, 3, 25, None) + [(o, rec, "pre")]):
        m(ids)
    hd = m.model.layers[1].self_attn.head_dim
    x = seen[-1][0]
    assert x[25:, 3*hd:4*hd].abs().max() == 0 and x[:25, 3*hd:4*hd].abs().max() > 0 and x[25:, 2*hd:3*hd].abs().max() > 0
    # generation: cache_position flows -> addition only at the single decision token
    pos = Pos(m); cap = []
    c0 = [(m.model.layers[0], capfn, "fwd")]
    with torch.no_grad(), hooks(pos.handle() + c0):
        m.generate(ids, max_new_tokens=3, do_sample=False)
    base = [c.clone() for c in cap]; cap.clear()
    with torch.no_grad(), hooks(pos.handle() + masked_addition(m, pos, 0, 5 * d, 39, 40) + c0):
        m.generate(ids, max_new_tokens=3, do_sample=False)
    delta = (cap[0] - base[0])[0]
    assert delta[:39].abs().max() < 1e-5 and abs(float(delta[39] @ d) - 5) < 1e-3, float(delta[39] @ d)
    assert pos.p is not None and int(pos.p[-1]) >= 40   # decode steps saw absolute positions
    print(kind, "causal hooks ok")
