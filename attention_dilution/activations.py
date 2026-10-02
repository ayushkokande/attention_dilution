"""Read selected residuals without constructing full vocabulary logits."""

from __future__ import annotations

from .shared import check_context_budget


def collect_residuals(model, tokenizer, chat_prompts, layers, batch_size, device,
                      *, context_budget=32768, request_positions=None):
    """Return CPU float32 [layer, prompt, position, hidden] residuals.

    Position 0 is the last token of the templated prompt (the readout).
    Position 1, when requested, is the last request token. Both refer to
    decoder-block outputs, before the model's final normalization.
    """
    import torch

    if batch_size <= 0 or not chat_prompts:
        raise ValueError("Positive batch size and nonempty prompts required")
    if request_positions is not None and len(request_positions) != len(chat_prompts):
        raise ValueError("Request position count does not match prompts")
    n_positions = 2 if request_positions is not None else 1
    output = torch.zeros(len(layers), len(chat_prompts), n_positions, model.config.hidden_size)
    cursor = [0]
    positions = [None]
    handles = []

    def make_hook(slot):
        def capture(module, inputs, result):
            hidden = result[0] if isinstance(result, tuple) else result
            indices = positions[0].to(hidden.device)
            rows = torch.arange(hidden.shape[0], device=hidden.device).unsqueeze(1)
            selected = hidden[rows, indices].detach().to("cpu", torch.float32)
            output[slot, cursor[0]:cursor[0] + hidden.shape[0]] = selected
        return capture

    try:
        for slot, layer in enumerate(layers):
            handles.append(model.model.layers[layer].register_forward_hook(make_hook(slot)))
        with torch.inference_mode():
            for start in range(0, len(chat_prompts), batch_size):
                batch = chat_prompts[start:start + batch_size]
                enc = tokenizer(batch, add_special_tokens=False, return_tensors="pt",
                                padding=True, truncation=False).to(device)
                width = enc["input_ids"].shape[1]
                check_context_budget(width, 0, context_budget)
                indices = torch.full((len(batch), n_positions), width - 1, dtype=torch.long)
                if request_positions is not None:
                    padding = width - enc["attention_mask"].sum(dim=1).cpu()
                    for row in range(len(batch)):
                        indices[row, 1] = request_positions[start + row] + padding[row]
                cursor[0], positions[0] = start, indices
                # The LM head would create [batch, tokens, vocabulary] logits,
                # which are unnecessary for residual measurements.
                model.model(**enc, use_cache=False)
                print(f"Captured {start + len(batch)}/{len(chat_prompts)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    return output
