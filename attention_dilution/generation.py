"""Generation with explicit failures, truncation, and incremental results."""

from __future__ import annotations

import json
from pathlib import Path

from .shared import check_context_budget, looks_like_refusal, strip_think_block


def generate_dataset(model, tokenizer, chat_prompts, source_prompts, *, device,
                     batch_size, max_new_tokens, temperature, context_budget,
                     path: Path, resume=False) -> list[dict]:
    import torch

    if batch_size <= 0 or max_new_tokens <= 0 or temperature < 0:
        raise ValueError("Batch size/output length must be positive; temperature non-negative")
    if len(chat_prompts) != len(source_prompts):
        raise ValueError("Source and formatted prompt counts differ")
    if resume and temperature > 0:
        raise ValueError("Sampling resumes are not supported; start a new run")
    rows = []
    if path.exists():
        if not resume:
            raise ValueError(f"Results already exist at {path}")
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                index = len(rows)
                if index >= len(source_prompts):
                    raise ValueError("Saved results contain too many prompts")
                if row["index"] != index or row["prompt"] != source_prompts[index]:
                    raise ValueError(f"Saved results do not match prompts at index {index}")
                rows.append(row)
    if len(rows) > len(source_prompts):
        raise ValueError("Saved results contain too many prompts")
    kwargs = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0,
              "pad_token_id": tokenizer.pad_token_id}
    if temperature > 0:
        kwargs.update(temperature=temperature, top_p=0.95)

    with path.open("a", encoding="utf-8") as handle, torch.inference_mode():
        for start in range(len(rows), len(chat_prompts), batch_size):
            batch = chat_prompts[start:start + batch_size]
            enc = tokenizer(batch, add_special_tokens=False, return_tensors="pt",
                            padding=True, truncation=False).to(device)
            width = enc["input_ids"].shape[1]
            check_context_budget(width, max_new_tokens, context_budget)
            try:
                outputs = model.generate(**enc, **kwargs)[:, width:]
                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                batch_rows = []
                for offset, raw in enumerate(decoded):
                    token_ids = outputs[offset].tolist()
                    eos = tokenizer.eos_token_id
                    # Count tokens before EOS/padding; a full allowance without EOS is truncated.
                    stop_ids = {tokenizer.pad_token_id}
                    if isinstance(eos, int):
                        stop_ids.add(eos)
                    elif eos:
                        stop_ids.update(eos)
                    completion = next((i for i, t in enumerate(token_ids) if t in stop_ids), len(token_ids))
                    response = strip_think_block(raw)
                    batch_rows.append({
                        "index": start + offset, "prompt": source_prompts[start + offset],
                        "n_prompt_tokens": int(enc["attention_mask"][offset].sum()),
                        "n_completion_tokens": completion, "response_raw": raw,
                        "response": response, "refused": looks_like_refusal(raw),
                        "truncated": completion >= max_new_tokens,
                        "status": "ok" if response.strip() else "empty",
                    })
                del outputs
            except torch.cuda.OutOfMemoryError:
                # A batch does not become a set of non-refusals when generation fails.
                torch.cuda.empty_cache()
                raise
            except Exception as exc:
                batch_rows = [{"index": start + offset, "prompt": source_prompts[start + offset],
                               "status": "error", "error": repr(exc), "refused": None}
                              for offset in range(len(batch))]
            for row in batch_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                rows.append(row)
            print(f"Generated {len(rows)}/{len(chat_prompts)}", flush=True)
    return rows
