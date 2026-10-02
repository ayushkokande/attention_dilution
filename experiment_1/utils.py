"""Compatibility imports for the numbered exploratory experiments.

The maintained core helpers live in attention_dilution.shared.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attention_dilution.shared import (
    ADV_BENCH_URL, MODEL_NAME, MODEL_SLUG, PROJECT_ROOT, RESULTS_ROOT,
    REFUSAL_MARKERS, _HARMLESS_EXCLUDE_SUBSTRINGS, _harmless_instruction_is_clean,
    format_chat_prompt, load_harmful_prompts, load_harmless_prompts,
    looks_like_refusal, results_dir, strip_think_block,
)


def ensure_transformers_cache_attr():
    import transformers
    if not hasattr(transformers, "TRANSFORMERS_CACHE"):
        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        transformers.TRANSFORMERS_CACHE = os.path.join(hf_home, "hub")


def get_device():
    import torch
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32
