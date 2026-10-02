"""Build the exploratory intent-by-topic pools.

The harmless mundane cell uses canonical filtered Alpaca[600:650), outside
core training, validation, and test ranges. Historical [200:250) pools
must be rebuilt: they overlap the expanded direction training pool.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXP1 = ROOT / "experiment_1"
for _p in (str(EXP1),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DATA = HERE / "data"

MANUAL_HARMFUL_MUNDANE = DATA / "manual_harmful_mundane.json"
MANUAL_HARMLESS_EDGY = DATA / "manual_harmless_edgy.json"

OUT_HARMFUL_EDGY = DATA / "harmful_edgy_topic.jsonl"
OUT_HARMFUL_MUNDANE = DATA / "harmful_mundane_topic.jsonl"
OUT_HARMLESS_EDGY = DATA / "harmless_edgy_topic.jsonl"
OUT_HARMLESS_MUNDANE = DATA / "harmless_mundane_topic.jsonl"

ADV_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/"
    "data/advbench/harmful_behaviors.csv"
)

CELL_SIZE = 50


def fetch_advbench_rows(start: int, n: int) -> list[tuple[int, str]]:
    raw = urllib.request.urlopen(ADV_URL).read().decode("utf-8").splitlines()
    rdr = csv.reader(raw)
    next(rdr)
    rows: list[tuple[int, str]] = []
    for file_row_idx, parts in enumerate(rdr):
        if file_row_idx < start:
            continue
        if len(rows) >= n:
            break
        rows.append((file_row_idx, parts[0].strip()))
    return rows


def fetch_alpaca_rows(n: int, start: int = 0) -> list[str]:
    """Fetch n harmless Alpaca instructions starting at post-filter offset.

    Filter mirrors experiment_1.utils.load_harmless_prompts: empty-input rows
    only, filtered through _harmless_instruction_is_clean. Same canonical
    filter exp_2 uses for d_hat training pool, so a `start` offset properly
    excludes prior cells.
    """
    from utils import load_harmless_prompts
    if n <= 0 or start < 0:
        raise ValueError("Invalid Alpaca slice")
    rows = load_harmless_prompts(start + n)
    if len(rows) < start + n:
        raise ValueError("Alpaca source does not contain the requested slice")
    return rows[start:start + n]


def load_manual(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected JSON list")
    out = []
    for row in raw:
        idx = int(row["index"])
        prompt = str(row.get("prompt", "")).strip()
        out.append({"index": idx, "prompt": prompt})
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--allow-partial",
        action="store_true",
        help="Skip empty manual rows instead of failing (dry-run mode).",
    )
    p.add_argument("--advbench-start", type=int, default=0)
    p.add_argument(
        "--alpaca-start",
        type=int,
        default=600,
        help="Post-filter Alpaca offset for harmless_mundane cell. "
             "Default 600 lies outside core train/validation/test ranges. "
             "Historical [200:250) pools must be rebuilt.",
    )
    return p.parse_args()


def build_manual_pool(
    path: Path,
    label_intent: int,
    label_topic: int,
    pool_name: str,
    allow_partial: bool,
) -> list[dict]:
    rows_in = load_manual(path)
    out: list[dict] = []
    empty: list[int] = []
    for r in rows_in:
        if not r["prompt"]:
            empty.append(r["index"])
            continue
        out.append({
            "pool": pool_name,
            "index": r["index"],
            "prompt": r["prompt"],
            "label_intent": label_intent,
            "label_topic": label_topic,
        })

    if empty and not allow_partial:
        sys.stderr.write(
            f"{path}: {len(empty)} rows missing non-empty `prompt` "
            f"(first: {empty[:5]}). Fill them or rerun with --allow-partial.\n"
        )
        sys.exit(1)
    if empty and allow_partial:
        sys.stderr.write(
            f"WARN {path}: skipping {len(empty)} empty rows (--allow-partial).\n"
        )

    if not allow_partial and len(out) != CELL_SIZE:
        sys.stderr.write(
            f"{path}: expected {CELL_SIZE} filled rows, got {len(out)}.\n"
        )
        sys.exit(1)
    return out


def main() -> None:
    args = parse_args()
    DATA.mkdir(parents=True, exist_ok=True)

    print(f"Fetching AdvBench[{args.advbench_start}:{args.advbench_start + CELL_SIZE}]...")
    adv_rows = fetch_advbench_rows(args.advbench_start, CELL_SIZE)
    harmful_edgy = [
        {
            "pool": "harmful_edgy_topic",
            "advbench_idx": idx,
            "prompt": text,
            "label_intent": 1,
            "label_topic": 1,
        }
        for idx, text in adv_rows
    ]
    write_jsonl(OUT_HARMFUL_EDGY, harmful_edgy)
    print(f"Wrote {OUT_HARMFUL_EDGY} ({len(harmful_edgy)} rows)")

    print(f"Fetching Alpaca[{args.alpaca_start}:{args.alpaca_start + CELL_SIZE}] post-filter...")
    alp = fetch_alpaca_rows(CELL_SIZE, start=args.alpaca_start)
    harmless_mundane = [
        {
            "pool": "harmless_mundane_topic",
            "index": args.alpaca_start + i,
            "prompt": text,
            "label_intent": 0,
            "label_topic": 0,
        }
        for i, text in enumerate(alp)
    ]
    write_jsonl(OUT_HARMLESS_MUNDANE, harmless_mundane)
    print(f"Wrote {OUT_HARMLESS_MUNDANE} ({len(harmless_mundane)} rows)")

    harmful_mundane = build_manual_pool(
        MANUAL_HARMFUL_MUNDANE, 1, 0, "harmful_mundane_topic", args.allow_partial
    )
    write_jsonl(OUT_HARMFUL_MUNDANE, harmful_mundane)
    print(f"Wrote {OUT_HARMFUL_MUNDANE} ({len(harmful_mundane)} rows)")

    harmless_edgy = build_manual_pool(
        MANUAL_HARMLESS_EDGY, 0, 1, "harmless_edgy_topic", args.allow_partial
    )
    write_jsonl(OUT_HARMLESS_EDGY, harmless_edgy)
    print(f"Wrote {OUT_HARMLESS_EDGY} ({len(harmless_edgy)} rows)")

    counts = {
        "harmful_edgy_topic": len(harmful_edgy),
        "harmful_mundane_topic": len(harmful_mundane),
        "harmless_edgy_topic": len(harmless_edgy),
        "harmless_mundane_topic": len(harmless_mundane),
    }
    print("\nFactorial cell counts:")
    for k, v in counts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

