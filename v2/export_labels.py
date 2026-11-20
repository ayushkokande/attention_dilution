"""Export public labels (no model response text) from judged files.

Usage: AD_OUT=<private dir with judged_*.jsonl> python v2/export_labels.py results/v2
Writes labels_<name>.jsonl.gz for every judged_<name>.jsonl, merging StrongREJECT scores ("sr" from
judged_<name>.sr.jsonl, "sr_seg" from judged_<name>.srseg.jsonl) when those files exist.
"""
from __future__ import annotations

import gzip
import json
import os
import sys
from pathlib import Path

SRC = Path(os.environ.get("AD_OUT", "results/v2"))
DST = Path(sys.argv[1] if len(sys.argv) > 1 else "results/v2")
DROP = {"response", "segment", "thinking", "first24", "user"}  # model text and the full prompt


def rows(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def main():
    DST.mkdir(parents=True, exist_ok=True)
    stems = sorted({q.name.split(".")[0] for q in SRC.glob("judged_*.jsonl")})
    for stem in stems:
        name = stem[len("judged_"):]
        p = SRC / f"{stem}.jsonl"
        if not p.exists():  # main grids: the .sr file carries every judged field plus the score
            p = SRC / f"{stem}.sr.jsonl"
        R = rows(p)
        for ext, field in ((".sr.jsonl", "sr"), (".srseg.jsonl", "sr_seg")):
            q = SRC / f"{stem}{ext}"
            if q.exists():
                extra = {r["rid"]: r.get(field) for r in rows(q)}
                for r in R:
                    r[field] = extra.get(r["rid"])
        with gzip.open(DST / f"labels_{name}.jsonl.gz", "wt") as f:
            for r in R:
                f.write(json.dumps({k: v for k, v in r.items() if k not in DROP}) + "\n")
        print(f"[labels] {name}: {len(R)} rows")


if __name__ == "__main__":
    main()
