"""
Dataset validity statistics from context_conditions.json.

Input:  data/processed/context_conditions.json
        Format: list of records with fields:
          reply_id, thread_id, label, platform, split,
          reply_only:str, useful:str|null, irrelevant:str|null,
          conflicting:str|null, mixed:str|null, lexical:str|null,
          valid_ti:bool, valid_cc:bool

Output: analysis/output/dataset_validity.csv

Reports per (split, condition):
  - total replies
  - non-null (available) count and %
  - valid_cc=True count (for c3/c4)
  - valid_ti=True count
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DATA_PROCESSED, OUTPUT, CONDITIONS, COND_TO_FIELD

import pandas as pd


def load_ctx(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    path = DATA_PROCESSED / "context_conditions.json"
    if not path.exists():
        print(f"[error] {path} not found")
        return

    records = load_ctx(path)

    rows = []
    for split in ["train", "dev", "test"]:
        split_recs = [r for r in records if r.get("split") == split]
        total = len(split_recs)
        for cond in CONDITIONS:
            field = COND_TO_FIELD[cond]
            available = [r for r in split_recs if r.get(field) is not None]
            valid_cc = [r for r in available if r.get("valid_cc", False)] if field in ("conflicting", "mixed") else []
            valid_ti = [r for r in available if r.get("valid_ti", False)] if field == "irrelevant" else []
            rows.append({
                "split": split,
                "condition": cond,
                "total_replies": total,
                "available": len(available),
                "available_%": round(len(available) / total * 100, 1) if total else 0,
                "valid_cc": len(valid_cc) if field in ("conflicting", "mixed") else None,
                "valid_ti": len(valid_ti) if field == "irrelevant" else None,
            })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    out = OUTPUT / "dataset_validity.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
