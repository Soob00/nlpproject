"""
Dataset validity table for paper.

Shows per-condition coverage and validity statistics for train/dev splits.
Output: analysis/output/dataset_validity_table.csv
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DATA_PROCESSED, OUTPUT, CONDITION_FIELDS

import pandas as pd


def main() -> None:
    path = DATA_PROCESSED / "context_conditions.json"
    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    rows = []
    for field in CONDITION_FIELDS:
        row = {"condition": field}
        for split in ["train", "dev"]:
            recs = [r for r in records if r.get("split") == split]
            total = len(recs)
            available = [r for r in recs if r.get(field) is not None]
            valid_cc  = [r for r in available if r.get("valid_cc", False)] if field in ("conflicting", "mixed") else []
            valid_ti  = [r for r in available if r.get("valid_ti", False)] if field == "irrelevant" else []

            avail_n   = len(available)
            avail_pct = round(avail_n / total * 100, 1) if total else 0.0
            vcc_n     = len(valid_cc)
            vti_n     = len(valid_ti)
            invalid_n = total - avail_n

            row[f"{split}_total"]     = total
            row[f"{split}_available"] = avail_n
            row[f"{split}_avail_%"]   = avail_pct
            row[f"{split}_invalid"]   = invalid_n
            row[f"{split}_valid_cc"]  = vcc_n if field in ("conflicting", "mixed") else None
            row[f"{split}_valid_ti"]  = vti_n if field == "irrelevant" else None

        rows.append(row)

    df = pd.DataFrame(rows)

    # Print paper-friendly view
    print("=== Dataset Validity Table ===\n")
    print(f"{'Condition':<12} | {'Train':^25} | {'Dev':^25}")
    print(f"{'':12} | {'avail/total (%)':^25} | {'avail/total (%)':^25}")
    print("-" * 70)
    for _, r in df.iterrows():
        tr = f"{r['train_available']}/{r['train_total']} ({r['train_avail_%']}%)"
        dv = f"{r['dev_available']}/{r['dev_total']} ({r['dev_avail_%']}%)"
        print(f"{r['condition']:<12} | {tr:^25} | {dv:^25}")

    print("\n=== valid_cc (conflicting/mixed validity) ===")
    for _, r in df.iterrows():
        if r["condition"] in ("conflicting", "mixed"):
            print(f"  {r['condition']}: train valid_cc={r['train_valid_cc']} / dev valid_cc={r['dev_valid_cc']}")

    print("\n=== valid_ti (thread info available) ===")
    for _, r in df.iterrows():
        if r["condition"] == "irrelevant":
            print(f"  {r['condition']}: train={r['train_valid_ti']} / dev={r['dev_valid_ti']}")

    out = OUTPUT / "dataset_validity_table.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
