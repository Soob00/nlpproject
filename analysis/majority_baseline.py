"""
Majority baseline: always predict the most frequent class ('comment').

Used in Section 5.1 to contextualise per-class F1 results.
Does not use any model output — computed directly from gold labels.

Input:  data/processed/context_conditions.json
Output: analysis/output/majority_baseline.csv

Columns: split, condition, majority_label, accuracy, support_f1, deny_f1,
         query_f1, comment_f1, macro_f1, n
"""
from __future__ import annotations
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DATA_PROCESSED, OUTPUT, CONDITION_FIELDS, LABELS, EVAL_SPLIT

import pandas as pd
from sklearn.metrics import classification_report


def majority_baseline(records: list[dict], field: str) -> dict | None:
    subset = [r for r in records if r.get(field) is not None]
    if not subset:
        return None

    golds = [r["label"] for r in subset]
    majority_label = Counter(golds).most_common(1)[0][0]
    preds = [majority_label] * len(golds)

    report  = classification_report(
        golds, preds, labels=LABELS, output_dict=True, zero_division=0,
    )
    accuracy = sum(g == p for g, p in zip(golds, preds)) / len(golds)

    return {
        "majority_label": majority_label,
        "accuracy":      round(accuracy, 4),
        "support_f1":    round(report["support"]["f1-score"], 4),
        "deny_f1":       round(report["deny"]["f1-score"], 4),
        "query_f1":      round(report["query"]["f1-score"], 4),
        "comment_f1":    round(report["comment"]["f1-score"], 4),
        "macro_f1":      round(report["macro avg"]["f1-score"], 4),
        "n":             len(subset),
    }


def main() -> None:
    path = DATA_PROCESSED / "context_conditions.json"
    with open(path, encoding="utf-8") as f:
        all_records = json.load(f)

    rows = []
    for split in [EVAL_SPLIT]:
        records = [r for r in all_records if r.get("split") == split]
        for field in CONDITION_FIELDS:
            result = majority_baseline(records, field)
            if result is None:
                continue
            rows.append({"split": split, "condition": field, **result})

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    out = OUTPUT / "majority_baseline.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved → {out}")

    # also print a compact one-line summary (majority label is the same across conditions)
    rep = rows[0] if rows else {}
    if rep:
        print(f"\nMajority label = '{rep['majority_label']}'")
        print(f"  accuracy  = {rep['accuracy']:.4f}")
        print(f"  macro_F1  = {rep['macro_f1']:.4f}")
        print(f"  comment_F1= {rep['comment_f1']:.4f}")
        print(f"  support/deny/query F1 = 0.000 (never predicted)")


if __name__ == "__main__":
    main()
