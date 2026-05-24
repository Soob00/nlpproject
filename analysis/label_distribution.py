"""
Predicted label distribution table (label collapse analysis).

Input:  data/results/experiment_results_*.json
Output: analysis/output/label_distribution.csv

Shows what % of predictions each model assigns to each label per condition.
"""
from __future__ import annotations
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import (
    DATA_RESULTS, OUTPUT, MODEL_FILES, CONDITIONS, COND_TO_FIELD,
    INT2LABEL, PRED_LABELS, EVAL_SPLIT, load_exp,
)

import pandas as pd


def pred_distribution(split_data: dict, condition: str) -> dict | None:
    if condition not in split_data:
        return None
    preds = [INT2LABEL.get(p, "invalid") for p in split_data[condition]["preds"]]
    total = len(preds)
    if total == 0:
        return None
    counts = Counter(preds)
    return {
        f"pred_{lbl}_%": round(counts.get(lbl, 0) / total * 100, 1)
        for lbl in PRED_LABELS
    } | {"n": total}


def main() -> None:
    rows = []
    for fname, model_tag in MODEL_FILES.items():
        path = DATA_RESULTS / fname
        if not path.exists():
            print(f"[skip] {path.name} not found")
            continue
        split_data = load_exp(path, EVAL_SPLIT)
        if not split_data:
            continue
        for cond in CONDITIONS:
            dist = pred_distribution(split_data, cond)
            if dist is None:
                continue
            rows.append({
                "model":     model_tag,
                "split":     EVAL_SPLIT,
                "condition": COND_TO_FIELD[cond],
                **dist,
            })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    out = OUTPUT / "label_distribution.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
