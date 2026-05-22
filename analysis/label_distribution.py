"""
Predicted label distribution table (label collapse analysis).

Input:  data/results/experiment_results_*.json
Output: analysis/output/label_distribution.csv

Shows what % of predictions each model assigns to each label per condition.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DATA_RESULTS, OUTPUT, MODEL_FILES, CONDITIONS, INT2LABEL, LABELS

import pandas as pd


def load_exp(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def pred_distribution(data: dict, condition: str) -> dict | None:
    if condition not in data:
        return None
    preds = [INT2LABEL[p] for p in data[condition]["preds"]]
    total = len(preds)
    if total == 0:
        return None
    counts = Counter(preds)
    return {
        f"pred_{label}_%": round(counts.get(label, 0) / total * 100, 1)
        for label in LABELS
    } | {"n": total}


def main() -> None:
    rows = []
    for fname, model_tag in MODEL_FILES.items():
        path = DATA_RESULTS / fname
        if not path.exists():
            print(f"[skip] {path} not found")
            continue
        data = load_exp(path)
        for cond in CONDITIONS:
            dist = pred_distribution(data, cond)
            if dist is None:
                continue
            rows.append({"model": model_tag, "condition": cond, **dist})

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    out = OUTPUT / "label_distribution.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
