"""
Per-class F1 table across models and conditions.

Input:  data/results/experiment_results_*.json
        Format: {condition: {golds:[int], preds:[int], macro_f1:float, per_class_f1:[f,f,f,f]}}
        Label int: 0=support 1=deny 2=query 3=comment

Output: analysis/output/per_class_f1.csv
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

# allow running from project root or analysis/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DATA_RESULTS, OUTPUT, MODEL_FILES, CONDITIONS, INT2LABEL, LABELS

import pandas as pd
from sklearn.metrics import classification_report


def load_exp(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def per_class_f1_from_data(data: dict, condition: str) -> dict | None:
    if condition not in data:
        return None
    cond = data[condition]
    golds = [INT2LABEL[g] for g in cond["golds"]]
    preds = [INT2LABEL[p] for p in cond["preds"]]
    report = classification_report(
        golds, preds, labels=LABELS, output_dict=True, zero_division=0
    )
    return {
        "support_f1": round(report["support"]["f1-score"], 4),
        "deny_f1":    round(report["deny"]["f1-score"], 4),
        "query_f1":   round(report["query"]["f1-score"], 4),
        "comment_f1": round(report["comment"]["f1-score"], 4),
        "macro_f1":   round(report["macro avg"]["f1-score"], 4),
        "n":          cond["n"],
    }


def main() -> None:
    rows = []
    for fname, model_tag in MODEL_FILES.items():
        path = DATA_RESULTS / fname
        if not path.exists():
            print(f"[skip] {path} not found")
            continue
        data = load_exp(path)
        for cond in CONDITIONS:
            result = per_class_f1_from_data(data, cond)
            if result is None:
                continue
            rows.append({"model": model_tag, "condition": cond, **result})

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    out = OUTPUT / "per_class_f1.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
