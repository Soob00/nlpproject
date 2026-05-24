"""
Per-class F1 table across models and conditions.

Input:  data/results/experiment_results_*.json
Output: analysis/output/per_class_f1.csv

Columns: model, split, condition, support_f1, deny_f1, query_f1, comment_f1, macro_f1, n
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import (
    DATA_RESULTS, OUTPUT, MODEL_FILES, CONDITIONS, COND_TO_FIELD,
    INT2LABEL, LABELS, EVAL_SPLIT, load_exp,
)

import pandas as pd
from sklearn.metrics import classification_report


def per_class_f1_from_data(split_data: dict, condition: str) -> dict | None:
    if condition not in split_data:
        return None
    cond  = split_data[condition]
    golds = [INT2LABEL.get(g, "invalid") for g in cond["golds"]]
    preds = [INT2LABEL.get(p, "invalid") for p in cond["preds"]]
    report = classification_report(
        golds, preds, labels=LABELS, output_dict=True, zero_division=0,
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
            print(f"[skip] {path.name} not found")
            continue
        split_data = load_exp(path, EVAL_SPLIT)
        if not split_data:
            print(f"[skip] no data for split='{EVAL_SPLIT}' in {path.name}")
            continue
        for cond in CONDITIONS:
            result = per_class_f1_from_data(split_data, cond)
            if result is None:
                continue
            rows.append({
                "model":     model_tag,
                "split":     EVAL_SPLIT,
                "condition": COND_TO_FIELD[cond],
                **result,
            })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    out = OUTPUT / "per_class_f1.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
