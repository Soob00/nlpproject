"""
Minority-class analysis for RumourEval experiments.

Metrics per (model, condition):
  minority_macro_f1  – macro-F1 over support/deny/query only (comment excluded)
  pred_comment_rate  – fraction of predictions that are 'comment'
  nc_macro_f1        – macro-F1 on non-comment gold subset
  nc_support_f1      – support F1 on non-comment subset
  nc_deny_f1         – deny F1 on non-comment subset
  nc_query_f1        – query F1 on non-comment subset
  nc_n               – number of non-comment gold samples

Input:  data/results/experiment_results_*.json
Output: analysis/output/minority_analysis.csv
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import (
    DATA_RESULTS, OUTPUT, MODEL_FILES, CONDITIONS, COND_TO_FIELD,
    INT2LABEL, EVAL_SPLIT, load_exp,
)

import pandas as pd
from sklearn.metrics import classification_report

MINORITY_LABELS = ["support", "deny", "query"]


def analyze(split_data: dict, condition: str) -> dict | None:
    if condition not in split_data:
        return None
    cond      = split_data[condition]
    golds_int = cond["golds"]
    preds_int = cond["preds"]
    n         = len(golds_int)
    if n == 0:
        return None

    golds_str = [INT2LABEL.get(g, "invalid") for g in golds_int]
    preds_str = [INT2LABEL.get(p, "invalid") for p in preds_int]

    report_all = classification_report(
        golds_str, preds_str, labels=MINORITY_LABELS, output_dict=True, zero_division=0,
    )
    minority_macro_f1 = round(
        sum(report_all[lbl]["f1-score"] for lbl in MINORITY_LABELS) / len(MINORITY_LABELS), 4,
    )
    pred_comment_rate = round(sum(p == 3 for p in preds_int) / n, 4)

    nc_golds = [g for g, _ in zip(golds_str, preds_str) if g != "comment"]
    nc_preds = [p for g, p in zip(golds_str, preds_str) if g != "comment"]
    nc_n     = len(nc_golds)

    if nc_n == 0:
        nc_macro_f1 = nc_support_f1 = nc_deny_f1 = nc_query_f1 = float("nan")
    else:
        nc_report   = classification_report(
            nc_golds, nc_preds, labels=MINORITY_LABELS, output_dict=True, zero_division=0,
        )
        nc_macro_f1   = round(sum(nc_report[lbl]["f1-score"] for lbl in MINORITY_LABELS) / len(MINORITY_LABELS), 4)
        nc_support_f1 = round(nc_report["support"]["f1-score"], 4)
        nc_deny_f1    = round(nc_report["deny"]["f1-score"], 4)
        nc_query_f1   = round(nc_report["query"]["f1-score"], 4)

    return {
        "minority_macro_f1": minority_macro_f1,
        "pred_comment_rate": pred_comment_rate,
        "nc_n":              nc_n,
        "nc_macro_f1":       nc_macro_f1,
        "nc_support_f1":     nc_support_f1,
        "nc_deny_f1":        nc_deny_f1,
        "nc_query_f1":       nc_query_f1,
        "n":                 n,
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
            continue
        for cond in CONDITIONS:
            result = analyze(split_data, cond)
            if result is None:
                continue
            rows.append({
                "model":     model_tag,
                "split":     EVAL_SPLIT,
                "condition": COND_TO_FIELD[cond],
                **result,
            })

    df = pd.DataFrame(rows)
    cols = [
        "model", "split", "condition",
        "minority_macro_f1", "pred_comment_rate",
        "nc_n", "nc_macro_f1", "nc_support_f1", "nc_deny_f1", "nc_query_f1", "n",
    ]
    df = df[cols]
    print(df.to_string(index=False))
    out = OUTPUT / "minority_analysis.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
