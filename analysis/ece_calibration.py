"""
ECE (Expected Calibration Error) and confidence statistics per model/condition.

Input:  data/results/confidence_results_*.json
        Format: list of records with fields:
          reply_id, split, true_label,
          {cond}_pred, {cond}_conf, {cond}_correct  (for each condition)

Output: analysis/output/ece_calibration.csv

Confidence = max softmax prob over {support, deny, query, comment} label tokens.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import (
    DATA_RESULTS, OUTPUT, CONF_FILES, CONDITIONS, COND_TO_FIELD, EVAL_SPLIT, load_conf,
)

import numpy as np
import pandas as pd

N_BINS = 10


def compute_ece(confs: list[float], corrects: list[int], n_bins: int = N_BINS) -> float:
    bins     = np.linspace(0.0, 1.0, n_bins + 1)
    conf_arr = np.array(confs)
    corr_arr = np.array(corrects, dtype=float)
    ece = 0.0
    n   = len(conf_arr)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf_arr > lo) & (conf_arr <= hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n * abs(corr_arr[mask].mean() - conf_arr[mask].mean())
    return round(float(ece), 4)


def condition_stats(records: list[dict], cond: str) -> dict | None:
    conf_key    = f"{cond}_conf"
    correct_key = f"{cond}_correct"
    subset = [r for r in records if r.get(conf_key) is not None]
    if not subset:
        return None

    confs    = [r[conf_key]    for r in subset]
    corrects = [r[correct_key] for r in subset]
    correct_confs = [c for c, ok in zip(confs, corrects) if ok == 1]
    wrong_confs   = [c for c, ok in zip(confs, corrects) if ok == 0]

    return {
        "n":            len(subset),
        "accuracy":     round(sum(corrects) / len(corrects), 4),
        "avg_conf":     round(float(np.mean(confs)), 4),
        "correct_conf": round(float(np.mean(correct_confs)), 4) if correct_confs else float("nan"),
        "wrong_conf":   round(float(np.mean(wrong_confs)),   4) if wrong_confs   else float("nan"),
        "ece":          compute_ece(confs, corrects),
    }


def main() -> None:
    rows = []
    for fname, model_tag in CONF_FILES.items():
        path = DATA_RESULTS / fname
        if not path.exists():
            print(f"[skip] {path.name} not found")
            continue
        records = load_conf(path, split=EVAL_SPLIT)
        if not records:
            print(f"[skip] no records for split='{EVAL_SPLIT}' in {path.name}")
            continue
        for cond in CONDITIONS:
            stats = condition_stats(records, cond)
            if stats is None:
                continue
            rows.append({
                "model":     model_tag,
                "split":     EVAL_SPLIT,
                "condition": COND_TO_FIELD[cond],
                **stats,
            })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    out = OUTPUT / "ece_calibration.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
