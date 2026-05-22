"""
Paired t-tests on sample-level confidence values.

Input:  data/results/confidence_results_*.json
        Format: list of records keyed by reply_id,
                {cond}_conf, {cond}_correct per condition

Output: analysis/output/significance_test.csv

Comparisons:
  1. FT correct conf vs FT wrong conf  (does FT overconfide when wrong?)
  2. FT wrong conf vs ADV wrong conf   (does ADV reduce wrong confidence?)
  3. FT reply_only conf vs FT useful/conflicting conf (does context raise confidence?)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DATA_RESULTS, OUTPUT

from scipy import stats
import numpy as np
import pandas as pd


def load_conf(path: Path) -> dict[str, dict]:
    """Returns {reply_id: record} mapping."""
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    return {r["reply_id"]: r for r in records}


def get_confs(records: dict[str, dict], cond: str, correct_only: bool | None = None) -> dict[str, float]:
    """Returns {reply_id: conf} filtered by correctness if specified."""
    out = {}
    for rid, r in records.items():
        conf = r.get(f"{cond}_conf")
        correct = r.get(f"{cond}_correct")
        if conf is None:
            continue
        if correct_only is True and correct != 1:
            continue
        if correct_only is False and correct != 0:
            continue
        out[rid] = conf
    return out


def paired_test(a_map: dict[str, float], b_map: dict[str, float], label: str) -> dict | None:
    """Paired t-test: same reply_ids must exist in both maps."""
    shared = sorted(set(a_map) & set(b_map))
    if len(shared) < 10:
        return None
    a = [a_map[i] for i in shared]
    b = [b_map[i] for i in shared]
    t, p = stats.ttest_rel(a, b)
    mean_diff = float(np.mean(np.array(a) - np.array(b)))
    return {
        "comparison": label,
        "test": "paired t-test",
        "n": len(shared),
        "mean_diff": round(mean_diff, 4),
        "t_stat": round(float(t), 4),
        "p_value": round(float(p), 6),
        "significant": p < 0.05,
    }


def welch_test(a_vals: list[float], b_vals: list[float], label: str) -> dict | None:
    """Welch t-test: for independent samples (e.g. correct vs wrong predictions)."""
    if len(a_vals) < 5 or len(b_vals) < 5:
        return None
    t, p = stats.ttest_ind(a_vals, b_vals, equal_var=False)
    mean_diff = float(np.mean(a_vals) - np.mean(b_vals))
    return {
        "comparison": label,
        "test": "Welch t-test",
        "n": len(a_vals) + len(b_vals),
        "mean_diff": round(mean_diff, 4),
        "t_stat": round(float(t), 4),
        "p_value": round(float(p), 6),
        "significant": p < 0.05,
    }


def main() -> None:
    ft_path = DATA_RESULTS / "confidence_results_1.5b_ft.json"
    adv_path = DATA_RESULTS / "confidence_results_1.5b_adv.json"

    if not ft_path.exists():
        print(f"[error] {ft_path} not found")
        return
    if not adv_path.exists():
        print(f"[error] {adv_path} not found")
        return

    ft = load_conf(ft_path)
    adv = load_conf(adv_path)
    rows = []

    for cond in ["reply_only", "useful", "conflicting"]:
        # 1. FT wrong conf vs FT correct conf (independent groups → Welch t-test)
        ft_wrong = get_confs(ft, cond, correct_only=False)
        ft_correct = get_confs(ft, cond, correct_only=True)
        row = welch_test(list(ft_wrong.values()), list(ft_correct.values()), f"FT wrong - correct conf | {cond}")
        if row:
            rows.append(row)

        # 2. FT wrong conf vs ADV wrong conf (same reply_ids → paired t-test)
        adv_wrong = get_confs(adv, cond, correct_only=False)
        row = paired_test(ft_wrong, adv_wrong, f"FT wrong - ADV wrong | {cond}")
        if row:
            rows.append(row)

    # 3. FT reply_only conf vs useful / conflicting (same reply_ids → paired t-test)
    ft_reply = get_confs(ft, "reply_only")
    for cond in ["useful", "conflicting"]:
        ft_cond = get_confs(ft, cond)
        row = paired_test(ft_reply, ft_cond, f"FT reply_only - {cond}")
        if row:
            rows.append(row)

    if not rows:
        print("No results (not enough shared samples).")
        return

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    out = OUTPUT / "significance_test.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
