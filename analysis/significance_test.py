"""
Paired / Welch t-tests on sample-level confidence values.

Input:  data/results/confidence_results_*.json

Output: analysis/output/significance_test.csv

Comparisons (all use real condition names in output):
  1. FT wrong conf vs FT correct conf      → are FT predictions overconfident when wrong?
  2. FT wrong conf vs ADV wrong conf       → does ADV training reduce wrong confidence?
  3. FT reply_only conf vs FT useful conf  → does context raise confidence?
  4. FT reply_only conf vs FT conflicting  → does conflicting context raise confidence?
  5. FT useful conf vs FT conflicting conf → is misleading context worse than useful?
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DATA_RESULTS, OUTPUT, EVAL_SPLIT, load_conf

import numpy as np
import pandas as pd
from scipy import stats


# ── helpers ────────────────────────────────────────────────────────────────────

def load_by_id(path: Path) -> dict[str, dict]:
    records = load_conf(path, split=EVAL_SPLIT)
    return {r["reply_id"]: r for r in records}


def resolve_result_file(name: str) -> Path:
    path = DATA_RESULTS / name
    if path.exists():
        return path
    lower_name = name.replace("1.5B", "1.5b")
    lower_path = DATA_RESULTS / lower_name
    if lower_path.exists():
        return lower_path
    return path


def get_confs(
    records: dict[str, dict],
    cond: str,
    correct_only: bool | None = None,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for rid, r in records.items():
        conf    = r.get(f"{cond}_conf")
        correct = r.get(f"{cond}_correct")
        if conf is None:
            continue
        if correct_only is True  and correct != 1:
            continue
        if correct_only is False and correct != 0:
            continue
        out[rid] = conf
    return out


def paired_test(
    a_map: dict[str, float],
    b_map: dict[str, float],
    label: str,
) -> dict | None:
    shared = sorted(set(a_map) & set(b_map))
    if len(shared) < 10:
        return None
    a = [a_map[i] for i in shared]
    b = [b_map[i] for i in shared]
    t, p = stats.ttest_rel(a, b)
    return {
        "comparison": label,
        "test":       "paired t-test",
        "n":          len(shared),
        "mean_diff":  round(float(np.mean(np.array(a) - np.array(b))), 4),
        "t_stat":     round(float(t), 4),
        "p_value":    round(float(p), 6),
        "significant": p < 0.05,
    }


def welch_test(
    a_vals: list[float],
    b_vals: list[float],
    label: str,
) -> dict | None:
    if len(a_vals) < 5 or len(b_vals) < 5:
        return None
    t, p = stats.ttest_ind(a_vals, b_vals, equal_var=False)
    return {
        "comparison": label,
        "test":       "Welch t-test",
        "n":          len(a_vals) + len(b_vals),
        "mean_diff":  round(float(np.mean(a_vals) - np.mean(b_vals)), 4),
        "t_stat":     round(float(t), 4),
        "p_value":    round(float(p), 6),
        "significant": p < 0.05,
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ft_path  = resolve_result_file("confidence_results_1.5B_ft.json")
    adv_path = resolve_result_file("confidence_results_1.5B_adv.json")

    for p in (ft_path, adv_path):
        if not p.exists():
            print(f"[error] {p} not found")
            return

    ft  = load_by_id(ft_path)
    adv = load_by_id(adv_path)
    rows = []

    # condition keys and readable names for comparisons
    COND_PAIRS = [
        ("c0", "reply_only"),
        ("c1", "useful"),
        ("c3", "conflicting"),
    ]

    # 1. FT wrong conf vs FT correct conf (independent → Welch)
    for cond, cname in COND_PAIRS:
        ft_wrong   = get_confs(ft, cond, correct_only=False)
        ft_correct = get_confs(ft, cond, correct_only=True)
        row = welch_test(
            list(ft_wrong.values()), list(ft_correct.values()),
            f"FT wrong - correct conf | {cname}",
        )
        if row:
            rows.append(row)

    # 2. FT wrong conf vs ADV wrong conf (same reply_ids → paired)
    for cond, cname in COND_PAIRS:
        ft_wrong  = get_confs(ft,  cond, correct_only=False)
        adv_wrong = get_confs(adv, cond, correct_only=False)
        row = paired_test(ft_wrong, adv_wrong, f"FT wrong - ADV wrong | {cname}")
        if row:
            rows.append(row)

    # 3. FT reply_only vs useful / conflicting (context raises confidence?)
    ft_reply = get_confs(ft, "c0")
    for cond, cname in [("c1", "useful"), ("c3", "conflicting")]:
        ft_cond = get_confs(ft, cond)
        row = paired_test(ft_reply, ft_cond, f"FT reply_only - {cname}")
        if row:
            rows.append(row)

    # 4. FT useful vs FT conflicting (misleading vs helpful context)
    ft_useful      = get_confs(ft, "c1")
    ft_conflicting = get_confs(ft, "c3")
    row = paired_test(ft_useful, ft_conflicting, "FT useful - conflicting")
    if row:
        rows.append(row)

    if not rows:
        print("No results (insufficient shared samples).")
        return

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    out = OUTPUT / "significance_test.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
