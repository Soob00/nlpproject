"""
Case study: qualitative examples for the discussion section.

Finds:
  A) Support failures  : gold=support, pred≠support in useful condition
  B) Query improvements: gold=query, wrong in reply_only → correct in useful

Input:
  data/results/experiment_results_1.5B_adv.json
  data/processed/context_conditions.json

Output:
  analysis/output/case_study_support_failures.txt
  analysis/output/case_study_query_improvements.txt
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DATA_RESULTS, DATA_PROCESSED, OUTPUT, INT2LABEL, EVAL_SPLIT, load_exp

N_EXAMPLES = 5


def load_ctx(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_pred_map(split_data: dict, condition: str) -> dict[str, tuple[str, str]]:
    if condition not in split_data:
        return {}
    cond = split_data[condition]
    golds = cond["golds"]
    preds = cond["preds"]
    reply_ids = cond.get("reply_ids")
    if reply_ids and len(reply_ids) == len(golds):
        return {
            rid: (INT2LABEL.get(g, "invalid"), INT2LABEL.get(p, "invalid"))
            for rid, g, p in zip(reply_ids, golds, preds)
        }
    return {
        str(i): (INT2LABEL.get(g, "invalid"), INT2LABEL.get(p, "invalid"))
        for i, (g, p) in enumerate(zip(golds, preds))
    }


def find_support_failures(
    ctx_records: list[dict],
    pred_map: dict[str, tuple[str, str]],
    n: int = N_EXAMPLES,
) -> list[dict]:
    results = []
    dev_records = [r for r in ctx_records if r.get("split") == EVAL_SPLIT]
    for i, r in enumerate(dev_records):
        key = r["reply_id"] if r["reply_id"] in pred_map else str(i)
        if key not in pred_map:
            continue
        gold, pred = pred_map[key]
        if gold == "support" and pred != "support":
            results.append({
                "reply_id":  r["reply_id"],
                "gold":      gold,
                "pred":      pred,
                "reply_only_text": (r.get("reply_only") or "")[:300],
                "useful_text":     (r.get("useful")     or "")[:400],
            })
        if len(results) >= n:
            break
    return results


def find_query_improvements(
    ctx_records: list[dict],
    c0_pred_map: dict[str, tuple[str, str]],
    c1_pred_map: dict[str, tuple[str, str]],
    n: int = N_EXAMPLES,
) -> list[dict]:
    results = []
    dev_records = [r for r in ctx_records if r.get("split") == EVAL_SPLIT]
    for i, r in enumerate(dev_records):
        c0_key = r["reply_id"] if r["reply_id"] in c0_pred_map else str(i)
        c1_key = r["reply_id"] if r["reply_id"] in c1_pred_map else str(i)
        if c0_key not in c0_pred_map or c1_key not in c1_pred_map:
            continue
        gold_r, pred_r = c0_pred_map[c0_key]
        _,      pred_u = c1_pred_map[c1_key]
        if gold_r == "query" and pred_r != "query" and pred_u == "query":
            results.append({
                "reply_id":        r["reply_id"],
                "gold":            gold_r,
                "reply_only_pred": pred_r,
                "useful_pred":     pred_u,
                "reply_only_text": (r.get("reply_only") or "")[:300],
                "useful_text":     (r.get("useful")     or "")[:400],
            })
        if len(results) >= n:
            break
    return results


def print_and_save(examples: list[dict], title: str, out_path: Path) -> None:
    lines = [f"=== {title} ===\n"]
    for ex in examples:
        lines.append(f"[reply_id: {ex['reply_id']}]")
        for k, v in ex.items():
            if k == "reply_id":
                continue
            lines.append(f"  {k}: {v}")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    out_path.write_text(text, encoding="utf-8")
    print(f"Saved → {out_path}")


def main() -> None:
    exp_path = DATA_RESULTS / "experiment_results_1.5B_adv.json"
    ctx_path = DATA_PROCESSED / "context_conditions.json"

    for p in (exp_path, ctx_path):
        if not p.exists():
            print(f"[error] {p} not found")
            return

    split_data = load_exp(exp_path, EVAL_SPLIT)
    ctx        = load_ctx(ctx_path)

    c0_preds = build_pred_map(split_data, "c0")
    c1_preds = build_pred_map(split_data, "c1")

    failures     = find_support_failures(ctx, c1_preds)
    improvements = find_query_improvements(ctx, c0_preds, c1_preds)

    print_and_save(failures,     "Support Failures (condition=useful)",     OUTPUT / "case_study_support_failures.txt")
    print_and_save(improvements, "Query Improvements (reply_only → useful)", OUTPUT / "case_study_query_improvements.txt")


if __name__ == "__main__":
    main()
