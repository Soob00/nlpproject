"""
Case study: qualitative examples for discussion section.

Finds:
  A) Support failures:  gold=support, pred≠support, in useful condition
  B) Query improvements: gold=query, wrong in reply_only → correct in useful

Input:
  data/results/experiment_results_1.5B_adv.json
    {condition: {golds:[int], preds:[int]}}  — same order as context_conditions dev split
  data/processed/context_conditions.json
    [{reply_id, label, split, reply_only:str, useful:str|null, ...}]

Output: prints examples to stdout (manual selection for paper)
        analysis/output/case_study_support_failures.txt
        analysis/output/case_study_query_improvements.txt
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DATA_RESULTS, DATA_PROCESSED, OUTPUT, INT2LABEL

N_EXAMPLES = 5


def load_exp(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_ctx(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_pred_map(data: dict, condition: str) -> dict[int, tuple[str, str]]:
    """Returns {index: (gold_label, pred_label)} for a condition."""
    if condition not in data:
        return {}
    golds = data[condition]["golds"]
    preds = data[condition]["preds"]
    return {i: (INT2LABEL[g], INT2LABEL[p]) for i, (g, p) in enumerate(zip(golds, preds))}


def find_support_failures(
    ctx_records: list[dict], pred_map: dict[int, tuple[str, str]], n: int = N_EXAMPLES
) -> list[dict]:
    results = []
    # context_conditions is ordered same as experiment_results (dev split only)
    dev_records = [r for r in ctx_records if r.get("split") == "dev"]
    for i, r in enumerate(dev_records):
        if i not in pred_map:
            continue
        gold, pred = pred_map[i]
        if gold == "support" and pred != "support":
            results.append({
                "reply_id": r["reply_id"],
                "gold": gold,
                "pred": pred,
                "reply_only_text": (r.get("reply_only") or "")[:300],
                "useful_text": (r.get("useful") or "")[:400],
            })
        if len(results) >= n:
            break
    return results


def find_query_improvements(
    ctx_records: list[dict],
    reply_pred_map: dict[int, tuple[str, str]],
    useful_pred_map: dict[int, tuple[str, str]],
    n: int = N_EXAMPLES,
) -> list[dict]:
    results = []
    dev_records = [r for r in ctx_records if r.get("split") == "dev"]
    for i, r in enumerate(dev_records):
        if i not in reply_pred_map or i not in useful_pred_map:
            continue
        gold_r, pred_r = reply_pred_map[i]
        gold_u, pred_u = useful_pred_map[i]
        if gold_r == "query" and pred_r != "query" and pred_u == "query":
            results.append({
                "reply_id": r["reply_id"],
                "gold": gold_r,
                "reply_only_pred": pred_r,
                "useful_pred": pred_u,
                "reply_only_text": (r.get("reply_only") or "")[:300],
                "useful_text": (r.get("useful") or "")[:400],
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

    if not exp_path.exists():
        print(f"[error] {exp_path} not found")
        return
    if not ctx_path.exists():
        print(f"[error] {ctx_path} not found")
        return

    data = load_exp(exp_path)
    ctx = load_ctx(ctx_path)

    reply_preds = build_pred_map(data, "reply_only")
    useful_preds = build_pred_map(data, "useful")

    failures = find_support_failures(ctx, useful_preds)
    print_and_save(failures, "Support Failures (condition=useful)", OUTPUT / "case_study_support_failures.txt")

    improvements = find_query_improvements(ctx, reply_preds, useful_preds)
    print_and_save(improvements, "Query Improvements (reply_only→useful)", OUTPUT / "case_study_query_improvements.txt")


if __name__ == "__main__":
    main()
