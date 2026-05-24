"""Shared path constants and helpers for all analysis scripts."""
from __future__ import annotations

import json
from pathlib import Path

ROOT          = Path(__file__).resolve().parent.parent
DATA_RESULTS  = ROOT / "data" / "results"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_RAW      = ROOT / "data" / "raw"
FIGURES       = ROOT / "figures"
OUTPUT        = ROOT / "analysis" / "output"

OUTPUT.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# ── label encoding ─────────────────────────────────────────────────────────────
LABELS:    list[str]      = ["support", "deny", "query", "comment"]
LABEL2INT: dict[str, int] = {label: i for i, label in enumerate(LABELS)}
INT2LABEL: dict[int, str] = {v: k for k, v in LABEL2INT.items()} | {-1: "invalid"}
PRED_LABELS: list[str]    = LABELS + ["invalid"]

# ── condition mappings ─────────────────────────────────────────────────────────
CONDITIONS:      list[str]       = ["c0", "c1", "c2", "c3", "c4", "c5"]
CONDITION_FIELDS: list[str]      = ["reply_only", "useful", "irrelevant",
                                    "conflicting", "mixed", "lexical"]
COND_TO_FIELD:   dict[str, str]  = dict(zip(CONDITIONS, CONDITION_FIELDS))
FIELD_TO_COND:   dict[str, str]  = dict(zip(CONDITION_FIELDS, CONDITIONS))

# ── split to evaluate ──────────────────────────────────────────────────────────
# Change to "test" to evaluate on the held-out test set.
EVAL_SPLIT: str = "dev"

# ── experiment result files ────────────────────────────────────────────────────
# Keys are filenames; values are display tags used in all output tables.
# Add/remove entries here to include/exclude models from all analyses.
MODEL_FILES: dict[str, str] = {
    # zero-shot (base model, no fine-tuning)
    "experiment_results_0.5B_zs.json":  "0.5B_zs",
    "experiment_results_1.5B_zs.json":  "1.5B_zs",
    "experiment_results_3B_zs.json":    "3B_zs",
    "experiment_results_7B_zs.json":    "7B_zs",
    # fine-tuned
    "experiment_results_0.5B_ft.json":  "0.5B_ft",
    "experiment_results_1.5B_ft.json":  "1.5B_ft",
    "experiment_results_3B_ft.json":    "3B_ft",
    "experiment_results_7B_ft.json":    "7B_ft",
    # adversarial fine-tuned
    "experiment_results_0.5B_adv.json": "0.5B_adv",
    "experiment_results_1.5B_adv.json": "1.5B_adv",
    "experiment_results_3B_adv.json":   "3B_adv",
    "experiment_results_7B_adv.json":   "7B_adv",
}

# ── confidence result files ────────────────────────────────────────────────────
CONF_FILES: dict[str, str] = {
    "confidence_results_0.5B_zs.json":  "0.5B_zs",
    "confidence_results_1.5B_zs.json":  "1.5B_zs",
    "confidence_results_3B_zs.json":    "3B_zs",
    "confidence_results_7B_zs.json":    "7B_zs",
    "confidence_results_0.5B_ft.json":  "0.5B_ft",
    "confidence_results_0.5B_adv.json": "0.5B_adv",
    "confidence_results_1.5B_ft.json":  "1.5B_ft",
    "confidence_results_1.5B_adv.json": "1.5B_adv",
    "confidence_results_3B_ft.json":    "3B_ft",
    "confidence_results_3B_adv.json":   "3B_adv",
    "confidence_results_7B_ft.json":    "7B_ft",
    "confidence_results_7B_adv.json":   "7B_adv",
}

# ── loaders ────────────────────────────────────────────────────────────────────

def load_exp(path: Path, split: str = EVAL_SPLIT) -> dict:
    """
    Load experiment results for a given split.

    Handles two file formats:
      New: {"dev": {"c0": {...}, ...}, "test": {...}}
      Old: {"c0": {...}, ...}   (assumed to be dev split)
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if any(k in data for k in ("dev", "test")):
        return data.get(split, {})
    if any(k in data for k in FIELD_TO_COND):
        return {FIELD_TO_COND.get(k, k): v for k, v in data.items()}
    return data  # old format: treat as dev


def load_conf(path: Path, split: str | None = None) -> list[dict]:
    """
    Load confidence records.
    If split is given, filter to that split only.
    """
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    if split is not None:
        if any("split" in r for r in records):
            records = [r for r in records if r.get("split") == split]
    return records
