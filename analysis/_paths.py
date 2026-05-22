"""Shared path constants for all analysis scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RESULTS = ROOT / "data" / "results"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_RAW = ROOT / "data" / "raw"
FIGURES = ROOT / "figures"
OUTPUT = ROOT / "analysis" / "output"

# Ensure output dirs exist at import time
OUTPUT.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# Label encoding used in experiment_results golds/preds
INT2LABEL: dict[int, str] = {0: "support", 1: "deny", 2: "query", 3: "comment"}
LABEL2INT: dict[str, int] = {v: k for k, v in INT2LABEL.items()}
LABELS: list[str] = ["support", "deny", "query", "comment"]
CONDITIONS: list[str] = ["reply_only", "useful", "irrelevant", "conflicting", "mixed", "lexical"]

# experiment_results filename → model tag
MODEL_FILES: dict[str, str] = {
    "experiment_results_0.5B.json": "0.5B",
    "experiment_results_1.5B_adv.json": "1.5B_adv",
    "experiment_results_3B_zeroshot.json": "3B_zero",
}

# confidence_results filename → model tag
CONF_FILES: dict[str, str] = {
    "confidence_results_1.5b_ft.json": "1.5B_ft",
    "confidence_results_1.5b_adv.json": "1.5B_adv",
}
