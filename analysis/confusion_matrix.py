"""
Confusion matrix figures for selected model/condition pairs.

Input:  data/results/experiment_results_*.json
Output: figures/cm_{model}_{condition}.png

Edit TARGETS below to choose which plots to generate.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DATA_RESULTS, FIGURES, INT2LABEL, PRED_LABELS, EVAL_SPLIT, load_exp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# (result filename, condition key, display title)
TARGETS: list[tuple[str, str, str]] = [
    ("experiment_results_1.5B_adv.json", "c0", "1.5B_adv | reply_only"),
    ("experiment_results_1.5B_adv.json", "c1", "1.5B_adv | useful"),
    ("experiment_results_3B_zs.json",    "c0", "3B_zs | reply_only"),
]


def plot_cm(golds: list[int], preds: list[int], title: str, out_path: Path) -> None:
    gold_strs = [INT2LABEL.get(g, "invalid") for g in golds]
    pred_strs = [INT2LABEL.get(p, "invalid") for p in preds]
    cm   = confusion_matrix(gold_strs, pred_strs, labels=PRED_LABELS)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=PRED_LABELS)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved → {out_path}")


def main() -> None:
    for fname, cond, title in TARGETS:
        path = DATA_RESULTS / fname
        if not path.exists():
            print(f"[skip] {path.name} not found")
            continue
        split_data = load_exp(path, EVAL_SPLIT)
        if cond not in split_data:
            print(f"[skip] condition '{cond}' not in {path.name}")
            continue
        golds = split_data[cond]["golds"]
        preds = split_data[cond]["preds"]
        # sanitise filename
        tag      = fname.replace("experiment_results_", "").replace(".json", "")
        out_path = FIGURES / f"cm_{tag}_{cond}.png"
        plot_cm(golds, preds, title, out_path)


if __name__ == "__main__":
    main()
