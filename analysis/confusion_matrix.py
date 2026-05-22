"""
Confusion matrix plots for selected model/condition pairs.

Input:  data/results/experiment_results_*.json
        Format: {condition: {golds:[int], preds:[int], ...}}
        Label int: 0=support 1=deny 2=query 3=comment

Output: figures/cm_{model}_{condition}.png
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DATA_RESULTS, FIGURES, INT2LABEL, LABELS

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# (experiment_results filename, condition)
TARGETS: list[tuple[str, str]] = [
    ("experiment_results_1.5B_adv.json", "reply_only"),
    ("experiment_results_1.5B_adv.json", "useful"),
    ("experiment_results_3B_zeroshot.json", "reply_only"),
]


def load_exp(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def plot_cm(golds: list[int], preds: list[int], title: str, out_path: Path) -> None:
    gold_strs = [INT2LABEL[g] for g in golds]
    pred_strs = [INT2LABEL[p] for p in preds]
    cm = confusion_matrix(gold_strs, pred_strs, labels=LABELS)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved → {out_path}")


def main() -> None:
    for fname, cond in TARGETS:
        path = DATA_RESULTS / fname
        if not path.exists():
            print(f"[skip] {path} not found")
            continue
        data = load_exp(path)
        if cond not in data:
            print(f"[skip] condition '{cond}' not in {fname}")
            continue
        model_tag = fname.replace("experiment_results_", "").replace(".json", "")
        golds = data[cond]["golds"]
        preds = data[cond]["preds"]
        title = f"{model_tag} | {cond}"
        out_path = FIGURES / f"cm_{model_tag}_{cond}.png"
        plot_cm(golds, preds, title, out_path)


if __name__ == "__main__":
    main()
