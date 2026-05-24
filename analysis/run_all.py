"""Run the paper analysis scripts in a stable order.

Analysis is CPU-only and does not load language models.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SCRIPTS = [
    "dataset_validity.py",
    "dataset_validity_table.py",
    "majority_baseline.py",
    "per_class_f1.py",
    "label_distribution.py",
    "minority_analysis.py",
    "ece_calibration.py",
    "significance_test.py",
    "case_study.py",
    "confusion_matrix.py",
]


def main() -> None:
    for script in SCRIPTS:
        path = ROOT / script
        print(f"\n=== {script} ===")
        result = subprocess.run([sys.executable, str(path)], cwd=ROOT.parent, check=False)
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
