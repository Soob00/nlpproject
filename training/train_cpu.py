"""CPU convenience wrapper for training/train.py.

CPU LoRA training is provided for completeness, but it is usually very slow.
Use it mainly for smoke tests or very small runs.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.train import main


def _ensure_cpu_defaults() -> None:
    if "--device" not in sys.argv:
        sys.argv.extend(["--device", "cpu"])
    if "--dtype" not in sys.argv:
        sys.argv.extend(["--dtype", "float32"])


if __name__ == "__main__":
    _ensure_cpu_defaults()
    main()
