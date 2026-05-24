"""Legacy wrapper for the 0.5B CPU zero-shot evaluation.

Prefer:
    python inference/run_eval_cpu.py --model-size 0.5b --variant zs --split dev
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inference.run_eval import main


def _set_legacy_defaults() -> None:
    defaults = {
        "--model-size": "0.5b",
        "--variant": "zs",
        "--split": "dev",
        "--device": "cpu",
        "--dtype": "float32",
    }
    for flag, value in defaults.items():
        if flag not in sys.argv:
            sys.argv.extend([flag, value])


if __name__ == "__main__":
    _set_legacy_defaults()
    main()
