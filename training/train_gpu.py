"""GPU convenience wrapper for training/train.py."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.train import main


def _ensure_gpu_defaults() -> None:
    if "--device" not in sys.argv:
        sys.argv.extend(["--device", "cuda"])
    if "--dtype" not in sys.argv:
        sys.argv.extend(["--dtype", "float16"])


if __name__ == "__main__":
    _ensure_gpu_defaults()
    main()
