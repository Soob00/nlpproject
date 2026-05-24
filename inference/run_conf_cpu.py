"""CPU convenience wrapper for inference/run_conf.py."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.run_conf import main


def _ensure_cpu_defaults() -> None:
    if "--device" not in sys.argv:
        sys.argv.extend(["--device", "cpu"])
    if "--dtype" not in sys.argv:
        sys.argv.extend(["--dtype", "float32"])


if __name__ == "__main__":
    _ensure_cpu_defaults()
    main()
