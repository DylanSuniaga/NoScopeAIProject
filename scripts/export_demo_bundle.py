from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from noscope_bio.pipeline import DATA_DIR, export_demo_bundle


def main():
    artifacts = export_demo_bundle()
    print(f"Exported demo bundle to {DATA_DIR}")
    print("Evaluation metrics:")
    for key, value in artifacts["evaluation_metrics"].items():
        print(f"  {key}: {value:.3f}")


if __name__ == "__main__":
    main()
