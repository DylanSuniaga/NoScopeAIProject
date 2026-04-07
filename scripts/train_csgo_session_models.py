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

from noscope_bio.csgo_offline_model import CSGO_MODEL_DIR, train_offline_csgo_models


def main():
    summary = train_offline_csgo_models()
    print(f"Saved CSGO models under {CSGO_MODEL_DIR}")
    print(f"Best model: {summary['best_model_name']}")
    print(f"Threshold: {summary['threshold']:.3f}")
    print("Validation metrics:")
    for key, value in summary["validation_metrics"].items():
        print(f"  {key}: {value:.3f}")
    print("Test metrics:")
    for key, value in summary["test_metrics"].items():
        print(f"  {key}: {value:.3f}")
    print("Candidate models:")
    for row in summary["candidate_models"]:
        print(
            f"  {row['model_name']}: "
            f"val_bal_acc={row['validation_metrics']['balanced_accuracy']:.3f}, "
            f"test_bal_acc={row['test_metrics']['balanced_accuracy']:.3f}, "
            f"test_precision={row['test_metrics']['precision']:.3f}, "
            f"test_recall={row['test_metrics']['recall']:.3f}"
        )


if __name__ == "__main__":
    main()
