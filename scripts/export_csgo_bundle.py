from __future__ import annotations

import argparse
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

from noscope_bio.csgo_pipeline import CSGO_DATA_DIR, export_csgo_bundle


def parse_args():
    parser = argparse.ArgumentParser(description="Build the exported CSGO archive bundle.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--legit-limit", type=int, default=None, help="Optional number of legit players to sample from the archive.")
    parser.add_argument("--cheat-limit", type=int, default=None, help="Optional number of cheater players to sample from the archive.")
    return parser.parse_args()


def main():
    args = parse_args()
    kwargs = {"seed": args.seed}
    if args.legit_limit is not None:
        kwargs["legit_limit"] = args.legit_limit
    if args.cheat_limit is not None:
        kwargs["cheat_limit"] = args.cheat_limit
    artifacts = export_csgo_bundle(**kwargs)
    print(f"Exported CSGO bundle to {CSGO_DATA_DIR}")
    print("Archive summary:")
    for key, value in artifacts["archive_summary"].items():
        print(f"  {key}: {value}")
    print("Model summary:")
    print("  window_encoder: sklearn MLPClassifier over flattened causal windows")
    print("  cheat_scorer: sklearn LogisticRegression over engineered window shifts")
    print("Evaluation metrics:")
    for key, value in artifacts["evaluation_metrics"].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    print("Split metrics:")
    for split, metrics in artifacts["split_metrics"].items():
        print(f"  [{split}]")
        for key in ["accuracy", "balanced_accuracy", "precision", "recall", "specificity", "false_positive_rate", "false_negative_rate"]:
            value = metrics[key]
            print(f"    {key}: {value:.3f}")


if __name__ == "__main__":
    main()
