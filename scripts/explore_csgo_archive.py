from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from noscope_bio.csgo_pipeline import RAW_FEATURE_NAMES, engineer_csgo_engagement, load_csgo_archive_arrays


def describe_archive():
    legit, cheaters = load_csgo_archive_arrays()
    print("legit", legit.shape, legit.dtype)
    print("cheaters", cheaters.shape, cheaters.dtype)
    for name, arr in [("legit", legit), ("cheater", cheaters)]:
        print(f"\n{name} summary")
        for idx, feature in enumerate(RAW_FEATURE_NAMES):
            values = arr[..., idx]
            print(f"  {feature}: mean={float(values.mean()):.4f} std={float(values.std()):.4f} min={float(values.min()):.4f} max={float(values.max()):.4f}")


def show_engineered_example():
    legit, _ = load_csgo_archive_arrays()
    frame = engineer_csgo_engagement(legit[0, 0], "L00000_E00", "L00000", "legit", "example", 0)
    cols = [
        "t",
        "view_yaw",
        "view_pitch",
        "target_error",
        "angular_speed",
        "stability_score",
        "fire_input",
        "flick_event",
        "fire_motion_coupling",
    ]
    print("\nengineered legit sample")
    print(frame[cols].head(12).to_string(index=False))
    print("\nsummary")
    print(frame[cols].describe().to_string())


def compare_fire_alignment():
    legit, cheaters = load_csgo_archive_arrays()
    sample_legit = legit[:300]
    sample_cheat = cheaters[:300]

    def describe(arr: np.ndarray):
        error = np.sqrt(arr[..., 2] ** 2 + arr[..., 3] ** 2)
        angular_speed = np.sqrt(arr[..., 0] ** 2 + arr[..., 1] ** 2)
        fire_mask = arr[..., 4] > 0.5
        fire_error = error[fire_mask]
        fire_speed = angular_speed[fire_mask]
        return {
            "fire_rate": float(arr[..., 4].mean()),
            "error_at_fire_mean": float(fire_error.mean()),
            "error_at_fire_lt_2deg": float((fire_error <= 2.0).mean()),
            "angular_speed_at_fire_mean": float(fire_speed.mean()),
            "angular_speed_at_fire_lt_0.5": float((fire_speed <= 0.5).mean()),
        }

    summary = pd.DataFrame([describe(sample_legit), describe(sample_cheat)], index=["legit", "cheater"])
    print("\nfire and alignment comparison")
    print(summary.to_string())


def main():
    describe_archive()
    show_engineered_example()
    compare_fire_alignment()


if __name__ == "__main__":
    main()
