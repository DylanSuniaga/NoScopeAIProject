from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SUMMARY_FEATURES


def summarize_window(window: pd.DataFrame) -> dict[str, float]:
    summary = {}
    for feature, aggregations in SUMMARY_FEATURES.items():
        values = window[feature].to_numpy()
        for agg in aggregations:
            key = f"{feature}_{agg}"
            if agg == "mean":
                summary[key] = float(np.mean(values))
            elif agg == "std":
                summary[key] = float(np.std(values))
            elif agg == "max":
                summary[key] = float(np.max(values))
    return summary


def window_session(
    session_df: pd.DataFrame,
    feature_columns: list[str],
    window_size: int = 32,
    stride: int = 8,
) -> tuple[np.ndarray, pd.DataFrame]:
    X = []
    rows = []
    session_df = session_df.sort_values("tick").reset_index(drop=True)

    for start in range(0, len(session_df) - window_size + 1, stride):
        end = start + window_size
        window = session_df.iloc[start:end]
        X.append(window[feature_columns].to_numpy(dtype=np.float32))
        row = {
            "session_id": window["session_id"].iloc[0],
            "player_id": window["player_id"].iloc[0],
            "mode": window["mode"].iloc[0],
            "start_tick": int(window["tick"].iloc[0]),
            "end_tick": int(window["tick"].iloc[-1]),
            "t_end_s": float(window["t"].iloc[-1]),
            "label_cheat": int(window["label_cheat"].iloc[0]),
            "window_cheat_fraction": float(window["cheat_active"].mean()) if "cheat_active" in window.columns else float(window["label_cheat"].iloc[0]),
            "window_confounder_fraction": float(window["confounder_active"].mean()) if "confounder_active" in window.columns else 0.0,
        }
        row.update(summarize_window(window))
        rows.append(row)

    return np.stack(X), pd.DataFrame(rows)


def combine_window_batches(
    sessions: dict[str, pd.DataFrame],
    feature_columns: list[str],
    window_size: int = 32,
    stride: int = 8,
) -> tuple[np.ndarray, pd.DataFrame]:
    arrays = []
    meta_frames = []
    for session_df in sessions.values():
        X, meta = window_session(session_df, feature_columns, window_size=window_size, stride=stride)
        arrays.append(X)
        meta_frames.append(meta)
    return np.concatenate(arrays, axis=0), pd.concat(meta_frames, ignore_index=True)
