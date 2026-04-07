from __future__ import annotations

import os
import json
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from .detector import CheatScorer, PageHinkley, _absolute_shift, _negative_shift, _positive_shift
from .model import embed_windows, predict_window_proba, train_fingerprint_model


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_DIR = ROOT / "real_data" / "archive"
CSGO_DATA_DIR = ROOT / "data" / "csgo_generated"

RAW_FEATURE_NAMES = [
    "AttackerDeltaYaw",
    "AttackerDeltaPitch",
    "CrosshairToVictimYaw",
    "CrosshairToVictimPitch",
    "Firing",
]

CSGO_DT = 6.0 / 192.0
CSGO_ENGAGEMENT_LENGTH = 192
CSGO_ENGAGEMENTS_PER_PLAYER = 30
CSGO_WINDOW_SIZE = 32
CSGO_WINDOW_STRIDE = 8
CSGO_DEFAULT_LEGIT_PLAYERS = 180
CSGO_DEFAULT_CHEATER_PLAYERS = 90
CSGO_MAX_ENCODER_WINDOWS_PER_CLASS = 12000
CSGO_WINDOWS_PER_ENGAGEMENT_FOR_ENCODER = 4
CSGO_SAMPLE_SESSION_COUNT = 18
CSGO_ENCODER_THRESHOLD = 0.5

CSGO_TELEMETRY_COLUMNS = ["target_error", "angular_speed", "fire_input"]

CSGO_FEATURE_COLUMNS = [
    "yaw_delta",
    "pitch_delta",
    "crosshair_to_victim_yaw",
    "crosshair_to_victim_pitch",
    "angular_speed",
    "angular_speed_sq",
    "angular_acceleration",
    "angular_jerk",
    "target_error",
    "target_error_sq",
    "target_error_delta",
    "target_error_improvement",
    "error_speed_ratio",
    "error_improvement_ratio",
    "on_target_soft",
    "on_target_tight",
    "on_target_dwell_ms",
    "lock_indicator",
    "lock_dwell_ms",
    "lock_strength",
    "heading_change",
    "curvature",
    "view_straightness",
    "stability_score",
    "settling_ratio",
    "yaw_reversal",
    "pitch_reversal",
    "direction_entropy_short",
    "micro_correction_score",
    "micro_to_speed_ratio",
    "angular_speed_autocorr_short",
    "fire_input",
    "fire_on_target",
    "time_since_fire_ms",
    "fire_motion_coupling",
    "fire_alignment_strength",
    "fire_stability_synergy",
    "last_fire_interval_ms",
    "last_stabilization_to_fire_ms",
    "flick_event",
    "flick_magnitude",
    "time_since_flick_ms",
    "snap_indicator",
    "snap_power",
    "aim_tension",
    "aim_efficiency",
]

CSGO_SUMMARY_FEATURES = {
    "angular_speed": ["mean", "std", "max"],
    "angular_speed_sq": ["mean", "std", "max"],
    "angular_acceleration": ["mean", "std"],
    "angular_jerk": ["mean", "std"],
    "target_error": ["mean", "std", "min"],
    "target_error_sq": ["mean", "std"],
    "target_error_delta": ["mean", "std", "min"],
    "target_error_improvement": ["mean", "std", "max"],
    "error_speed_ratio": ["mean", "std", "min"],
    "error_improvement_ratio": ["mean", "std", "max"],
    "on_target_soft": ["mean"],
    "on_target_tight": ["mean"],
    "on_target_dwell_ms": ["mean", "max"],
    "lock_indicator": ["mean"],
    "lock_dwell_ms": ["mean", "max"],
    "lock_strength": ["mean", "max"],
    "heading_change": ["mean", "std"],
    "curvature": ["mean", "std"],
    "view_straightness": ["mean", "std", "max"],
    "stability_score": ["mean", "std", "max"],
    "settling_ratio": ["mean", "std", "max"],
    "yaw_reversal": ["mean"],
    "pitch_reversal": ["mean"],
    "direction_entropy_short": ["mean", "std"],
    "micro_correction_score": ["mean", "std"],
    "micro_to_speed_ratio": ["mean", "std"],
    "angular_speed_autocorr_short": ["mean", "std"],
    "fire_input": ["mean"],
    "fire_on_target": ["mean"],
    "time_since_fire_ms": ["mean", "std"],
    "fire_motion_coupling": ["mean", "max"],
    "fire_alignment_strength": ["mean", "max"],
    "fire_stability_synergy": ["mean", "max"],
    "last_fire_interval_ms": ["mean", "std"],
    "last_stabilization_to_fire_ms": ["mean", "std"],
    "flick_event": ["mean"],
    "flick_magnitude": ["mean", "std", "max"],
    "time_since_flick_ms": ["mean", "std"],
    "snap_indicator": ["mean", "max"],
    "snap_power": ["mean", "std", "max"],
    "aim_tension": ["mean", "std", "max"],
    "aim_efficiency": ["mean", "max"],
}

BAYES_PREVALENCE_SCENARIOS = [
    ("observed_test_prevalence", None),
    ("moderate_deployment_10pct", 0.10),
    ("rare_deployment_5pct", 0.05),
    ("rare_deployment_1pct", 0.01),
    ("very_rare_deployment_0.1pct", 0.001),
]


def _angle_diff(a: float, b: float) -> float:
    return float(np.arctan2(np.sin(a - b), np.cos(a - b)))


def _normalized_entropy(values: list[float], bins: int = 8) -> float:
    if len(values) < 3:
        return 1.0
    hist, _ = np.histogram(values, bins=bins, range=(-np.pi, np.pi))
    probs = hist.astype(np.float64)
    total = probs.sum()
    if total <= 0:
        return 1.0
    probs /= total
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs))
    return float(entropy / np.log(bins))


def _lag1_autocorr(values: list[float]) -> float:
    if len(values) < 4:
        return 0.5
    x = np.asarray(values, dtype=np.float64)
    x0 = x[:-1] - x[:-1].mean()
    x1 = x[1:] - x[1:].mean()
    denom = np.linalg.norm(x0) * np.linalg.norm(x1)
    if denom <= 1e-8:
        return 0.5
    corr = float(np.dot(x0, x1) / denom)
    return float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))


def _local_straightness(points: list[np.ndarray]) -> float:
    if len(points) < 3:
        return 0.5
    path = 0.0
    for idx in range(1, len(points)):
        path += float(np.linalg.norm(points[idx] - points[idx - 1]))
    chord = float(np.linalg.norm(points[-1] - points[0]))
    if path <= 1e-8:
        return 0.5
    return float(np.clip(chord / path, 0.0, 1.0))


def _std_floor(column: str) -> float:
    if "encoder_prob" in column:
        return 0.03
    if "ratio" in column:
        return 0.04
    if "strength" in column or "synergy" in column:
        return 0.03
    if "power" in column:
        return 0.06
    if "tension" in column:
        return 0.08
    if "target_error" in column:
        return 0.25
    if "angular_speed" in column:
        return 0.08
    if "angular_acceleration" in column or "angular_jerk" in column:
        return 0.05
    if "heading_change" in column or "curvature" in column:
        return 0.04
    if "straightness" in column or "stability" in column:
        return 0.03
    if "entropy" in column or "correction" in column:
        return 0.04
    if "reversal" in column or "fire_input" in column:
        return 0.02
    if "fire_motion_coupling" in column or "aim_efficiency" in column:
        return 0.04
    if column.endswith("_ms") or "_ms_" in column:
        return 10.0
    if "flick" in column or "snap" in column or "lock" in column or "on_target" in column:
        return 0.02
    return 0.03


def load_csgo_archive_arrays() -> tuple[np.memmap, np.memmap]:
    legit = np.load(ARCHIVE_DIR / "legit" / "legit.npy", mmap_mode="r")
    cheaters = np.load(ARCHIVE_DIR / "cheaters" / "cheaters.npy", mmap_mode="r")
    return legit, cheaters


def _player_split_indices(num_players: int, seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(num_players)
    train_end = int(0.7 * num_players)
    val_end = int(0.85 * num_players)
    return {
        "train": np.sort(order[:train_end]),
        "validation": np.sort(order[train_end:val_end]),
        "test": np.sort(order[val_end:]),
    }


def _summarize_feature(values: np.ndarray, agg: str) -> float:
    if agg == "mean":
        return float(np.mean(values))
    if agg == "std":
        return float(np.std(values))
    if agg == "max":
        return float(np.max(values))
    if agg == "min":
        return float(np.min(values))
    raise ValueError(f"Unsupported aggregation: {agg}")


def summarize_frame(frame: pd.DataFrame) -> dict[str, float]:
    summary: dict[str, float] = {}
    for feature, aggregations in CSGO_SUMMARY_FEATURES.items():
        values = frame[feature].to_numpy(dtype=np.float64)
        for agg in aggregations:
            summary[f"{feature}_{agg}"] = _summarize_feature(values, agg)
    return summary


def _window_session(
    frame: pd.DataFrame,
    window_size: int = CSGO_WINDOW_SIZE,
    stride: int = CSGO_WINDOW_STRIDE,
) -> tuple[np.ndarray, pd.DataFrame]:
    arrays = []
    rows = []
    frame = frame.sort_values("tick").reset_index(drop=True)
    for start in range(0, len(frame) - window_size + 1, stride):
        end = start + window_size
        window = frame.iloc[start:end]
        arrays.append(window[CSGO_FEATURE_COLUMNS].to_numpy(dtype=np.float32))
        row = {
            "session_id": window["session_id"].iloc[0],
            "player_id": window["player_id"].iloc[0],
            "player_index": int(window["player_index"].iloc[0]),
            "source_label": window["source_label"].iloc[0],
            "split": window["split"].iloc[0],
            "engagement_index": int(window["engagement_index"].iloc[0]),
            "window_index": len(rows),
            "start_tick": int(window["tick"].iloc[0]),
            "end_tick": int(window["tick"].iloc[-1]),
            "t_end_s": float(window["t"].iloc[-1]),
            "label_cheat": int(window["label_cheat"].iloc[0]),
        }
        row.update(summarize_frame(window))
        rows.append(row)
    return np.stack(arrays), pd.DataFrame(rows)


def engineer_csgo_engagement(
    sequence: np.ndarray,
    session_id: str,
    player_id: str,
    source_label: str,
    split: str,
    engagement_index: int,
    player_index: int = -1,
) -> pd.DataFrame:
    yaw_delta = sequence[:, 0].astype(np.float64)
    pitch_delta = sequence[:, 1].astype(np.float64)
    crosshair_yaw = sequence[:, 2].astype(np.float64)
    crosshair_pitch = sequence[:, 3].astype(np.float64)
    fire_input = sequence[:, 4].astype(np.float64)

    view_yaw = np.cumsum(yaw_delta)
    view_pitch = np.cumsum(pitch_delta)
    angular_speed = np.sqrt(yaw_delta**2 + pitch_delta**2)
    angular_acceleration = np.diff(angular_speed, prepend=angular_speed[0])
    angular_jerk = np.diff(angular_acceleration, prepend=angular_acceleration[0])
    heading = np.arctan2(pitch_delta, yaw_delta + 1e-8)
    heading_change = np.zeros_like(heading)
    for idx in range(1, len(heading)):
        heading_change[idx] = abs(_angle_diff(heading[idx], heading[idx - 1]))
    curvature = heading_change / np.maximum(angular_speed, 1e-4)

    target_error = np.sqrt(crosshair_yaw**2 + crosshair_pitch**2)
    target_error_delta = np.diff(target_error, prepend=target_error[0])
    target_error_improvement = np.maximum(np.roll(target_error, 1) - target_error, 0.0)
    target_error_improvement[0] = 0.0
    on_target_soft = (target_error < 2.0).astype(np.float64)
    on_target_tight = (target_error < 0.75).astype(np.float64)

    position_history: list[np.ndarray] = []
    heading_history: list[float] = []
    speed_history: list[float] = []
    heading_change_history: list[float] = []

    on_target_dwell_ms = np.zeros(len(sequence), dtype=np.float64)
    lock_indicator = np.zeros(len(sequence), dtype=np.float64)
    lock_dwell_ms = np.zeros(len(sequence), dtype=np.float64)
    view_straightness = np.zeros(len(sequence), dtype=np.float64)
    stability_score = np.zeros(len(sequence), dtype=np.float64)
    yaw_reversal = np.zeros(len(sequence), dtype=np.float64)
    pitch_reversal = np.zeros(len(sequence), dtype=np.float64)
    direction_entropy_short = np.ones(len(sequence), dtype=np.float64)
    micro_correction_score = np.zeros(len(sequence), dtype=np.float64)
    angular_speed_autocorr_short = np.full(len(sequence), 0.5, dtype=np.float64)
    fire_on_target = np.zeros(len(sequence), dtype=np.float64)
    time_since_fire_ms = np.zeros(len(sequence), dtype=np.float64)
    fire_motion_coupling = np.zeros(len(sequence), dtype=np.float64)
    last_fire_interval_ms = np.full(len(sequence), 1000.0, dtype=np.float64)
    last_stabilization_to_fire_ms = np.full(len(sequence), 1000.0, dtype=np.float64)
    flick_event = np.zeros(len(sequence), dtype=np.float64)
    flick_magnitude = np.zeros(len(sequence), dtype=np.float64)
    time_since_flick_ms = np.zeros(len(sequence), dtype=np.float64)
    snap_indicator = np.zeros(len(sequence), dtype=np.float64)
    aim_efficiency = np.zeros(len(sequence), dtype=np.float64)

    current_on_target_ms = 0.0
    current_lock_ms = 0.0
    current_time_since_fire = 10_000.0
    current_time_since_flick = 10_000.0
    last_fire_tick = None
    last_stable_tick = None

    for idx in range(len(sequence)):
        position_history.append(np.array([view_yaw[idx], view_pitch[idx]], dtype=np.float64))
        if len(position_history) > 12:
            position_history.pop(0)
        heading_history.append(float(heading[idx]))
        if len(heading_history) > 14:
            heading_history.pop(0)
        speed_history.append(float(angular_speed[idx]))
        if len(speed_history) > 12:
            speed_history.pop(0)
        heading_change_history.append(float(heading_change[idx]))
        if len(heading_change_history) > 12:
            heading_change_history.pop(0)

        if on_target_soft[idx] > 0.5:
            current_on_target_ms += CSGO_DT * 1000.0
        else:
            current_on_target_ms = 0.0
        on_target_dwell_ms[idx] = current_on_target_ms

        is_lock = float(on_target_tight[idx] > 0.5 and angular_speed[idx] < 0.35 and heading_change[idx] < 0.05)
        lock_indicator[idx] = is_lock
        if is_lock > 0.5:
            current_lock_ms += CSGO_DT * 1000.0
            last_stable_tick = idx
        else:
            current_lock_ms = 0.0
        lock_dwell_ms[idx] = current_lock_ms

        view_straightness[idx] = _local_straightness(position_history)
        direction_entropy_short[idx] = _normalized_entropy(heading_history)
        micro_correction_score[idx] = (
            float(np.std(heading_change_history)) + 0.30 * float(np.std(np.diff(speed_history)))
            if len(speed_history) > 2
            else 0.08
        )
        angular_speed_autocorr_short[idx] = _lag1_autocorr(speed_history)

        if idx > 0:
            yaw_reversal[idx] = float(abs(yaw_delta[idx]) > 0.05 and abs(yaw_delta[idx - 1]) > 0.05 and np.sign(yaw_delta[idx]) != np.sign(yaw_delta[idx - 1]))
            pitch_reversal[idx] = float(abs(pitch_delta[idx]) > 0.05 and abs(pitch_delta[idx - 1]) > 0.05 and np.sign(pitch_delta[idx]) != np.sign(pitch_delta[idx - 1]))

        stability_score[idx] = float(
            np.clip(
                (1.0 - min(target_error[idx] / 6.0, 1.0))
                * (1.0 - min(angular_speed[idx] / 8.0, 1.0))
                * (1.0 - min(micro_correction_score[idx] / 1.0, 1.0))
                * (0.35 + 0.65 * view_straightness[idx]),
                0.0,
                1.0,
            )
        )

        snap_indicator[idx] = float(angular_speed[idx] > 6.5 and target_error_improvement[idx] > 2.5)
        flick_event[idx] = float(angular_speed[idx] > 6.5 and target_error_improvement[idx] > 0.9)
        flick_magnitude[idx] = float(angular_speed[idx] * target_error_improvement[idx])
        aim_efficiency[idx] = float(target_error_improvement[idx] / (angular_speed[idx] + 1e-3))

        if flick_event[idx] > 0.5:
            current_time_since_flick = 0.0
        else:
            current_time_since_flick += CSGO_DT * 1000.0
        time_since_flick_ms[idx] = current_time_since_flick

        if fire_input[idx] > 0.5:
            fire_on_target[idx] = on_target_soft[idx]
            current_time_since_fire = 0.0
            if last_fire_tick is not None:
                last_fire_interval_ms[idx] = (idx - last_fire_tick) * CSGO_DT * 1000.0
            if last_stable_tick is not None:
                last_stabilization_to_fire_ms[idx] = max(0.0, (idx - last_stable_tick) * CSGO_DT * 1000.0)
            fire_motion_coupling[idx] = float(
                np.clip(
                    (0.40 + 0.60 * stability_score[idx])
                    * (1.0 + 0.55 * on_target_soft[idx] + 0.45 * on_target_tight[idx])
                    * (1.0 + min(target_error_improvement[idx], 4.0) / 4.0),
                    0.0,
                    4.0,
                )
            )
            last_fire_tick = idx
        else:
            current_time_since_fire += CSGO_DT * 1000.0
            if idx > 0:
                last_fire_interval_ms[idx] = last_fire_interval_ms[idx - 1]
                last_stabilization_to_fire_ms[idx] = last_stabilization_to_fire_ms[idx - 1]
                fire_motion_coupling[idx] = fire_motion_coupling[idx - 1] * 0.88
        time_since_fire_ms[idx] = current_time_since_fire

    angular_speed_sq = angular_speed**2
    target_error_sq = target_error**2
    error_speed_ratio = target_error / (angular_speed + 0.25)
    error_improvement_ratio = target_error_improvement / (target_error + 1e-3)
    lock_strength = on_target_tight * stability_score
    settling_ratio = stability_score / (angular_speed + 0.15)
    micro_to_speed_ratio = micro_correction_score / (angular_speed + 0.10)
    fire_alignment_strength = fire_input * (1.0 / (1.0 + target_error))
    fire_stability_synergy = fire_input * stability_score * (1.0 / (1.0 + target_error))
    snap_power = snap_indicator * target_error_improvement * angular_speed
    aim_tension = target_error * angular_speed

    centered_yaw = view_yaw - np.median(view_yaw)
    centered_pitch = view_pitch - np.median(view_pitch)
    replay_scale = max(float(np.percentile(np.abs(np.concatenate([centered_yaw, centered_pitch])), 98)), 1.0)
    replay_x = np.clip(centered_yaw / replay_scale, -1.05, 1.05)
    replay_y = np.clip(centered_pitch / replay_scale, -1.05, 1.05)

    label_cheat = int(source_label == "cheater")
    return pd.DataFrame(
        {
            "session_id": session_id,
            "player_id": player_id,
            "player_index": player_index,
            "source_label": source_label,
            "split": split,
            "engagement_index": engagement_index,
            "tick": np.arange(len(sequence)),
            "t": np.arange(len(sequence)) * CSGO_DT - 5.0,
            "view_yaw": view_yaw,
            "view_pitch": view_pitch,
            "replay_x": replay_x,
            "replay_y": replay_y,
            "crosshair_to_victim_yaw": crosshair_yaw,
            "crosshair_to_victim_pitch": crosshair_pitch,
            "yaw_delta": yaw_delta,
            "pitch_delta": pitch_delta,
            "angular_speed": angular_speed,
            "angular_speed_sq": angular_speed_sq,
            "angular_acceleration": angular_acceleration,
            "angular_jerk": angular_jerk,
            "target_error": target_error,
            "target_error_sq": target_error_sq,
            "target_error_delta": target_error_delta,
            "target_error_improvement": target_error_improvement,
            "error_speed_ratio": error_speed_ratio,
            "error_improvement_ratio": error_improvement_ratio,
            "on_target_soft": on_target_soft,
            "on_target_tight": on_target_tight,
            "on_target_dwell_ms": on_target_dwell_ms,
            "lock_indicator": lock_indicator,
            "lock_dwell_ms": lock_dwell_ms,
            "lock_strength": lock_strength,
            "heading_change": heading_change,
            "curvature": curvature,
            "view_straightness": view_straightness,
            "stability_score": stability_score,
            "settling_ratio": settling_ratio,
            "yaw_reversal": yaw_reversal,
            "pitch_reversal": pitch_reversal,
            "direction_entropy_short": direction_entropy_short,
            "micro_correction_score": micro_correction_score,
            "micro_to_speed_ratio": micro_to_speed_ratio,
            "angular_speed_autocorr_short": angular_speed_autocorr_short,
            "fire_input": fire_input,
            "fire_on_target": fire_on_target,
            "time_since_fire_ms": time_since_fire_ms,
            "fire_motion_coupling": fire_motion_coupling,
            "fire_alignment_strength": fire_alignment_strength,
            "fire_stability_synergy": fire_stability_synergy,
            "last_fire_interval_ms": last_fire_interval_ms,
            "last_stabilization_to_fire_ms": last_stabilization_to_fire_ms,
            "flick_event": flick_event,
            "flick_magnitude": flick_magnitude,
            "time_since_flick_ms": time_since_flick_ms,
            "snap_indicator": snap_indicator,
            "snap_power": snap_power,
            "aim_tension": aim_tension,
            "aim_efficiency": aim_efficiency,
            "label_cheat": label_cheat,
        }
    )


def _build_split_player_ids(legit_limit: int, cheat_limit: int, seed: int = 7) -> pd.DataFrame:
    legit_total, cheat_total = load_csgo_archive_arrays()[0].shape[0], load_csgo_archive_arrays()[1].shape[0]
    rng = np.random.default_rng(seed)
    legit_selected = rng.choice(legit_total, size=min(legit_limit, legit_total), replace=False)
    cheat_selected = rng.choice(cheat_total, size=min(cheat_limit, cheat_total), replace=False)
    legit_splits = _player_split_indices(len(legit_selected), seed=seed)
    cheat_splits = _player_split_indices(len(cheat_selected), seed=seed + 17)
    rows = []
    for split_name, indices in legit_splits.items():
        for idx in indices:
            player_index = int(legit_selected[idx])
            rows.append({"source_label": "legit", "player_index": player_index, "player_id": f"L{player_index:05d}", "split": split_name, "label_cheat": 0})
    for split_name, indices in cheat_splits.items():
        for idx in indices:
            player_index = int(cheat_selected[idx])
            rows.append({"source_label": "cheater", "player_index": player_index, "player_id": f"C{player_index:05d}", "split": split_name, "label_cheat": 1})
    return pd.DataFrame(rows)


def _build_engagement_catalog(seed: int = 7, legit_limit: int = CSGO_DEFAULT_LEGIT_PLAYERS, cheat_limit: int = CSGO_DEFAULT_CHEATER_PLAYERS) -> pd.DataFrame:
    legit, cheaters = load_csgo_archive_arrays()
    legit_limit = min(legit_limit, legit.shape[0])
    cheat_limit = min(cheat_limit, cheaters.shape[0])
    players = _build_split_player_ids(legit_limit, cheat_limit, seed=seed)
    rows = []
    for row in players.itertuples(index=False):
        for engagement_index in range(CSGO_ENGAGEMENTS_PER_PLAYER):
            rows.append(
                {
                    "session_id": f"{row.player_id}_E{engagement_index:02d}",
                    "player_id": row.player_id,
                    "player_index": row.player_index,
                    "source_label": row.source_label,
                    "split": row.split,
                    "label_cheat": row.label_cheat,
                    "engagement_index": engagement_index,
                }
            )
    return pd.DataFrame(rows)


def _fetch_sequence(source_arrays: tuple[np.memmap, np.memmap], source_label: str, player_index: int, engagement_index: int) -> np.ndarray:
    legit, cheaters = source_arrays
    source = legit if source_label == "legit" else cheaters
    return np.array(source[player_index, engagement_index], dtype=np.float32)


def _build_encoder_training_set(
    catalog: pd.DataFrame,
    source_arrays: tuple[np.memmap, np.memmap],
    seed: int = 7,
    max_windows_per_class: int = CSGO_MAX_ENCODER_WINDOWS_PER_CLASS,
) -> tuple[StandardScaler, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_catalog = catalog[catalog["split"] == "train"].copy()
    windows_by_class = {0: [], 1: []}
    counts = {0: 0, 1: 0}

    for label in [0, 1]:
        rows = train_catalog[train_catalog["label_cheat"] == label].to_dict(orient="records")
        order = rng.permutation(len(rows))
        for idx in order:
            row = rows[int(idx)]
            raw_sequence = _fetch_sequence(source_arrays, row["source_label"], row["player_index"], row["engagement_index"])
            frame = engineer_csgo_engagement(
                raw_sequence,
                row["session_id"],
                row["player_id"],
                row["source_label"],
                row["split"],
                row["engagement_index"],
                row["player_index"],
            )
            X_windows, meta_windows = _window_session(frame)
            if len(X_windows) == 0:
                continue
            window_priority = (
                meta_windows["fire_input_mean"].to_numpy(dtype=np.float64) * 3.0
                + meta_windows["fire_on_target_mean"].to_numpy(dtype=np.float64) * 2.5
                + meta_windows["on_target_soft_mean"].to_numpy(dtype=np.float64) * 1.5
                + meta_windows["target_error_improvement_max"].to_numpy(dtype=np.float64) * 0.25
                - meta_windows["target_error_mean"].to_numpy(dtype=np.float64) * 0.02
            )
            priority_order = np.argsort(window_priority)[::-1]
            take = min(CSGO_WINDOWS_PER_ENGAGEMENT_FOR_ENCODER, len(X_windows), max_windows_per_class - counts[label])
            if take <= 0:
                break
            top_k = priority_order[: min(max(2 * take, take), len(priority_order))]
            if len(top_k) <= take:
                take_idx = top_k
            else:
                take_idx = rng.choice(top_k, size=take, replace=False)
            windows_by_class[label].append(X_windows[take_idx])
            counts[label] += take
            if counts[label] >= max_windows_per_class:
                break

    X_raw = np.concatenate(
        [
            np.concatenate(windows_by_class[0], axis=0),
            np.concatenate(windows_by_class[1], axis=0),
        ],
        axis=0,
    ).astype(np.float32)
    y = np.concatenate(
        [
            np.zeros(sum(arr.shape[0] for arr in windows_by_class[0]), dtype=np.int64),
            np.ones(sum(arr.shape[0] for arr in windows_by_class[1]), dtype=np.int64),
        ],
        axis=0,
    )
    order = rng.permutation(len(X_raw))
    X_raw = X_raw[order]
    y = y[order]

    scaler = StandardScaler()
    flat = X_raw.reshape(-1, X_raw.shape[-1])
    flat_scaled = scaler.fit_transform(flat).astype(np.float32)
    X_scaled = flat_scaled.reshape(X_raw.shape)
    return scaler, X_scaled, y


def _scale_frame(frame: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    scaled = frame.copy()
    scaled[CSGO_FEATURE_COLUMNS] = scaler.transform(frame[CSGO_FEATURE_COLUMNS].to_numpy(dtype=np.float32))
    return scaled


def _build_window_feature_table(
    catalog: pd.DataFrame,
    source_arrays: tuple[np.memmap, np.memmap],
    scaler: StandardScaler,
    model,
) -> pd.DataFrame:
    rows = []
    for row in catalog.itertuples(index=False):
        raw_sequence = _fetch_sequence(source_arrays, row.source_label, row.player_index, row.engagement_index)
        raw_frame = engineer_csgo_engagement(
            raw_sequence,
            row.session_id,
            row.player_id,
            row.source_label,
            row.split,
            row.engagement_index,
            row.player_index,
        )
        scaled_frame = _scale_frame(raw_frame, scaler)
        X_windows_scaled, _ = _window_session(scaled_frame)
        _, meta_raw = _window_session(raw_frame)
        embeddings = embed_windows(model, X_windows_scaled)
        encoder_probs = predict_window_proba(model, X_windows_scaled)
        if encoder_probs.ndim > 1:
            encoder_probs = encoder_probs[:, -1]
        for idx, meta_row in meta_raw.iterrows():
            record = meta_row.to_dict()
            record["encoder_prob"] = float(encoder_probs[idx])
            for emb_idx, value in enumerate(embeddings[idx]):
                record[f"embedding_{emb_idx:02d}"] = float(value)
            rows.append(record)
    return pd.DataFrame(rows)


def _clean_population_baseline(train_windows: pd.DataFrame) -> dict[str, dict]:
    clean_train = train_windows[train_windows["label_cheat"] == 0].copy()
    emb_cols = [col for col in clean_train.columns if col.startswith("embedding_")]
    excluded = {
        "session_id",
        "player_id",
        "player_index",
        "source_label",
        "split",
        "engagement_index",
        "window_index",
        "start_tick",
        "end_tick",
        "t_end_s",
        "label_cheat",
    }
    summary_cols = [col for col in clean_train.columns if col not in excluded and col not in emb_cols]
    emb = clean_train[emb_cols].to_numpy(dtype=np.float64)
    emb_mean = emb.mean(axis=0)
    emb_dist = np.linalg.norm(emb - emb_mean, axis=1)
    summary = clean_train[summary_cols]
    summary_std = summary.std().fillna(0.0)
    summary_std = pd.Series({column: max(float(summary_std[column]), _std_floor(column)) for column in summary.columns})
    return {
        "embedding_mean": emb_mean,
        "embedding_distance_mean": float(emb_dist.mean()),
        "embedding_distance_std": float(max(float(emb_dist.std()), 0.20)),
        "summary_mean": summary.mean().to_dict(),
        "summary_std": summary_std.to_dict(),
    }


def _build_csgo_classifier_frame(features: pd.DataFrame, baseline: dict[str, dict]) -> pd.DataFrame:
    emb_cols = [col for col in features.columns if col.startswith("embedding_")]
    rows = []
    for row in features.itertuples(index=False):
        row_dict = row._asdict()
        emb = np.array([row_dict[col] for col in emb_cols], dtype=np.float64)
        emb_dist = float(np.linalg.norm(emb - baseline["embedding_mean"]))
        emb_z = max(0.0, (emb_dist - baseline["embedding_distance_mean"]) / baseline["embedding_distance_std"])
        record = {
            "session_id": row.session_id,
            "player_id": row.player_id,
            "player_index": int(row.player_index),
            "source_label": row.source_label,
            "split": row.split,
            "engagement_index": int(row.engagement_index),
            "window_index": int(row.window_index),
            "t_end_s": float(row.t_end_s),
            "label_cheat": int(row.label_cheat),
            "emb_z": emb_z,
            "encoder_signal": _positive_shift(row_dict["encoder_prob"], baseline["summary_mean"]["encoder_prob"], baseline["summary_std"]["encoder_prob"]),
            "target_error_drop": _negative_shift(row_dict["target_error_mean"], baseline["summary_mean"]["target_error_mean"], baseline["summary_std"]["target_error_mean"]),
            "target_error_stability_shift": _negative_shift(row_dict["target_error_std"], baseline["summary_mean"]["target_error_std"], baseline["summary_std"]["target_error_std"]),
            "target_error_min_drop": _negative_shift(row_dict["target_error_min"], baseline["summary_mean"]["target_error_min"], baseline["summary_std"]["target_error_min"]),
            "error_speed_ratio_drop": _negative_shift(row_dict["error_speed_ratio_mean"], baseline["summary_mean"]["error_speed_ratio_mean"], baseline["summary_std"]["error_speed_ratio_mean"]),
            "error_improvement_rise": _positive_shift(row_dict["target_error_improvement_max"], baseline["summary_mean"]["target_error_improvement_max"], baseline["summary_std"]["target_error_improvement_max"]),
            "error_improvement_ratio_rise": _positive_shift(row_dict["error_improvement_ratio_mean"], baseline["summary_mean"]["error_improvement_ratio_mean"], baseline["summary_std"]["error_improvement_ratio_mean"]),
            "on_target_rise": _positive_shift(row_dict["on_target_soft_mean"], baseline["summary_mean"]["on_target_soft_mean"], baseline["summary_std"]["on_target_soft_mean"]),
            "tight_on_target_rise": _positive_shift(row_dict["on_target_tight_mean"], baseline["summary_mean"]["on_target_tight_mean"], baseline["summary_std"]["on_target_tight_mean"]),
            "target_dwell_rise": _positive_shift(row_dict["on_target_dwell_ms_mean"], baseline["summary_mean"]["on_target_dwell_ms_mean"], baseline["summary_std"]["on_target_dwell_ms_mean"]),
            "lock_rise": _positive_shift(row_dict["lock_indicator_mean"], baseline["summary_mean"]["lock_indicator_mean"], baseline["summary_std"]["lock_indicator_mean"]),
            "lock_dwell_rise": _positive_shift(row_dict["lock_dwell_ms_mean"], baseline["summary_mean"]["lock_dwell_ms_mean"], baseline["summary_std"]["lock_dwell_ms_mean"]),
            "lock_strength_rise": _positive_shift(row_dict["lock_strength_mean"], baseline["summary_mean"]["lock_strength_mean"], baseline["summary_std"]["lock_strength_mean"]),
            "stability_rise": _positive_shift(row_dict["stability_score_mean"], baseline["summary_mean"]["stability_score_mean"], baseline["summary_std"]["stability_score_mean"]),
            "straightness_rise": _positive_shift(row_dict["view_straightness_mean"], baseline["summary_mean"]["view_straightness_mean"], baseline["summary_std"]["view_straightness_mean"]),
            "settling_ratio_rise": _positive_shift(row_dict["settling_ratio_mean"], baseline["summary_mean"]["settling_ratio_mean"], baseline["summary_std"]["settling_ratio_mean"]),
            "entropy_drop": _negative_shift(row_dict["direction_entropy_short_mean"], baseline["summary_mean"]["direction_entropy_short_mean"], baseline["summary_std"]["direction_entropy_short_mean"]),
            "micro_correction_drop": _negative_shift(row_dict["micro_correction_score_mean"], baseline["summary_mean"]["micro_correction_score_mean"], baseline["summary_std"]["micro_correction_score_mean"]),
            "micro_to_speed_drop": _negative_shift(row_dict["micro_to_speed_ratio_mean"], baseline["summary_mean"]["micro_to_speed_ratio_mean"], baseline["summary_std"]["micro_to_speed_ratio_mean"]),
            "curvature_drop": _negative_shift(row_dict["curvature_mean"], baseline["summary_mean"]["curvature_mean"], baseline["summary_std"]["curvature_mean"]),
            "angular_regularity_shift": _negative_shift(row_dict["angular_speed_std"], baseline["summary_mean"]["angular_speed_std"], baseline["summary_std"]["angular_speed_std"]),
            "reversal_drop": max(
                _negative_shift(row_dict["yaw_reversal_mean"], baseline["summary_mean"]["yaw_reversal_mean"], baseline["summary_std"]["yaw_reversal_mean"]),
                _negative_shift(row_dict["pitch_reversal_mean"], baseline["summary_mean"]["pitch_reversal_mean"], baseline["summary_std"]["pitch_reversal_mean"]),
            ),
            "fire_rate_shift": _positive_shift(row_dict["fire_input_mean"], baseline["summary_mean"]["fire_input_mean"], baseline["summary_std"]["fire_input_mean"]),
            "fire_on_target_shift": _positive_shift(row_dict["fire_on_target_mean"], baseline["summary_mean"]["fire_on_target_mean"], baseline["summary_std"]["fire_on_target_mean"]),
            "fire_coupling_shift": _positive_shift(row_dict["fire_motion_coupling_max"], baseline["summary_mean"]["fire_motion_coupling_max"], baseline["summary_std"]["fire_motion_coupling_max"]),
            "fire_alignment_strength_rise": _positive_shift(row_dict["fire_alignment_strength_mean"], baseline["summary_mean"]["fire_alignment_strength_mean"], baseline["summary_std"]["fire_alignment_strength_mean"]),
            "fire_stability_synergy_rise": _positive_shift(row_dict["fire_stability_synergy_mean"], baseline["summary_mean"]["fire_stability_synergy_mean"], baseline["summary_std"]["fire_stability_synergy_mean"]),
            "fire_interval_regularization": _negative_shift(row_dict["last_fire_interval_ms_std"], baseline["summary_mean"]["last_fire_interval_ms_std"], baseline["summary_std"]["last_fire_interval_ms_std"]),
            "stabilization_to_fire_drop": _negative_shift(row_dict["last_stabilization_to_fire_ms_mean"], baseline["summary_mean"]["last_stabilization_to_fire_ms_mean"], baseline["summary_std"]["last_stabilization_to_fire_ms_mean"]),
            "flick_rate_shift": _positive_shift(row_dict["flick_event_mean"], baseline["summary_mean"]["flick_event_mean"], baseline["summary_std"]["flick_event_mean"]),
            "flick_magnitude_shift": _positive_shift(row_dict["flick_magnitude_max"], baseline["summary_mean"]["flick_magnitude_max"], baseline["summary_std"]["flick_magnitude_max"]),
            "flick_cooldown_drop": _negative_shift(row_dict["time_since_flick_ms_mean"], baseline["summary_mean"]["time_since_flick_ms_mean"], baseline["summary_std"]["time_since_flick_ms_mean"]),
            "snap_rise": _positive_shift(row_dict["snap_indicator_mean"], baseline["summary_mean"]["snap_indicator_mean"], baseline["summary_std"]["snap_indicator_mean"]),
            "snap_power_rise": _positive_shift(row_dict["snap_power_mean"], baseline["summary_mean"]["snap_power_mean"], baseline["summary_std"]["snap_power_mean"]),
            "aim_tension_drop": _negative_shift(row_dict["aim_tension_mean"], baseline["summary_mean"]["aim_tension_mean"], baseline["summary_std"]["aim_tension_mean"]),
            "aim_efficiency_rise": _positive_shift(row_dict["aim_efficiency_mean"], baseline["summary_mean"]["aim_efficiency_mean"], baseline["summary_std"]["aim_efficiency_mean"]),
        }
        record["automation_signature"] = max(
            record["encoder_signal"],
            record["target_error_drop"],
            record["error_speed_ratio_drop"],
            record["error_improvement_ratio_rise"],
            record["tight_on_target_rise"],
            record["lock_rise"],
            record["lock_strength_rise"],
            record["fire_on_target_shift"],
            record["fire_coupling_shift"],
            record["fire_alignment_strength_rise"],
            record["fire_stability_synergy_rise"],
            record["snap_rise"],
            record["snap_power_rise"],
            record["aim_efficiency_rise"],
        )
        record["movement_signature"] = max(
            record["straightness_rise"],
            record["settling_ratio_rise"],
            record["entropy_drop"],
            record["micro_correction_drop"],
            record["micro_to_speed_drop"],
            record["curvature_drop"],
            record["angular_regularity_shift"],
            record["reversal_drop"],
            record["aim_tension_drop"],
        )
        rows.append(record)
    return pd.DataFrame(rows)


def fit_csgo_cheat_scorer(feature_frame: pd.DataFrame) -> CheatScorer:
    feature_names = [
        "emb_z",
        "encoder_signal",
        "target_error_drop",
        "target_error_stability_shift",
        "target_error_min_drop",
        "error_speed_ratio_drop",
        "error_improvement_rise",
        "error_improvement_ratio_rise",
        "on_target_rise",
        "tight_on_target_rise",
        "target_dwell_rise",
        "lock_rise",
        "lock_dwell_rise",
        "lock_strength_rise",
        "stability_rise",
        "straightness_rise",
        "settling_ratio_rise",
        "entropy_drop",
        "micro_correction_drop",
        "micro_to_speed_drop",
        "curvature_drop",
        "angular_regularity_shift",
        "reversal_drop",
        "fire_rate_shift",
        "fire_on_target_shift",
        "fire_coupling_shift",
        "fire_alignment_strength_rise",
        "fire_stability_synergy_rise",
        "fire_interval_regularization",
        "stabilization_to_fire_drop",
        "flick_rate_shift",
        "flick_magnitude_shift",
        "flick_cooldown_drop",
        "snap_rise",
        "snap_power_rise",
        "aim_tension_drop",
        "aim_efficiency_rise",
        "automation_signature",
        "movement_signature",
    ]
    X = feature_frame[feature_names].to_numpy(dtype=np.float64)
    y = feature_frame["label_cheat"].to_numpy(dtype=np.int64)
    feature_mean = X.mean(axis=0)
    feature_std = X.std(axis=0) + 1e-6
    X_scaled = (X - feature_mean) / feature_std
    classifier = LogisticRegression(max_iter=400, class_weight="balanced", random_state=7)
    classifier.fit(X_scaled, y)
    return CheatScorer(
        feature_names=feature_names,
        feature_mean=feature_mean,
        feature_std=feature_std,
        weights=classifier.coef_[0].astype(np.float64),
        bias=float(classifier.intercept_[0]),
    )


def build_csgo_session_feature_table(window_frame: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        window_frame.sort_values(["session_id", "window_index"])
        .groupby(["session_id", "player_id", "player_index", "source_label", "split", "engagement_index", "label_cheat"], as_index=False)
        .agg(
            {
                "cheat_probability": ["mean", "max", "std"],
                "emb_z": ["mean", "max"],
                "encoder_signal": ["mean", "max"],
                "automation_signature": ["mean", "max"],
                "movement_signature": ["mean", "max"],
                "target_error_drop": ["mean", "max"],
                "error_speed_ratio_drop": ["mean", "max"],
                "error_improvement_ratio_rise": ["mean", "max"],
                "tight_on_target_rise": ["mean", "max"],
                "lock_rise": ["mean", "max"],
                "lock_strength_rise": ["mean", "max"],
                "fire_on_target_shift": ["mean", "max"],
                "fire_coupling_shift": ["mean", "max"],
                "fire_alignment_strength_rise": ["mean", "max"],
                "fire_stability_synergy_rise": ["mean", "max"],
                "snap_rise": ["mean", "max"],
                "snap_power_rise": ["mean", "max"],
                "aim_efficiency_rise": ["mean", "max"],
                "straightness_rise": ["mean", "max"],
                "settling_ratio_rise": ["mean", "max"],
                "entropy_drop": ["mean", "max"],
                "micro_correction_drop": ["mean", "max"],
                "micro_to_speed_drop": ["mean", "max"],
                "curvature_drop": ["mean", "max"],
                "flick_magnitude_shift": ["mean", "max"],
            }
        )
    )
    grouped.columns = ["_".join([str(c) for c in col if c != ""]) if isinstance(col, tuple) else col for col in grouped.columns]
    grouped = grouped.rename(
        columns={
            "session_id_": "session_id",
            "player_id_": "player_id",
            "player_index_": "player_index",
            "source_label_": "source_label",
            "split_": "split",
            "engagement_index_": "engagement_index",
            "label_cheat_": "label_cheat",
        }
    )
    grouped["cheat_probability_std"] = grouped["cheat_probability_std"].fillna(0.0)
    return grouped


def _prepare_session_analysis(window_frame: pd.DataFrame) -> dict:
    ordered = window_frame.sort_values("t_end_s").reset_index(drop=True).copy()
    detector = PageHinkley(delta=0.008, threshold=0.22, alpha=0.99)
    gate_notes = []
    rows = []
    for idx, row in ordered.iterrows():
        raw_score = float(row["cheat_probability"])
        gating_penalty = 1.0
        local_notes = []
        if row["encoder_signal"] < 0.35 and row["automation_signature"] < 0.55:
            gating_penalty *= 0.78
            local_notes.append("low encoder signal")
        if row["target_error_drop"] < 0.18 and row["on_target_rise"] < 0.18 and row["lock_rise"] < 0.18:
            gating_penalty *= 0.82
            local_notes.append("little victim-alignment gain")
        gated_score = raw_score * gating_penalty
        change_detected = int(detector.update(gated_score))
        rows.append(
            {
                "session_id": row["session_id"],
                "player_id": row["player_id"],
                "source_label": row["source_label"],
                "t_end_s": row["t_end_s"],
                "raw_score": raw_score,
                "gated_score": gated_score,
                "change_detected": change_detected,
            }
        )
        if idx == int(np.argmax(ordered["cheat_probability"].to_numpy(dtype=np.float64))):
            gate_notes = local_notes

    score_df = pd.DataFrame(rows)
    peak_idx = int(score_df["gated_score"].idxmax())
    peak_row = ordered.iloc[peak_idx]
    candidate_features = [
        "encoder_signal",
        "target_error_drop",
        "error_speed_ratio_drop",
        "error_improvement_ratio_rise",
        "tight_on_target_rise",
        "lock_rise",
        "lock_strength_rise",
        "fire_on_target_shift",
        "fire_coupling_shift",
        "fire_alignment_strength_rise",
        "fire_stability_synergy_rise",
        "snap_rise",
        "snap_power_rise",
        "aim_efficiency_rise",
        "straightness_rise",
        "settling_ratio_rise",
        "entropy_drop",
        "micro_correction_drop",
        "micro_to_speed_drop",
        "curvature_drop",
        "flick_magnitude_shift",
    ]
    top_deltas = []
    for feature in candidate_features:
        value = float(peak_row[feature])
        top_deltas.append({"feature": feature, "delta": value, "direction": "up" if value >= 0 else "down"})
    top_deltas.sort(key=lambda item: abs(item["delta"]), reverse=True)
    explanations_map = {
        "encoder_signal": "The engagement windows looked more like cheater-labeled patterns than legit windows.",
        "target_error_drop": "Average crosshair-to-victim error dropped sharply relative to the legit baseline.",
        "error_speed_ratio_drop": "The aim achieved lower victim error for the same amount of angular motion than legit windows usually do.",
        "error_improvement_ratio_rise": "Target-error improvement per unit of remaining error rose above the legit baseline.",
        "tight_on_target_rise": "The crosshair stayed tightly aligned with the victim more often than normal.",
        "lock_rise": "Short lock-like aim segments increased relative to legit behavior.",
        "lock_strength_rise": "Lock-like moments became stronger by combining tight alignment with stable aim.",
        "fire_on_target_shift": "Shots were fired while already aligned more often than expected.",
        "fire_coupling_shift": "Firing became unusually synchronized with stabilized aim.",
        "fire_alignment_strength_rise": "Fire inputs happened at stronger victim-alignment levels than the legit baseline.",
        "fire_stability_synergy_rise": "Shots occurred during unusually stable and aligned aim states.",
        "snap_rise": "Rapid error-collapsing snap events appeared more often than normal.",
        "snap_power_rise": "Snap-like events carried more combined speed and error-correction power than legit windows usually show.",
        "aim_efficiency_rise": "The engagement corrected aim error with unusually efficient motion.",
        "straightness_rise": "View-angle motion became more direct and less meandering than the legit baseline.",
        "settling_ratio_rise": "The aim settled into low-motion stable states faster than the legit baseline.",
        "entropy_drop": "Aim direction entropy collapsed relative to the legit baseline.",
        "micro_correction_drop": "Micro-corrections dropped, suggesting cleaner-than-human aim adjustment.",
        "micro_to_speed_drop": "Fine-grained correction fell relative to motion speed, suggesting cleaner-than-human tracking.",
        "curvature_drop": "The aim path became unusually low-curvature during the suspicious segment.",
        "flick_magnitude_shift": "The engagement showed stronger flick-and-settle behavior than legit windows usually do.",
    }
    return {
        "score_df": score_df,
        "top_feature_deltas": top_deltas[:6],
        "base_explanations": [explanations_map[item["feature"]] for item in top_deltas[:4]],
        "gate_reasons": gate_notes,
    }


def _materialize_session_report(prepared: dict, threshold: float) -> dict:
    score_df = prepared["score_df"].copy()
    score_df["running_peak_score"] = score_df["gated_score"].cummax()
    score_df["running_high_count"] = (score_df["gated_score"] >= threshold).astype(int).cumsum()
    score_df["running_change_count"] = score_df["change_detected"].astype(int).cumsum()
    score_df["running_verdict"] = np.where(
        (score_df["running_peak_score"] >= threshold)
        & ((score_df["running_change_count"] > 0) | (score_df["running_high_count"] >= 2)),
        "Suspicious",
        "Likely Legit",
    )
    verdict = str(score_df["running_verdict"].iloc[-1])
    explanations = prepared["base_explanations"] if verdict == "Suspicious" else ["The engagement stayed within the expected range of legit aim behavior."]
    return {
        "verdict": verdict,
        "peak_score": float(score_df["gated_score"].max()),
        "window_scores": score_df,
        "top_feature_deltas": prepared["top_feature_deltas"],
        "explanations": explanations,
        "gate_reasons": prepared["gate_reasons"],
        "high_window_count": int((score_df["gated_score"] >= threshold).sum()),
        "change_detected_count": int(score_df["change_detected"].sum()),
    }


def _session_reports_for_threshold(prepared_map: dict[str, dict], threshold: float) -> dict[str, dict]:
    return {session_id: _materialize_session_report(prepared, threshold) for session_id, prepared in prepared_map.items()}


def _evaluate_sessions(analysis_cache: dict[str, dict], session_catalog: pd.DataFrame) -> dict[str, float]:
    eval_rows = session_catalog[["session_id", "label_cheat"]].drop_duplicates()
    y_true = []
    y_pred = []
    for row in eval_rows.itertuples(index=False):
        y_true.append(int(row.label_cheat))
        y_pred.append(int(analysis_cache[row.session_id]["verdict"] == "Suspicious"))
    y_true_arr = np.array(y_true, dtype=int)
    y_pred_arr = np.array(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).ravel()
    prevalence = float(y_true_arr.mean())
    sensitivity = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    specificity = float(tn / max(1, fp + tn))
    balanced_accuracy = float((sensitivity + specificity) / 2.0)
    majority_baseline_accuracy = float(max(prevalence, 1.0 - prevalence))
    ppv = float(tp / max(1, tp + fp))
    npv = float(tn / max(1, tn + fn))
    lr_positive = float(sensitivity / max(1e-9, 1.0 - specificity)) if specificity < 1.0 else float("inf")
    lr_negative = float((1.0 - sensitivity) / max(specificity, 1e-9))
    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "balanced_accuracy": balanced_accuracy,
        "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "recall": sensitivity,
        "specificity": specificity,
        "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true_arr, y_pred_arr)) if len(np.unique(y_pred_arr)) > 1 else 0.0,
        "false_positive_rate": float(fp / max(1, fp + tn)),
        "false_negative_rate": float(fn / max(1, fn + tp)),
        "prevalence": prevalence,
        "predicted_positive_rate": float(y_pred_arr.mean()),
        "majority_baseline_accuracy": majority_baseline_accuracy,
        "ppv_at_observed_prevalence": ppv,
        "npv_at_observed_prevalence": npv,
        "positive_likelihood_ratio": lr_positive,
        "negative_likelihood_ratio": lr_negative,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def _bayes_posterior_table(metrics: dict[str, float]) -> pd.DataFrame:
    sensitivity = float(metrics["recall"])
    specificity = float(metrics["specificity"])
    observed_prevalence = float(metrics["prevalence"])
    rows = []
    for label, prevalence in BAYES_PREVALENCE_SCENARIOS:
        prior = observed_prevalence if prevalence is None else float(prevalence)
        positive_den = sensitivity * prior + (1.0 - specificity) * (1.0 - prior)
        negative_den = specificity * (1.0 - prior) + (1.0 - sensitivity) * prior
        ppv = (sensitivity * prior / positive_den) if positive_den > 0 else 0.0
        npv = (specificity * (1.0 - prior) / negative_den) if negative_den > 0 else 0.0
        posterior_cheat_given_negative = ((1.0 - sensitivity) * prior / negative_den) if negative_den > 0 else 0.0
        rows.append(
            {
                "scenario": label,
                "assumed_prevalence": prior,
                "posterior_cheat_given_positive": ppv,
                "posterior_legit_given_negative": npv,
                "posterior_cheat_given_negative": posterior_cheat_given_negative,
            }
        )
    return pd.DataFrame(rows)


def _choose_threshold(validation_prepared: dict[str, dict], validation_sessions: pd.DataFrame) -> float:
    best_threshold = 0.55
    best_objective = float("-inf")
    for threshold in np.linspace(0.35, 0.90, 12):
        reports = _session_reports_for_threshold(validation_prepared, float(threshold))
        metrics = _evaluate_sessions(reports, validation_sessions)
        objective = (
            metrics["balanced_accuracy"]
            + 0.20 * metrics["precision"]
            + 0.10 * metrics["mcc"]
            - 0.20 * metrics["false_positive_rate"]
        )
        if objective > best_objective:
            best_objective = objective
            best_threshold = float(threshold)
    return best_threshold


def analyze_csgo_engagement(frame: pd.DataFrame, baseline: dict[str, dict], cheat_scorer: CheatScorer, decision_threshold: float) -> dict:
    emb_cols = [col for col in frame.columns if col.startswith("embedding_")]
    classifier_frame = frame.copy()
    if emb_cols:
        classifier_frame = _build_csgo_classifier_frame(classifier_frame, baseline)
    classifier_frame["cheat_probability"] = cheat_scorer.predict_proba(classifier_frame)
    prepared = _prepare_session_analysis(classifier_frame)
    return _materialize_session_report(prepared, decision_threshold)


def build_csgo_pipeline(
    seed: int = 7,
    legit_limit: int = CSGO_DEFAULT_LEGIT_PLAYERS,
    cheat_limit: int = CSGO_DEFAULT_CHEATER_PLAYERS,
) -> dict:
    source_arrays = load_csgo_archive_arrays()
    catalog = _build_engagement_catalog(seed=seed, legit_limit=legit_limit, cheat_limit=cheat_limit)

    scaler, X_encoder, y_encoder = _build_encoder_training_set(catalog, source_arrays, seed=seed)
    model, training_metrics = train_fingerprint_model(X_encoder, y_encoder, num_players=2, epochs=35, batch_size=128, lr=1e-3)

    window_features = _build_window_feature_table(catalog, source_arrays, scaler, model)
    train_windows = window_features[window_features["split"] == "train"].copy()
    validation_windows = window_features[window_features["split"] == "validation"].copy()
    test_windows = window_features[window_features["split"] == "test"].copy()
    baseline = _clean_population_baseline(train_windows)

    classifier_train = _build_csgo_classifier_frame(train_windows, baseline)
    classifier_validation = _build_csgo_classifier_frame(validation_windows, baseline)
    classifier_test = _build_csgo_classifier_frame(test_windows, baseline)

    cheat_scorer = fit_csgo_cheat_scorer(classifier_train)
    for frame in (classifier_train, classifier_validation, classifier_test):
        frame["cheat_probability"] = cheat_scorer.predict_proba(frame)

    session_features_train = build_csgo_session_feature_table(classifier_train)
    session_features_validation = build_csgo_session_feature_table(classifier_validation)
    session_features_test = build_csgo_session_feature_table(classifier_test)
    session_feature_table = pd.concat([session_features_train, session_features_validation, session_features_test], ignore_index=True)

    validation_prepared = {session_id: _prepare_session_analysis(group) for session_id, group in classifier_validation.groupby("session_id")}
    train_prepared = {session_id: _prepare_session_analysis(group) for session_id, group in classifier_train.groupby("session_id")}
    test_prepared = {session_id: _prepare_session_analysis(group) for session_id, group in classifier_test.groupby("session_id")}
    threshold = _choose_threshold(
        validation_prepared,
        catalog[catalog["split"] == "validation"][["session_id", "label_cheat"]].drop_duplicates(),
    )

    analysis_train = _session_reports_for_threshold(train_prepared, threshold)
    analysis_validation = _session_reports_for_threshold(validation_prepared, threshold)
    analysis_test = _session_reports_for_threshold(test_prepared, threshold)
    analysis_cache = {}
    analysis_cache.update(analysis_train)
    analysis_cache.update(analysis_validation)
    analysis_cache.update(analysis_test)

    split_metrics = {
        "train": _evaluate_sessions(analysis_train, catalog[catalog["split"] == "train"][["session_id", "label_cheat"]].drop_duplicates()),
        "validation": _evaluate_sessions(analysis_validation, catalog[catalog["split"] == "validation"][["session_id", "label_cheat"]].drop_duplicates()),
        "test": _evaluate_sessions(analysis_test, catalog[catalog["split"] == "test"][["session_id", "label_cheat"]].drop_duplicates()),
    }
    bayes_reference = _bayes_posterior_table(split_metrics["test"])

    test_manifest = catalog[catalog["split"] == "test"][["session_id", "player_id", "player_index", "source_label", "split", "engagement_index", "label_cheat"]].drop_duplicates()
    test_manifest = test_manifest.copy()
    test_manifest["peak_score"] = test_manifest["session_id"].map(lambda sid: analysis_test[sid]["peak_score"])
    test_manifest["verdict"] = test_manifest["session_id"].map(lambda sid: analysis_test[sid]["verdict"])

    cheat_samples = test_manifest[test_manifest["label_cheat"] == 1].sort_values("peak_score", ascending=False).head(CSGO_SAMPLE_SESSION_COUNT // 2)
    legit_samples = test_manifest[test_manifest["label_cheat"] == 0].sort_values("peak_score", ascending=False).head(CSGO_SAMPLE_SESSION_COUNT // 2)
    sample_manifest = pd.concat([cheat_samples, legit_samples], ignore_index=True)

    sample_sessions = {}
    sample_reports = {}
    for row in sample_manifest.itertuples(index=False):
        raw_sequence = _fetch_sequence(source_arrays, row.source_label, row.player_index, row.engagement_index)
        frame = engineer_csgo_engagement(
            raw_sequence,
            row.session_id,
            row.player_id,
            row.source_label,
            row.split,
            row.engagement_index,
            row.player_index,
        )
        sample_sessions[row.session_id] = frame
        sample_reports[row.session_id] = analysis_test[row.session_id]

    archive_summary = {
        "legit_players_in_archive": int(source_arrays[0].shape[0]),
        "cheater_players_in_archive": int(source_arrays[1].shape[0]),
        "engagements_per_player": CSGO_ENGAGEMENTS_PER_PLAYER,
        "timesteps_per_engagement": CSGO_ENGAGEMENT_LENGTH,
        "raw_feature_count": len(RAW_FEATURE_NAMES),
        "window_size": CSGO_WINDOW_SIZE,
        "window_stride": CSGO_WINDOW_STRIDE,
        "legit_players_used": int(legit_limit),
        "cheater_players_used": int(cheat_limit),
    }

    return {
        "model": model,
        "training_metrics": training_metrics,
        "scaler": scaler,
        "catalog": catalog,
        "window_features": window_features,
        "baseline": baseline,
        "cheat_scorer": cheat_scorer,
        "classifier_windows": pd.concat([classifier_train, classifier_validation, classifier_test], ignore_index=True),
        "session_feature_table": session_feature_table,
        "decision_threshold": threshold,
        "split_metrics": split_metrics,
        "evaluation_metrics": split_metrics["test"],
        "bayes_reference": bayes_reference,
        "sample_manifest": sample_manifest,
        "sample_sessions": sample_sessions,
        "sample_reports": sample_reports,
        "archive_summary": archive_summary,
    }


def export_csgo_bundle(seed: int = 7, legit_limit: int = CSGO_DEFAULT_LEGIT_PLAYERS, cheat_limit: int = CSGO_DEFAULT_CHEATER_PLAYERS) -> dict:
    artifacts = build_csgo_pipeline(seed=seed, legit_limit=legit_limit, cheat_limit=cheat_limit)
    session_dir = CSGO_DATA_DIR / "sessions"
    score_dir = CSGO_DATA_DIR / "scores"
    CSGO_DATA_DIR.mkdir(parents=True, exist_ok=True)
    session_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)

    sample_manifest_rows = []
    reports_payload = {}
    for session_id, frame in artifacts["sample_sessions"].items():
        report = artifacts["sample_reports"][session_id]
        frame.to_csv(session_dir / f"{session_id}.csv", index=False)
        report["window_scores"].to_csv(score_dir / f"{session_id}.csv", index=False)
        reports_payload[session_id] = {
            "verdict": report["verdict"],
            "peak_score": report["peak_score"],
            "top_feature_deltas": report["top_feature_deltas"],
            "explanations": report["explanations"],
            "gate_reasons": report["gate_reasons"],
            "high_window_count": report["high_window_count"],
            "change_detected_count": report["change_detected_count"],
        }
        sample_manifest_rows.append(
            {
                "session_id": session_id,
                "player_id": frame["player_id"].iloc[0],
                "player_index": int(frame["player_index"].iloc[0]),
                "source_label": frame["source_label"].iloc[0],
                "split": frame["split"].iloc[0],
                "engagement_index": int(frame["engagement_index"].iloc[0]),
                "verdict": report["verdict"],
                "peak_score": report["peak_score"],
                "label_cheat": int(frame["label_cheat"].iloc[0]),
            }
        )

    pd.DataFrame([artifacts["evaluation_metrics"]]).to_csv(CSGO_DATA_DIR / "evaluation_metrics.csv", index=False)
    pd.DataFrame(artifacts["split_metrics"]).T.to_csv(CSGO_DATA_DIR / "split_metrics.csv", index_label="split")
    artifacts["bayes_reference"].to_csv(CSGO_DATA_DIR / "bayes_reference.csv", index=False)
    artifacts["window_features"].to_csv(CSGO_DATA_DIR / "window_features.csv", index=False)
    artifacts["classifier_windows"].to_csv(CSGO_DATA_DIR / "classifier_windows.csv", index=False)
    artifacts["session_feature_table"].to_csv(CSGO_DATA_DIR / "session_feature_table.csv", index=False)
    pd.DataFrame(sample_manifest_rows).to_csv(CSGO_DATA_DIR / "sample_manifest.csv", index=False)
    with (CSGO_DATA_DIR / "sample_reports.json").open("w") as f:
        json.dump(reports_payload, f, indent=2)
    with (CSGO_DATA_DIR / "metadata.json").open("w") as f:
        json.dump(
            {
                "decision_threshold": artifacts["decision_threshold"],
                "archive_summary": artifacts["archive_summary"],
                "raw_feature_names": RAW_FEATURE_NAMES,
                "engineered_feature_columns": CSGO_FEATURE_COLUMNS,
                "model_summary": {
                    "window_encoder": "sklearn MLPClassifier over flattened causal windows",
                    "cheat_scorer": "sklearn LogisticRegression over engineered window shifts",
                    "train_once_export_pattern": True,
                },
            },
            f,
            indent=2,
        )
    return artifacts


def load_exported_csgo_bundle() -> dict | None:
    if not CSGO_DATA_DIR.exists():
        return None
    metrics_path = CSGO_DATA_DIR / "evaluation_metrics.csv"
    split_metrics_path = CSGO_DATA_DIR / "split_metrics.csv"
    bayes_path = CSGO_DATA_DIR / "bayes_reference.csv"
    manifest_path = CSGO_DATA_DIR / "sample_manifest.csv"
    reports_path = CSGO_DATA_DIR / "sample_reports.json"
    metadata_path = CSGO_DATA_DIR / "metadata.json"
    window_features_path = CSGO_DATA_DIR / "window_features.csv"
    classifier_windows_path = CSGO_DATA_DIR / "classifier_windows.csv"
    session_feature_path = CSGO_DATA_DIR / "session_feature_table.csv"
    if not all(path.exists() for path in [metrics_path, split_metrics_path, bayes_path, manifest_path, reports_path, metadata_path, window_features_path, classifier_windows_path, session_feature_path]):
        return None
    evaluation_metrics = pd.read_csv(metrics_path).iloc[0].to_dict()
    split_metrics = pd.read_csv(split_metrics_path).set_index("split").to_dict(orient="index")
    bayes_reference = pd.read_csv(bayes_path)
    manifest = pd.read_csv(manifest_path)
    with reports_path.open() as f:
        reports = json.load(f)
    with metadata_path.open() as f:
        metadata = json.load(f)

    sample_sessions = {}
    sample_reports = {}
    for row in manifest.itertuples(index=False):
        session_id = row.session_id
        session_df = pd.read_csv(CSGO_DATA_DIR / "sessions" / f"{session_id}.csv")
        score_df = pd.read_csv(CSGO_DATA_DIR / "scores" / f"{session_id}.csv")
        sample_sessions[session_id] = session_df
        report = reports[session_id]
        report["window_scores"] = score_df
        sample_reports[session_id] = report
    return {
        "evaluation_metrics": evaluation_metrics,
        "split_metrics": split_metrics,
        "bayes_reference": bayes_reference,
        "sample_manifest": manifest,
        "sample_sessions": sample_sessions,
        "sample_reports": sample_reports,
        "decision_threshold": float(metadata["decision_threshold"]),
        "archive_summary": metadata["archive_summary"],
        "raw_feature_names": metadata["raw_feature_names"],
        "engineered_feature_columns": metadata["engineered_feature_columns"],
        "model_summary": metadata.get("model_summary", {}),
        "window_features": pd.read_csv(window_features_path),
        "classifier_windows": pd.read_csv(classifier_windows_path),
        "session_feature_table": pd.read_csv(session_feature_path),
    }
