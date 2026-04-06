from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _std_floor(column: str) -> float:
    if column.startswith("speed"):
        return 0.02
    if column.startswith("acceleration") or column.startswith("jerk"):
        return 0.015
    if column.startswith("heading_change"):
        return 0.05
    if column.startswith("angular_velocity"):
        return 0.45
    if column.startswith("curvature"):
        return 0.18
    if column.startswith("pause_ms"):
        return 35.0
    if column.startswith("burst_duration_ms"):
        return 45.0
    if column.startswith("local_straightness"):
        return 0.04
    if column.startswith("direction_entropy_short"):
        return 0.06
    if column.startswith("roughness_score"):
        return 0.04
    if column.startswith("speed_autocorr_short"):
        return 0.05
    if column.startswith("click_motion_coupling"):
        return 0.18
    if column.startswith("last_click_interval_ms"):
        return 28.0
    if column.startswith("last_stabilization_delay_ms"):
        return 22.0
    if column.startswith("ping_ms"):
        return 5.0
    if column.startswith("jitter_ms"):
        return 1.5
    if column.startswith("packet_loss_pct"):
        return 0.12
    if column.startswith("command_age_ms"):
        return 4.0
    if column.startswith("packet_interarrival_ms"):
        return 1.5
    if column.startswith("input_burstiness"):
        return 0.05
    if column.startswith("server_correction_magnitude"):
        return 0.08
    if column.startswith("tick_desync_ms"):
        return 0.9
    if column.startswith("sensitivity"):
        return 0.05
    return 0.03


@dataclass
class PageHinkley:
    delta: float = 0.01
    threshold: float = 0.35
    alpha: float = 0.99
    mean: float = 0.0
    cumulative: float = 0.0
    minimum: float = 0.0
    seen: int = 0

    def update(self, value: float) -> bool:
        self.seen += 1
        self.mean = self.alpha * self.mean + (1 - self.alpha) * value if self.seen > 1 else value
        self.cumulative += value - self.mean - self.delta
        self.minimum = min(self.minimum, self.cumulative)
        return (self.cumulative - self.minimum) > self.threshold


@dataclass
class CheatScorer:
    feature_names: list[str]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    weights: np.ndarray
    bias: float

    def predict_proba(self, feature_frame: pd.DataFrame) -> np.ndarray:
        X = feature_frame[self.feature_names].to_numpy(dtype=np.float64)
        X_scaled = (X - self.feature_mean) / self.feature_std
        logits = X_scaled @ self.weights + self.bias
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))


def build_player_baselines(embeddings: np.ndarray, meta: pd.DataFrame) -> dict[str, dict]:
    excluded = {
        "session_id",
        "player_id",
        "mode",
        "start_tick",
        "end_tick",
        "t_end_s",
        "label_cheat",
        "window_cheat_fraction",
        "window_confounder_fraction",
    }
    summary_cols = [col for col in meta.columns if col not in excluded]
    baselines = {}
    for player_id, player_meta in meta.groupby("player_id"):
        idx = player_meta.index.to_numpy()
        emb = embeddings[idx]
        emb_mean = emb.mean(axis=0)
        emb_distances = np.linalg.norm(emb - emb_mean, axis=1)
        summary = player_meta[summary_cols]
        summary_std = summary.std().fillna(0.0)
        summary_std = pd.Series({column: max(float(summary_std[column]), _std_floor(column)) for column in summary.columns})
        baselines[player_id] = {
            "embedding_mean": emb_mean,
            "embedding_std": emb.std(axis=0) + 1e-6,
            "embedding_distance_mean": float(emb_distances.mean()),
            "embedding_distance_std": float(max(float(emb_distances.std()), 0.18)),
            "summary_mean": summary.mean().to_dict(),
            "summary_std": summary_std.to_dict(),
        }
    return baselines


def _positive_shift(value: float, mean: float, std: float) -> float:
    return max(0.0, (value - mean) / max(std, 1e-6))


def _negative_shift(value: float, mean: float, std: float) -> float:
    return max(0.0, (mean - value) / max(std, 1e-6))


def _absolute_shift(value: float, mean: float, std: float) -> float:
    return abs(value - mean) / max(std, 1e-6)


def build_window_feature_frame(
    embeddings: np.ndarray,
    meta: pd.DataFrame,
    baselines: dict[str, dict],
) -> pd.DataFrame:
    rows = []
    for idx, row in meta.iterrows():
        player_id = row["player_id"]
        baseline = baselines[player_id]
        emb = embeddings[idx]
        emb_dist = float(np.linalg.norm(emb - baseline["embedding_mean"]))
        emb_z = max(0.0, (emb_dist - baseline["embedding_distance_mean"]) / baseline["embedding_distance_std"])

        feature_row = {
            "emb_z": emb_z,
            "straightness_rise": _positive_shift(
                row["local_straightness_mean"],
                baseline["summary_mean"]["local_straightness_mean"],
                baseline["summary_std"]["local_straightness_mean"],
            ),
            "entropy_drop": _negative_shift(
                row["direction_entropy_short_mean"],
                baseline["summary_mean"]["direction_entropy_short_mean"],
                baseline["summary_std"]["direction_entropy_short_mean"],
            ),
            "curvature_drop": _negative_shift(
                row["curvature_mean"],
                baseline["summary_mean"]["curvature_mean"],
                baseline["summary_std"]["curvature_mean"],
            ),
            "curvature_regularity_shift": _negative_shift(
                row["curvature_std"],
                baseline["summary_mean"]["curvature_std"],
                baseline["summary_std"]["curvature_std"],
            ),
            "roughness_drop": _negative_shift(
                row["roughness_score_mean"],
                baseline["summary_mean"]["roughness_score_mean"],
                baseline["summary_std"]["roughness_score_mean"],
            ),
            "heading_regularity_shift": _negative_shift(
                row["heading_change_std"],
                baseline["summary_mean"]["heading_change_std"],
                baseline["summary_std"]["heading_change_std"],
            ),
            "jerk_regularity_shift": _negative_shift(
                row["jerk_std"],
                baseline["summary_mean"]["jerk_std"],
                baseline["summary_std"]["jerk_std"],
            ),
            "speed_regularity_shift": _negative_shift(
                row["speed_std"],
                baseline["summary_mean"]["speed_std"],
                baseline["summary_std"]["speed_std"],
            ),
            "burst_consistency_shift": _negative_shift(
                row["burst_duration_ms_std"],
                baseline["summary_mean"]["burst_duration_ms_std"],
                baseline["summary_std"]["burst_duration_ms_std"],
            ),
            "autocorr_rise": _positive_shift(
                row["speed_autocorr_short_mean"],
                baseline["summary_mean"]["speed_autocorr_short_mean"],
                baseline["summary_std"]["speed_autocorr_short_mean"],
            ),
            "click_rate_shift": _positive_shift(
                row["click_mean"],
                baseline["summary_mean"]["click_mean"],
                baseline["summary_std"]["click_mean"],
            ),
            "click_coupling_shift": _positive_shift(
                row["click_motion_coupling_max"],
                baseline["summary_mean"]["click_motion_coupling_max"],
                baseline["summary_std"]["click_motion_coupling_max"],
            ),
            "click_interval_regularization": _negative_shift(
                row["last_click_interval_ms_std"],
                baseline["summary_mean"]["last_click_interval_ms_std"],
                baseline["summary_std"]["last_click_interval_ms_std"],
            ),
            "stabilization_delay_drop": _negative_shift(
                row["last_stabilization_delay_ms_mean"],
                baseline["summary_mean"]["last_stabilization_delay_ms_mean"],
                baseline["summary_std"]["last_stabilization_delay_ms_mean"],
            ),
            "stabilization_delay_regularity": _negative_shift(
                row["last_stabilization_delay_ms_std"],
                baseline["summary_mean"]["last_stabilization_delay_ms_std"],
                baseline["summary_std"]["last_stabilization_delay_ms_std"],
            ),
            "motion_activity_abs_shift": _absolute_shift(
                row["motion_active_mean"],
                baseline["summary_mean"]["motion_active_mean"],
                baseline["summary_std"]["motion_active_mean"],
            ),
            "pause_shift": _absolute_shift(
                row["pause_ms_mean"],
                baseline["summary_mean"]["pause_ms_mean"],
                baseline["summary_std"]["pause_ms_mean"],
            ),
            "ping_change": _positive_shift(
                row["ping_ms_mean"],
                baseline["summary_mean"]["ping_ms_mean"],
                baseline["summary_std"]["ping_ms_mean"],
            ),
            "jitter_change": _positive_shift(
                row["jitter_ms_mean"],
                baseline["summary_mean"]["jitter_ms_mean"],
                baseline["summary_std"]["jitter_ms_mean"],
            ),
            "packet_loss_change": _positive_shift(
                row["packet_loss_pct_mean"],
                baseline["summary_mean"]["packet_loss_pct_mean"],
                baseline["summary_std"]["packet_loss_pct_mean"],
            ),
            "command_age_change": _positive_shift(
                row["command_age_ms_mean"],
                baseline["summary_mean"]["command_age_ms_mean"],
                baseline["summary_std"]["command_age_ms_mean"],
            ),
            "interarrival_change": _positive_shift(
                row["packet_interarrival_ms_mean"],
                baseline["summary_mean"]["packet_interarrival_ms_mean"],
                baseline["summary_std"]["packet_interarrival_ms_mean"],
            ),
            "desync_change": _positive_shift(
                row["tick_desync_ms_mean"],
                baseline["summary_mean"]["tick_desync_ms_mean"],
                baseline["summary_std"]["tick_desync_ms_mean"],
            ),
            "server_correction_change": _positive_shift(
                row["server_correction_magnitude_mean"],
                baseline["summary_mean"]["server_correction_magnitude_mean"],
                baseline["summary_std"]["server_correction_magnitude_mean"],
            ),
            "sensitivity_delta": abs(
                row["sensitivity_mean"] / max(baseline["summary_mean"]["sensitivity_mean"], 1e-6) - 1.0
            ),
            "window_cheat_fraction": float(row.get("window_cheat_fraction", 0.0)),
            "window_confounder_fraction": float(row.get("window_confounder_fraction", 0.0)),
            "t_end_s": float(row["t_end_s"]),
            "session_id": row["session_id"],
            "player_id": player_id,
            "mode": row["mode"],
        }
        feature_row["automation_signature"] = max(
            feature_row["straightness_rise"],
            feature_row["entropy_drop"],
            feature_row["curvature_drop"],
            feature_row["click_coupling_shift"],
            feature_row["stabilization_delay_drop"],
            feature_row["click_interval_regularization"],
        )
        feature_row["movement_signature"] = max(
            feature_row["speed_regularity_shift"],
            feature_row["jerk_regularity_shift"],
            feature_row["heading_regularity_shift"],
            feature_row["burst_consistency_shift"],
            feature_row["autocorr_rise"],
        )
        feature_row["network_stress"] = max(
            feature_row["ping_change"],
            feature_row["jitter_change"],
            feature_row["packet_loss_change"],
            feature_row["command_age_change"],
            feature_row["desync_change"],
            feature_row["server_correction_change"],
        )
        rows.append(feature_row)

    return pd.DataFrame(rows)


def fit_cheat_scorer(feature_frame: pd.DataFrame) -> CheatScorer:
    feature_names = [
        "emb_z",
        "straightness_rise",
        "entropy_drop",
        "curvature_drop",
        "curvature_regularity_shift",
        "roughness_drop",
        "heading_regularity_shift",
        "jerk_regularity_shift",
        "speed_regularity_shift",
        "burst_consistency_shift",
        "autocorr_rise",
        "click_rate_shift",
        "click_coupling_shift",
        "click_interval_regularization",
        "stabilization_delay_drop",
        "stabilization_delay_regularity",
        "motion_activity_abs_shift",
        "pause_shift",
        "ping_change",
        "jitter_change",
        "packet_loss_change",
        "command_age_change",
        "interarrival_change",
        "desync_change",
        "server_correction_change",
        "sensitivity_delta",
        "automation_signature",
        "movement_signature",
        "network_stress",
    ]
    X = feature_frame[feature_names]
    y = (feature_frame["window_cheat_fraction"] >= 0.5).astype(int)
    X_np = X.to_numpy(dtype=np.float64)
    y_np = y.to_numpy(dtype=np.float64)
    feature_mean = X_np.mean(axis=0)
    feature_std = X_np.std(axis=0) + 1e-6
    X_scaled = (X_np - feature_mean) / feature_std

    weights = np.zeros(X_scaled.shape[1], dtype=np.float64)
    bias = 0.0
    positive_weight = max(1.0, float((len(y_np) - y_np.sum()) / max(1.0, y_np.sum())))
    sample_weight = np.where(y_np > 0.5, positive_weight, 1.0)
    lr = 0.045
    l2 = 0.002

    for _ in range(1200):
        logits = X_scaled @ weights + bias
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
        error = (probs - y_np) * sample_weight
        grad_w = (X_scaled.T @ error) / len(X_scaled) + l2 * weights
        grad_b = float(np.mean(error))
        weights -= lr * grad_w
        bias -= lr * grad_b

    return CheatScorer(
        feature_names=feature_names,
        feature_mean=feature_mean,
        feature_std=feature_std,
        weights=weights,
        bias=bias,
    )


def _top_feature_deltas(summary: dict[str, float], baseline: dict[str, float], baseline_std: dict[str, float]) -> list[dict[str, float | str]]:
    ranked = []
    for feature, value in summary.items():
        if feature not in baseline:
            continue
        std = max(float(baseline_std.get(feature, 1.0)), 1e-6)
        z = (float(value) - float(baseline[feature])) / std
        ranked.append((feature, z))
    ranked.sort(key=lambda item: abs(item[1]), reverse=True)
    top = []
    for feature, z in ranked[:6]:
        top.append(
            {
                "feature": feature,
                "delta": float(z),
                "direction": "up" if z >= 0 else "down",
            }
        )
    return top


def analyze_session_windows(
    embeddings: np.ndarray,
    meta: pd.DataFrame,
    baselines: dict[str, dict],
    cheat_scorer: CheatScorer | None = None,
    decision_threshold: float = 0.72,
) -> dict:
    player_id = meta["player_id"].iloc[0]
    mode = meta["mode"].iloc[0]
    baseline = baselines[player_id]
    detector = PageHinkley()
    rows = []
    gate_reasons = []
    feature_frame = build_window_feature_frame(embeddings, meta, baselines)
    learned_probs = cheat_scorer.predict_proba(feature_frame) if cheat_scorer is not None else None
    running_peak_score = 0.0
    running_high_window_count = 0
    running_change_detected_count = 0

    for idx, row in meta.iterrows():
        emb = embeddings[idx]
        emb_dist = float(np.linalg.norm(emb - baseline["embedding_mean"]))
        feat_row = feature_frame.loc[idx]

        heuristic_prob = 1.0 / (
            1.0
            + np.exp(
                -(
                    -2.5
                    + 0.45 * feat_row["emb_z"]
                    + 0.90 * feat_row["straightness_rise"]
                    + 1.15 * feat_row["entropy_drop"]
                    + 0.85 * feat_row["curvature_drop"]
                    + 0.70 * feat_row["roughness_drop"]
                    + 0.65 * feat_row["click_coupling_shift"]
                    + 0.55 * feat_row["click_interval_regularization"]
                    + 0.55 * feat_row["stabilization_delay_drop"]
                    + 0.45 * feat_row["autocorr_rise"]
                )
            )
        )
        learned_prob = float(learned_probs[idx]) if learned_probs is not None else float(heuristic_prob)
        raw_score = float(0.75 * learned_prob + 0.25 * heuristic_prob)

        automation_signature = feat_row["automation_signature"] > 2.0
        network_stress = feat_row["network_stress"] > 1.8

        gate_factor = 1.0
        if network_stress and not automation_signature:
            gate_factor *= 0.10
            gate_reasons.append(
                "Network stress rose sharply through ping, jitter, packet loss, or command age, so the gate discounted the anomaly unless it also showed an automation-like motor signature."
            )
        if feat_row["sensitivity_delta"] > 0.18 and feat_row["automation_signature"] < 1.9:
            gate_factor *= 0.18
            gate_reasons.append(
                "A large sensitivity change can explain broad cursor-dynamics drift, so the gate reduced the score when no strong locking or click-timing signature appeared."
            )
        if mode == "patch_shift" and feat_row["automation_signature"] < 1.8:
            gate_factor *= 0.22
            gate_reasons.append(
                "Patch or environment shift present, so broad movement drift was treated as less suspicious unless the cursor behavior also became abnormally low-entropy."
            )
        if feat_row["network_stress"] > 2.4 and feat_row["automation_signature"] < 1.5:
            gate_factor *= 0.45

        gated_score = raw_score * gate_factor
        change_detected = int(detector.update(gated_score))
        running_peak_score = max(running_peak_score, gated_score)
        if gated_score >= max(0.58, decision_threshold - 0.08):
            running_high_window_count += 1
        if change_detected:
            running_change_detected_count += 1
        running_verdict = (
            "Suspicious"
            if running_peak_score >= decision_threshold and (running_change_detected_count > 0 or running_high_window_count >= 3)
            else "Likely Legit"
        )
        rows.append(
            {
                "session_id": row["session_id"],
                "player_id": player_id,
                "mode": mode,
                "t_end_s": row["t_end_s"],
                "embedding_distance": emb_dist,
                "embedding_z": feat_row["emb_z"],
                "cheat_probability": learned_prob,
                "raw_score": raw_score,
                "gated_score": gated_score,
                "change_detected": change_detected,
                "running_peak_score": running_peak_score,
                "running_high_window_count": running_high_window_count,
                "running_change_detected_count": running_change_detected_count,
                "running_verdict": running_verdict,
            }
        )

    score_df = pd.DataFrame(rows)
    peak_idx = score_df["gated_score"].idxmax()
    peak_summary = meta.loc[peak_idx]
    top_shifts = _top_feature_deltas(
        peak_summary.to_dict(),
        baseline["summary_mean"],
        baseline["summary_std"],
    )

    final_row = score_df.iloc[-1]
    high_window_count = int(final_row["running_high_window_count"])
    change_detected_count = int(final_row["running_change_detected_count"])
    verdict = str(final_row["running_verdict"])
    explanations = []
    human_names = {
        "local_straightness_mean": "Cursor paths became much straighter than this player's baseline.",
        "direction_entropy_short_mean": "Movement direction entropy collapsed relative to the player's normal cursor behavior.",
        "curvature_mean": "Cursor curvature dropped, making the motion look unusually clean and direct.",
        "roughness_score_mean": "Micro-corrections became much smoother and less noisy than usual.",
        "angular_velocity_mean": "Cursor turning behavior shifted noticeably relative to the player's baseline.",
        "click_mean": "Click frequency changed materially relative to this player's usual rhythm.",
        "click_motion_coupling_mean": "Clicks became more tightly linked to cursor stabilization than usual.",
        "speed_autocorr_short_mean": "Cursor speed became more periodic or mechanically regular.",
        "speed_autocorr_short_std": "Cursor tempo variability changed sharply relative to the player's baseline.",
        "click_motion_coupling_max": "Clicks became tightly coupled to abrupt cursor stabilization.",
        "last_click_interval_ms_std": "Click intervals became much more consistent than this player's normal rhythm.",
        "last_stabilization_delay_ms_mean": "The delay between movement stabilization and clicking dropped sharply.",
        "burst_duration_ms_std": "Movement bursts became unusually uniform in duration.",
        "jitter_ms_mean": "Network jitter changed materially, which may explain degraded control quality.",
        "packet_loss_pct_mean": "Packet loss increased relative to the player's normal network conditions.",
        "command_age_ms_mean": "Server-side command age rose noticeably, consistent with lag or delayed inputs.",
        "tick_desync_ms_mean": "The server observed more client-server desynchronization than usual.",
    }
    for shift in top_shifts[:4]:
        message = human_names.get(shift["feature"], f"{shift['feature']} moved far from the baseline.")
        explanations.append(message)

    if verdict == "Likely Legit" and not gate_reasons:
        gate_reasons.append("No strong confounder was needed because the session stayed close to the player's normal cursor-motor fingerprint.")

    deduped_reasons = []
    seen = set()
    for reason in gate_reasons:
        if reason not in seen:
            deduped_reasons.append(reason)
            seen.add(reason)

    if verdict == "Likely Legit":
        explanations = ["Overall behavior stayed within the player's acceptable cursor-motor range after causal validation."]
        if deduped_reasons:
            explanations.append("The observed drift was treated as explainable noise or an external confounder, not a sustained automation signature.")

    return {
        "verdict": verdict,
        "peak_score": float(score_df["gated_score"].max()),
        "window_scores": score_df,
        "top_feature_deltas": top_shifts,
        "explanations": explanations,
        "gate_reasons": deduped_reasons,
        "high_window_count": high_window_count,
        "change_detected_count": change_detected_count,
    }
