from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from .config import CHEAT_MODES, DEMO_FEATURE_COLUMNS, FORBIDDEN_MODEL_COLUMNS, SimulationConfig
from .dataset import combine_window_batches, window_session
from .detector import analyze_session_windows, build_player_baselines, build_window_feature_frame, fit_cheat_scorer
from .model import embed_windows, train_fingerprint_model
from .simulator import generate_player_profiles, simulate_session


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "generated"
BASELINE_SESSIONS_PER_PLAYER = 5
EVAL_REPS_PER_MODE = 3
EVAL_SPLIT_NAMES = ("calibration_train", "validation", "test")


def _assert_causal_feature_contract() -> None:
    overlap = sorted(set(DEMO_FEATURE_COLUMNS) & FORBIDDEN_MODEL_COLUMNS)
    if overlap:
        raise ValueError(f"Model feature leakage detected in DEMO_FEATURE_COLUMNS: {overlap}")


def _scale_sessions(
    fit_sessions: dict[str, pd.DataFrame],
    all_sessions: dict[str, pd.DataFrame],
) -> tuple[StandardScaler, dict[str, pd.DataFrame]]:
    scaler = StandardScaler()
    fit_frame = pd.concat([df[DEMO_FEATURE_COLUMNS] for df in fit_sessions.values()], ignore_index=True)
    scaler.fit(fit_frame)

    scaled = {}
    for session_id, df in all_sessions.items():
        copy = df.copy()
        copy[DEMO_FEATURE_COLUMNS] = scaler.transform(copy[DEMO_FEATURE_COLUMNS])
        scaled[session_id] = copy
    return scaler, scaled


def _build_session_tables(seed: int = 7) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    profiles = generate_player_profiles(seed=seed)
    sessions: dict[str, pd.DataFrame] = {}
    meta_rows = []
    sim_config = SimulationConfig()

    for player_idx, profile in enumerate(profiles):
        for baseline_idx in range(BASELINE_SESSIONS_PER_PLAYER):
            session_id = f"{profile.player_id}_baseline_{baseline_idx:02d}"
            df = simulate_session(profile, session_id, mode="clean", seed=seed + player_idx * 100 + baseline_idx, config=sim_config)
            sessions[session_id] = df
            meta_rows.append(
                {
                    "session_id": session_id,
                    "player_id": profile.player_id,
                    "mode": "clean",
                    "split": "baseline",
                    "eval_split": "",
                    "replicate": baseline_idx,
                }
            )

        eval_modes = ["clean", "aimbot", "triggerbot", "macro_consistency", "high_ping", "sensitivity_change", "patch_shift"]
        for rep_idx, eval_split in enumerate(EVAL_SPLIT_NAMES):
            for mode_idx, mode in enumerate(eval_modes):
                session_id = f"{profile.player_id}_{mode}_{rep_idx:02d}"
                df = simulate_session(profile, session_id, mode=mode, seed=seed + player_idx * 100 + 20 + rep_idx * 20 + mode_idx, config=sim_config)
                sessions[session_id] = df
                meta_rows.append(
                    {
                        "session_id": session_id,
                        "player_id": profile.player_id,
                        "mode": mode,
                        "split": "evaluation",
                        "eval_split": eval_split,
                        "replicate": rep_idx,
                    }
                )

    return sessions, pd.DataFrame(meta_rows)


def _prepare_model_inputs(sessions: dict[str, pd.DataFrame], session_meta: pd.DataFrame):
    baseline_ids = session_meta[session_meta["split"] == "baseline"]["session_id"].tolist()
    baseline_sessions = {session_id: sessions[session_id] for session_id in baseline_ids}
    _, scaled_sessions = _scale_sessions(baseline_sessions, sessions)

    X_train, meta_train = combine_window_batches({sid: scaled_sessions[sid] for sid in baseline_ids}, DEMO_FEATURE_COLUMNS)
    players = sorted(meta_train["player_id"].unique())
    player_to_idx = {player: idx for idx, player in enumerate(players)}
    y_train = meta_train["player_id"].map(player_to_idx).to_numpy(dtype=np.int64)

    return scaled_sessions, X_train, meta_train, y_train, player_to_idx


def _build_analyzer(model, baselines, scaler, cheat_scorer=None, decision_threshold: float = 0.72):
    def analyze(session_df: pd.DataFrame) -> dict:
        raw = session_df.copy()
        scaled = raw.copy()
        scaled[DEMO_FEATURE_COLUMNS] = scaler.transform(scaled[DEMO_FEATURE_COLUMNS])
        X_session, meta_session = window_session(scaled, DEMO_FEATURE_COLUMNS)
        _, summary_meta = window_session(raw, DEMO_FEATURE_COLUMNS)
        for col in summary_meta.columns:
            if col not in {"session_id", "player_id", "mode", "start_tick", "end_tick", "t_end_s", "label_cheat"}:
                meta_session[col] = summary_meta[col]
        embeddings = embed_windows(model, X_session)
        return analyze_session_windows(
            embeddings,
            meta_session,
            baselines,
            cheat_scorer=cheat_scorer,
            decision_threshold=decision_threshold,
        )

    return analyze


def _prepare_evaluation_window_features(model, baselines, scaled_sessions, raw_sessions, session_meta, session_ids: list[str]):
    eval_ids = list(session_ids)
    X_eval_scaled, meta_eval_scaled = combine_window_batches({sid: scaled_sessions[sid] for sid in eval_ids}, DEMO_FEATURE_COLUMNS)
    _, meta_eval_raw = combine_window_batches({sid: raw_sessions[sid] for sid in eval_ids}, DEMO_FEATURE_COLUMNS)
    embeddings_eval = embed_windows(model, X_eval_scaled)
    for col in meta_eval_raw.columns:
        if col not in {"session_id", "player_id", "mode", "start_tick", "end_tick", "t_end_s", "label_cheat"}:
            meta_eval_scaled[col] = meta_eval_raw[col]
    feature_frame = build_window_feature_frame(embeddings_eval, meta_eval_scaled, baselines)
    return feature_frame


def _scan_best_threshold(analysis_cache: dict[str, dict], session_meta: pd.DataFrame) -> float:
    eval_rows = session_meta.copy()
    y_true = np.array([int(mode in CHEAT_MODES) for mode in eval_rows["mode"]], dtype=int)
    session_ids = eval_rows["session_id"].tolist()
    best_threshold = 0.72
    best_score = float("-inf")

    for threshold in np.linspace(0.45, 0.90, 19):
        preds = np.array(
            [
                int(
                    analysis_cache[session_id]["peak_score"] >= threshold
                    and (
                        analysis_cache[session_id]["change_detected_count"] > 0
                        or analysis_cache[session_id]["high_window_count"] >= 3
                    )
                )
                for session_id in session_ids
            ],
            dtype=int,
        )
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        fpr = fp / max(1, fp + tn)
        objective = acc + 0.25 * prec + 0.15 * rec - 0.35 * fpr
        if objective > best_score:
            best_score = objective
            best_threshold = float(threshold)

    return best_threshold


def _evaluate_sessions(analysis_cache: dict[str, dict], session_meta: pd.DataFrame) -> dict[str, float]:
    eval_rows = session_meta.copy()
    y_true = []
    y_pred = []
    for _, row in eval_rows.iterrows():
        session_id = row["session_id"]
        y_true.append(int(row["mode"] in CHEAT_MODES))
        y_pred.append(int(analysis_cache[session_id]["verdict"] == "Suspicious"))

    y_true_arr = np.array(y_true, dtype=int)
    y_pred_arr = np.array(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).ravel()

    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "false_positive_rate": float(fp / max(1, fp + tn)),
        "false_negative_rate": float(fn / max(1, fn + tp)),
    }


def _session_ids_for_eval_split(session_meta: pd.DataFrame, eval_split: str) -> list[str]:
    return session_meta[
        (session_meta["split"] == "evaluation") & (session_meta["eval_split"] == eval_split)
    ]["session_id"].tolist()


def build_demo_pipeline(seed: int = 7) -> dict:
    _assert_causal_feature_contract()
    sessions, session_meta = _build_session_tables(seed=seed)
    scaled_sessions, X_train, meta_train, y_train, player_to_idx = _prepare_model_inputs(sessions, session_meta)
    model, training_metrics = train_fingerprint_model(X_train, y_train, num_players=len(player_to_idx))

    baseline_ids = session_meta[session_meta["split"] == "baseline"]["session_id"].tolist()
    X_baseline, meta_baseline = combine_window_batches({sid: scaled_sessions[sid] for sid in baseline_ids}, DEMO_FEATURE_COLUMNS)
    raw_baseline_X, raw_baseline_meta = combine_window_batches({sid: sessions[sid] for sid in baseline_ids}, DEMO_FEATURE_COLUMNS)
    embeddings = embed_windows(model, X_baseline)
    for col in raw_baseline_meta.columns:
        if col not in {"session_id", "player_id", "mode", "start_tick", "end_tick", "t_end_s", "label_cheat"}:
            meta_baseline[col] = raw_baseline_meta[col]
    baselines = build_player_baselines(embeddings, meta_baseline)

    baseline_sessions = {sid: sessions[sid] for sid in baseline_ids}
    scaler, _ = _scale_sessions(baseline_sessions, sessions)
    calibration_ids = _session_ids_for_eval_split(session_meta, "calibration_train")
    validation_ids = _session_ids_for_eval_split(session_meta, "validation")
    test_ids = _session_ids_for_eval_split(session_meta, "test")
    eval_ids = session_meta[session_meta["split"] == "evaluation"]["session_id"].tolist()

    calibration_feature_frame = _prepare_evaluation_window_features(
        model,
        baselines,
        scaled_sessions,
        sessions,
        session_meta,
        calibration_ids,
    )
    cheat_scorer = fit_cheat_scorer(calibration_feature_frame)

    calibration_analyzer = _build_analyzer(model, baselines, scaler, cheat_scorer=cheat_scorer, decision_threshold=0.72)
    evaluation_sessions = {sid: sessions[sid] for sid in eval_ids}
    validation_cache = {sid: calibration_analyzer(sessions[sid]) for sid in validation_ids}
    validation_meta = session_meta[(session_meta["split"] == "evaluation") & (session_meta["eval_split"] == "validation")].copy()
    decision_threshold = _scan_best_threshold(validation_cache, validation_meta)

    analyzer = _build_analyzer(model, baselines, scaler, cheat_scorer=cheat_scorer, decision_threshold=decision_threshold)
    analysis_cache = {sid: analyzer(df) for sid, df in evaluation_sessions.items()}
    split_metrics = {}
    for eval_split in EVAL_SPLIT_NAMES:
        split_meta = session_meta[(session_meta["split"] == "evaluation") & (session_meta["eval_split"] == eval_split)].copy()
        split_metrics[eval_split] = _evaluate_sessions(analysis_cache, split_meta)
    evaluation_metrics = split_metrics["test"]

    return {
        "model": model,
        "training_metrics": training_metrics,
        "player_to_idx": player_to_idx,
        "baselines": baselines,
        "cheat_scorer": cheat_scorer,
        "decision_threshold": decision_threshold,
        "session_meta": session_meta,
        "evaluation_sessions": evaluation_sessions,
        "default_session_ids": test_ids,
        "split_metrics": split_metrics,
        "analysis_cache": analysis_cache,
        "evaluation_metrics": evaluation_metrics,
        "analyzer": analyzer,
    }


def export_demo_bundle(seed: int = 7) -> dict:
    artifacts = build_demo_pipeline(seed=seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session_meta = artifacts["session_meta"]

    for session_id, df in artifacts["evaluation_sessions"].items():
        df.to_csv(DATA_DIR / f"{session_id}.csv", index=False)

    session_meta.to_csv(DATA_DIR / "session_manifest.csv", index=False)
    pd.DataFrame([artifacts["evaluation_metrics"]]).to_csv(DATA_DIR / "evaluation_metrics.csv", index=False)
    pd.DataFrame(artifacts["split_metrics"]).T.to_csv(DATA_DIR / "split_metrics.csv", index_label="split")
    return artifacts


def load_or_build_session_from_upload(upload) -> dict:
    raw_bytes = upload.read()
    session_df = pd.read_csv(io.BytesIO(raw_bytes))
    required = {"session_id", "player_id", "mode", "tick", "t"} | set(DEMO_FEATURE_COLUMNS) | {"cursor_x", "cursor_y", "label_cheat"}
    missing = sorted(required - set(session_df.columns))
    if missing:
        raise ValueError(f"Uploaded CSV is missing required columns: {missing}")

    meta = {
        "session_id": str(session_df["session_id"].iloc[0]),
        "player_id": str(session_df["player_id"].iloc[0]),
        "mode": str(session_df["mode"].iloc[0]),
    }
    return {"session_df": session_df, "meta": meta}
