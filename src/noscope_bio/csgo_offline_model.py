from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
CSGO_DATA_DIR = ROOT / "data" / "csgo_generated"
CSGO_MODEL_DIR = ROOT / "artifacts" / "csgo_models"

SESSION_ID_COLUMNS = {"session_id", "player_id", "player_index", "source_label", "split", "engagement_index", "label_cheat"}
BAYES_PREVALENCE_SCENARIOS = (
    ("observed_test_prevalence", None),
    ("10_percent", 0.10),
    ("5_percent", 0.05),
    ("1_percent", 0.01),
    ("0.1_percent", 0.001),
)


def _require_exported_session_table() -> pd.DataFrame:
    path = CSGO_DATA_DIR / "session_feature_table.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run python3 scripts/export_csgo_bundle.py first.")
    return pd.read_csv(path)


def get_session_feature_columns(session_features: pd.DataFrame) -> list[str]:
    return [col for col in session_features.columns if col not in SESSION_ID_COLUMNS]


def _metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prevalence = float(y_true.mean())
    sensitivity = float(recall_score(y_true, y_pred, zero_division=0))
    specificity = float(tn / max(1, fp + tn))
    majority_baseline_accuracy = float(max(prevalence, 1.0 - prevalence))
    ppv = float(tp / max(1, tp + fp))
    npv = float(tn / max(1, tn + fn))
    lr_positive = float(sensitivity / max(1e-9, 1.0 - specificity)) if specificity < 1.0 else float("inf")
    lr_negative = float((1.0 - sensitivity) / max(specificity, 1e-9))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": sensitivity,
        "specificity": specificity,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
        "false_positive_rate": float(fp / max(1, fp + tn)),
        "false_negative_rate": float(fn / max(1, fn + tp)),
        "prevalence": prevalence,
        "predicted_positive_rate": float(y_pred.mean()),
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


def _bayes_posterior_table(metrics: dict[str, float]) -> list[dict[str, float | str]]:
    sensitivity = float(metrics["recall"])
    specificity = float(metrics["specificity"])
    observed_prevalence = float(metrics["prevalence"])
    rows: list[dict[str, float | str]] = []
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
    return rows


def _threshold_search(probabilities: np.ndarray, y_true: np.ndarray) -> float:
    best_threshold = 0.5
    best_score = float("-inf")
    for threshold in np.linspace(0.20, 0.85, 14):
        preds = (probabilities >= threshold).astype(int)
        precision = precision_score(y_true, preds, zero_division=0)
        score = balanced_accuracy_score(y_true, preds) + 0.20 * precision
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def _build_candidate_models() -> dict[str, Pipeline]:
    baseline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=600, class_weight="balanced", random_state=7)),
        ]
    )
    medium = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_depth=5,
                    learning_rate=0.05,
                    max_iter=220,
                    min_samples_leaf=20,
                    validation_fraction=0.1,
                    random_state=7,
                ),
            ),
        ]
    )
    return {
        "baseline_logistic": baseline,
        "medium_hgbt": medium,
    }


def train_offline_csgo_models() -> dict:
    session_features = _require_exported_session_table()
    feature_columns = get_session_feature_columns(session_features)

    train_df = session_features[session_features["split"] == "train"].copy()
    validation_df = session_features[session_features["split"] == "validation"].copy()
    test_df = session_features[session_features["split"] == "test"].copy()

    X_train = train_df[feature_columns]
    y_train = train_df["label_cheat"].to_numpy(dtype=int)
    X_validation = validation_df[feature_columns]
    y_validation = validation_df["label_cheat"].to_numpy(dtype=int)
    X_test = test_df[feature_columns]
    y_test = test_df["label_cheat"].to_numpy(dtype=int)

    candidates = _build_candidate_models()
    results = []
    trained_models: dict[str, Pipeline] = {}

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        validation_prob = model.predict_proba(X_validation)[:, 1]
        threshold = _threshold_search(validation_prob, y_validation)

        validation_pred = (validation_prob >= threshold).astype(int)
        test_prob = model.predict_proba(X_test)[:, 1]
        test_pred = (test_prob >= threshold).astype(int)

        result = {
            "model_name": name,
            "threshold": threshold,
            "validation_metrics": _metric_row(y_validation, validation_pred),
            "test_metrics": _metric_row(y_test, test_pred),
        }
        score = result["validation_metrics"]["balanced_accuracy"] + 0.20 * result["validation_metrics"]["precision"]
        result["selection_score"] = float(score)
        results.append(result)
        trained_models[name] = model

    results_df = pd.DataFrame(
        [
            {
                "model_name": row["model_name"],
                "threshold": row["threshold"],
                "validation_balanced_accuracy": row["validation_metrics"]["balanced_accuracy"],
                "validation_precision": row["validation_metrics"]["precision"],
                "validation_recall": row["validation_metrics"]["recall"],
                "validation_f1": row["validation_metrics"]["f1"],
                "validation_specificity": row["validation_metrics"]["specificity"],
                "test_balanced_accuracy": row["test_metrics"]["balanced_accuracy"],
                "test_precision": row["test_metrics"]["precision"],
                "test_recall": row["test_metrics"]["recall"],
                "test_f1": row["test_metrics"]["f1"],
                "test_specificity": row["test_metrics"]["specificity"],
                "test_accuracy": row["test_metrics"]["accuracy"],
                "test_mcc": row["test_metrics"]["mcc"],
                "selection_score": row["selection_score"],
            }
            for row in results
        ]
    ).sort_values("selection_score", ascending=False)

    best_name = str(results_df.iloc[0]["model_name"])
    best_result = next(row for row in results if row["model_name"] == best_name)
    best_model = trained_models[best_name]
    best_threshold = float(best_result["threshold"])

    train_prob = best_model.predict_proba(X_train)[:, 1]
    validation_prob = best_model.predict_proba(X_validation)[:, 1]
    test_prob = best_model.predict_proba(X_test)[:, 1]
    split_metrics = {
        "train": _metric_row(y_train, (train_prob >= best_threshold).astype(int)),
        "validation": best_result["validation_metrics"],
        "test": best_result["test_metrics"],
    }
    bayes_reference = _bayes_posterior_table(split_metrics["test"])

    CSGO_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, CSGO_MODEL_DIR / f"{best_name}.joblib")
    joblib.dump(best_model, CSGO_MODEL_DIR / "best_model.joblib")
    results_df.to_csv(CSGO_MODEL_DIR / "model_comparison.csv", index=False)

    summary = {
        "best_model_name": best_name,
        "threshold": best_threshold,
        "feature_columns": feature_columns,
        "validation_metrics": best_result["validation_metrics"],
        "test_metrics": best_result["test_metrics"],
        "split_metrics": split_metrics,
        "bayes_reference": bayes_reference,
        "candidate_models": results,
    }
    with (CSGO_MODEL_DIR / "model_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


def load_persisted_csgo_model() -> dict | None:
    summary_path = CSGO_MODEL_DIR / "model_summary.json"
    model_path = CSGO_MODEL_DIR / "best_model.joblib"
    if not summary_path.exists() or not model_path.exists():
        return None
    with summary_path.open() as f:
        summary = json.load(f)
    model = joblib.load(model_path)
    return {
        "summary": summary,
        "model": model,
    }


def predict_with_persisted_csgo_model(feature_frame: pd.DataFrame, persisted: dict) -> pd.DataFrame:
    summary = persisted["summary"]
    model = persisted["model"]
    feature_columns = summary["feature_columns"]
    missing = [col for col in feature_columns if col not in feature_frame.columns]
    if missing:
        raise ValueError(f"Missing required feature columns for persisted CSGO model: {missing}")
    X = feature_frame[feature_columns]
    probabilities = model.predict_proba(X)[:, 1]
    threshold = float(summary["threshold"])
    return pd.DataFrame(
        {
            "session_id": feature_frame["session_id"].to_numpy(),
            "offline_probability": probabilities,
            "offline_prediction": np.where(probabilities >= threshold, "Suspicious", "Likely Legit"),
        }
    )
