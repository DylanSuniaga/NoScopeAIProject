"""Generate figures and tables used in the final report.

Reads persisted artifacts (no model retraining) and writes:

    report/figures/fig_confusion_csgo.pdf
    report/figures/fig_bayes_prevalence.pdf
    report/figures/fig_roc_comparison.pdf
    report/figures/fig_feature_importance.pdf
    report/figures/fig_synthetic_confusion.pdf
    report/tables/scenario_easy.csv
    report/tables/scenario_average.csv
    report/tables/scenario_challenging.csv
    report/tables/model_comparison_with_dummy.csv

Usage:
    python3 scripts/build_report_figures.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from noscope_bio.pipeline import export_demo_bundle


def _split_session_table(session_table: pd.DataFrame, feature_cols: list[str]):
    """Return (X_train, y_train, X_test, y_test) using the baked-in split column."""
    train_df = session_table[session_table["split"] == "train"]
    test_df = session_table[session_table["split"] == "test"]
    X_train = train_df[feature_cols].to_numpy()
    y_train = train_df["label_cheat"].to_numpy(dtype=int)
    X_test = test_df[feature_cols].to_numpy()
    y_test = test_df["label_cheat"].to_numpy(dtype=int)
    return X_train, y_train, X_test, y_test

REPORT = ROOT / "report"
FIG_DIR = REPORT / "figures"
TAB_DIR = REPORT / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _savefig(fig, name: str) -> Path:
    path = FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")
    return path


def _save_csv(df: pd.DataFrame, name: str) -> Path:
    path = TAB_DIR / name
    df.to_csv(path, index=False)
    print(f"  wrote {path.relative_to(ROOT)}")
    return path


# ---------------------------------------------------------------------------
# Load persisted artifacts
# ---------------------------------------------------------------------------

def load_csgo_artifacts() -> dict:
    art_dir = ROOT / "artifacts" / "csgo_models"
    with (art_dir / "model_summary.json").open() as fh:
        summary = json.load(fh)
    comparison = pd.read_csv(art_dir / "model_comparison.csv")

    bundle_dir = ROOT / "data" / "csgo_generated"
    session_table = pd.read_csv(bundle_dir / "session_feature_table.csv")
    with (bundle_dir / "metadata.json").open() as fh:
        metadata = json.load(fh)

    return {
        "summary": summary,
        "comparison": comparison,
        "session_table": session_table,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Figure 1: CSGO session-level confusion matrix (logistic regression)
# ---------------------------------------------------------------------------

def fig_confusion_csgo(summary: dict) -> None:
    test = summary["test_metrics"]
    cm = np.array([[test["tn"], test["fp"]], [test["fn"], test["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for (i, j), val in np.ndenumerate(cm):
        color = "white" if val > cm.max() / 2 else "black"
        ax.text(j, i, f"{val}", ha="center", va="center",
                color=color, fontsize=12, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Legit", "Cheat"])
    ax.set_yticklabels(["Legit", "Cheat"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("CSGO test-set confusion matrix\n"
                 "(logistic regression session model)")
    _savefig(fig, "fig_confusion_csgo.pdf")


# ---------------------------------------------------------------------------
# Figure 2: synthetic test confusion matrix
# ---------------------------------------------------------------------------

def fig_confusion_synthetic(synth_art: dict) -> None:
    m = synth_art["evaluation_metrics"]
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]]).astype(int)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(cm, cmap="Greens")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for (i, j), val in np.ndenumerate(cm):
        color = "white" if val > cm.max() / 2 else "black"
        ax.text(j, i, f"{val}", ha="center", va="center",
                color=color, fontsize=12, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Legit", "Cheat"])
    ax.set_yticklabels(["Legit", "Cheat"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Synthetic test-set confusion matrix")
    _savefig(fig, "fig_synthetic_confusion.pdf")


# ---------------------------------------------------------------------------
# Figure 3: Bayes posterior vs. deployment prevalence (the "base rate collapse")
# ---------------------------------------------------------------------------

def fig_bayes_prevalence(summary: dict, synth_bayes: pd.DataFrame) -> None:
    csgo = pd.DataFrame(summary["bayes_reference"])
    csgo_sorted = csgo.sort_values("assumed_prevalence")
    synth_sorted = synth_bayes.sort_values("assumed_prevalence")

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(csgo_sorted["assumed_prevalence"],
            csgo_sorted["posterior_cheat_given_positive"],
            marker="o", label="Real CSGO (logistic reg.)", color="#b91c1c")
    ax.plot(synth_sorted["assumed_prevalence"],
            synth_sorted["posterior_cheat_given_positive"],
            marker="s", label="Synthetic", color="#065f46")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.6,
            label="Prior (no classifier)")

    ax.set_xscale("log")
    ax.set_yscale("linear")
    ax.set_xlim(5e-4, 1.0)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Deployment prevalence (log scale)")
    ax.set_ylabel(r"$P(\mathrm{cheat}\ |\ \mathrm{positive})$")
    ax.set_title("Base-rate effect on positive predictive value")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left")
    _savefig(fig, "fig_bayes_prevalence.pdf")


# ---------------------------------------------------------------------------
# Figure 4: ROC comparison — logistic vs HGBT vs dummy (on session table)
# ---------------------------------------------------------------------------

def fig_roc_comparison(summary: dict, session_table: pd.DataFrame) -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier

    feature_cols = summary["feature_columns"]
    X_train, y_train, X_test, y_test = _split_session_table(session_table, feature_cols)

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    log = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ]).fit(X_train, y_train)

    hgbt = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", HistGradientBoostingClassifier(
            max_depth=4, max_iter=180, learning_rate=0.08,
            l2_regularization=1.0, random_state=42)),
    ]).fit(X_train, y_train)

    dummy = DummyClassifier(strategy="stratified", random_state=42).fit(X_train, y_train)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for name, model, color in [
        ("Logistic regression", log, "#1d4ed8"),
        ("Gradient-boosted trees", hgbt, "#b45309"),
        ("Stratified dummy", dummy, "#6b7280"),
    ]:
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, linewidth=1.6)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.7)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC on CSGO test set (session level)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower right")
    _savefig(fig, "fig_roc_comparison.pdf")


# ---------------------------------------------------------------------------
# Figure 5: Feature importance (logistic coefficient magnitude) split by family
# ---------------------------------------------------------------------------

SMOOTH_TOKENS = (
    "speed", "accel", "jerk", "curvature", "straight", "settling",
    "lock", "snap", "fire_alignment", "aim_efficiency",
    "tight_on_target", "error_improvement", "error_speed_ratio",
    "target_error", "fire_coupling", "fire_on_target",
    "fire_stability", "flick_magnitude",
)
STOCHASTIC_TOKENS = (
    "entropy", "reversal", "micro_correction", "micro_to_speed",
    "autocorr", "direction_entropy",
)


def _family(feature_name: str) -> str:
    lowered = feature_name.lower()
    if any(tok in lowered for tok in STOCHASTIC_TOKENS):
        return "Stochastic"
    if any(tok in lowered for tok in SMOOTH_TOKENS):
        return "Smooth"
    if "cheat_probability" in lowered or "emb_" in lowered or "encoder_signal" in lowered:
        return "Embedding"
    if "automation" in lowered or "movement_signature" in lowered:
        return "Aggregate"
    return "Other"


def fig_feature_importance(summary: dict, session_table: pd.DataFrame) -> None:
    feature_cols = summary["feature_columns"]
    X_train, y_train, _X_test, _y_test = _split_session_table(session_table, feature_cols)

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    log = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ]).fit(X_train, y_train)

    coef = log.named_steps["clf"].coef_[0]
    df = pd.DataFrame({
        "feature": feature_cols,
        "coef": coef,
        "abs": np.abs(coef),
    })
    df["family"] = df["feature"].map(_family)
    df = df.sort_values("abs", ascending=False).head(15)

    palette = {
        "Smooth": "#1d4ed8",
        "Stochastic": "#b91c1c",
        "Embedding": "#047857",
        "Aggregate": "#7c3aed",
        "Other": "#6b7280",
    }
    colors = df["family"].map(palette).tolist()

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    y_pos = np.arange(len(df))[::-1]
    ax.barh(y_pos, df["coef"].values, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["feature"].values, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Logistic regression coefficient (standardized features)")
    ax.set_title("Top-15 features: smooth vs. stochastic contribution")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in palette.values()]
    ax.legend(handles, palette.keys(), loc="lower right", fontsize=8, frameon=True)
    _savefig(fig, "fig_feature_importance.pdf")


# ---------------------------------------------------------------------------
# Tables: three-scenario evaluation + dummy baseline row
# ---------------------------------------------------------------------------

def tables_scenarios(summary: dict, synth_art: dict, session_table: pd.DataFrame) -> None:
    # Dummy row on CSGO (majority-class + stratified)
    feature_cols = summary["feature_columns"]
    X_train, y_train, X_test, y_test = _split_session_table(session_table, feature_cols)

    # majority-class = predict "legit" (the majority)
    y_pred_majority = np.zeros_like(y_test)
    maj_acc = float((y_pred_majority == y_test).mean())
    maj_bal_acc = float(balanced_accuracy_score(y_test, y_pred_majority))

    # stratified dummy
    dummy = DummyClassifier(strategy="stratified", random_state=42)
    dummy.fit(X_train, y_train)
    y_pred_strat = dummy.predict(X_test)
    strat_acc = float((y_pred_strat == y_test).mean())
    strat_bal_acc = float(balanced_accuracy_score(y_test, y_pred_strat))
    strat_precision = float(precision_score(y_test, y_pred_strat, zero_division=0))
    strat_recall = float(recall_score(y_test, y_pred_strat, zero_division=0))
    cm_strat = confusion_matrix(y_test, y_pred_strat)
    tn_s, fp_s, fn_s, tp_s = cm_strat.ravel()
    strat_specificity = float(tn_s / (tn_s + fp_s)) if (tn_s + fp_s) else 0.0
    strat_mcc = float(matthews_corrcoef(y_test, y_pred_strat))

    # -------- Easy: synthetic test --------
    m = synth_art["evaluation_metrics"]
    easy = pd.DataFrame([
        {"model": "NoScope-Bio (fingerprint + logistic)", "accuracy": m["accuracy"],
         "balanced_accuracy": m["balanced_accuracy"], "precision": m["precision"],
         "recall": m["recall"], "specificity": m["specificity"], "mcc": m["mcc"]},
        {"model": "Majority-class baseline", "accuracy": m["majority_baseline_accuracy"],
         "balanced_accuracy": 0.5, "precision": float("nan"),
         "recall": 0.0, "specificity": 1.0, "mcc": 0.0},
    ])
    _save_csv(easy.round(4), "scenario_easy.csv")

    # -------- Average: real CSGO at observed prevalence --------
    t = summary["test_metrics"]
    comp = pd.read_csv(ROOT / "artifacts" / "csgo_models" / "model_comparison.csv")
    avg_rows = [
        {"model": "NoScope-Bio (logistic regression)",
         "accuracy": t["accuracy"], "balanced_accuracy": t["balanced_accuracy"],
         "precision": t["precision"], "recall": t["recall"],
         "specificity": t["specificity"], "mcc": t["mcc"]},
    ]
    for _, row in comp.iterrows():
        if row["model_name"] == "baseline_logistic":
            continue
        avg_rows.append({
            "model": row["model_name"],
            "accuracy": row["test_accuracy"],
            "balanced_accuracy": row["test_balanced_accuracy"],
            "precision": row["test_precision"],
            "recall": row["test_recall"],
            "specificity": row["test_specificity"],
            "mcc": row["test_mcc"],
        })
    avg_rows.append({
        "model": "Stratified dummy",
        "accuracy": strat_acc, "balanced_accuracy": strat_bal_acc,
        "precision": strat_precision, "recall": strat_recall,
        "specificity": strat_specificity, "mcc": strat_mcc,
    })
    avg_rows.append({
        "model": "Majority-class baseline",
        "accuracy": maj_acc, "balanced_accuracy": maj_bal_acc,
        "precision": float("nan"), "recall": 0.0,
        "specificity": 1.0, "mcc": 0.0,
    })
    avg = pd.DataFrame(avg_rows)
    _save_csv(avg.round(4), "scenario_average.csv")

    # -------- Challenging: real CSGO at low deployment prevalence --------
    bayes = pd.DataFrame(summary["bayes_reference"])
    bayes_ch = bayes[bayes["scenario"].isin(
        ["observed_test_prevalence", "10_percent", "5_percent",
         "1_percent", "0.1_percent"]
    )].copy()
    bayes_ch = bayes_ch.rename(columns={
        "assumed_prevalence": "prevalence",
        "posterior_cheat_given_positive": "ppv",
        "posterior_legit_given_negative": "npv",
        "posterior_cheat_given_negative": "missed_cheater_rate",
    })
    _save_csv(bayes_ch[["scenario", "prevalence", "ppv", "npv",
                        "missed_cheater_rate"]].round(4),
              "scenario_challenging.csv")

    # -------- Table 4: consolidated model comparison with dummy --------
    cons = pd.concat([
        pd.DataFrame([
            {"model": "Majority-class baseline",
             "val_balanced_acc": float("nan"),
             "test_balanced_acc": maj_bal_acc,
             "test_precision": float("nan"),
             "test_recall": 0.0,
             "test_specificity": 1.0,
             "test_mcc": 0.0},
            {"model": "Stratified dummy",
             "val_balanced_acc": float("nan"),
             "test_balanced_acc": strat_bal_acc,
             "test_precision": strat_precision,
             "test_recall": strat_recall,
             "test_specificity": strat_specificity,
             "test_mcc": strat_mcc},
        ]),
        comp.rename(columns={
            "model_name": "model",
            "validation_balanced_accuracy": "val_balanced_acc",
            "test_balanced_accuracy": "test_balanced_acc",
            "test_precision": "test_precision",
            "test_recall": "test_recall",
            "test_specificity": "test_specificity",
            "test_mcc": "test_mcc",
        })[["model", "val_balanced_acc", "test_balanced_acc",
            "test_precision", "test_recall", "test_specificity",
            "test_mcc"]],
    ], ignore_index=True)
    _save_csv(cons.round(4), "model_comparison_with_dummy.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("[1/6] Loading CSGO artifacts ...")
    csgo = load_csgo_artifacts()

    print("[2/6] Running synthetic pipeline (for metrics + Bayes reference) ...")
    synth = export_demo_bundle()

    print("[3/6] Figure: CSGO confusion matrix")
    fig_confusion_csgo(csgo["summary"])

    print("[4/6] Figure: Synthetic confusion matrix + Bayes sweep")
    fig_confusion_synthetic(synth)
    fig_bayes_prevalence(csgo["summary"], synth["bayes_reference"])

    print("[5/6] Figure: ROC comparison + feature importance")
    fig_roc_comparison(csgo["summary"], csgo["session_table"])
    fig_feature_importance(csgo["summary"], csgo["session_table"])

    print("[6/6] Tables: three-scenario + consolidated comparison")
    tables_scenarios(csgo["summary"], synth, csgo["session_table"])

    print("\nDone. Outputs under report/figures and report/tables.")


if __name__ == "__main__":
    main()
