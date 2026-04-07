from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from noscope_bio.config import DEMO_FEATURE_COLUMNS
from noscope_bio.config import SERVER_TELEMETRY_COLUMNS
from noscope_bio.csgo_offline_model import load_persisted_csgo_model, predict_with_persisted_csgo_model
from noscope_bio.csgo_pipeline import CSGO_FEATURE_COLUMNS as CSGO_ENGINEERED_FEATURE_COLUMNS
from noscope_bio.csgo_pipeline import CSGO_TELEMETRY_COLUMNS
from noscope_bio.csgo_pipeline import load_exported_csgo_bundle
from noscope_bio.pipeline import build_demo_pipeline, load_or_build_session_from_upload


st.set_page_config(page_title="NoScope-Bio", page_icon=":dart:", layout="wide")

PIPELINE_CACHE_VERSION = "2026-04-06-04"


def normalize_artifacts(artifacts: dict) -> dict:
    normalized = dict(artifacts)
    normalized.setdefault("decision_threshold", 0.8)
    normalized.setdefault("split_metrics", {"test": normalized.get("evaluation_metrics", {})})
    normalized.setdefault("default_session_ids", sorted(normalized.get("evaluation_sessions", {}).keys()))
    return normalized


@st.cache_resource(show_spinner=True)
def get_artifacts(cache_version: str = PIPELINE_CACHE_VERSION):
    _ = cache_version
    return normalize_artifacts(build_demo_pipeline())


def normalize_csgo_artifacts(artifacts: dict) -> dict:
    normalized = dict(artifacts)
    normalized.setdefault("decision_threshold", 0.8)
    return normalized


def _humanize_model_name(name: str) -> str:
    mapping = {
        "baseline_logistic": "Logistic Regression",
        "medium_hgbt": "HistGradientBoosting",
        "window_pipeline": "Causal Window Pipeline",
    }
    return mapping.get(name, name.replace("_", " ").title())


@st.cache_resource(show_spinner=True)
def get_csgo_artifacts(cache_version: str = PIPELINE_CACHE_VERSION):
    _ = cache_version
    artifacts = load_exported_csgo_bundle()
    if artifacts is None:
        return None
    return normalize_csgo_artifacts(artifacts)


@st.cache_resource(show_spinner=False)
def get_persisted_csgo_model(cache_version: str = PIPELINE_CACHE_VERSION):
    _ = cache_version
    return load_persisted_csgo_model()


def _get_replay_columns(session_df: pd.DataFrame) -> tuple[str, str]:
    x_col = "replay_x" if "replay_x" in session_df.columns else "view_yaw"
    y_col = "replay_y" if "replay_y" in session_df.columns else "view_pitch"
    return x_col, y_col


def _suggest_csgo_replay_range(session_df: pd.DataFrame, before_s: float = 3.5, after_s: float = 1.5) -> tuple[float, float]:
    min_t = float(session_df["t"].min())
    max_t = float(session_df["t"].max())
    fire_rows = session_df[session_df["fire_input"] > 0.5]
    if fire_rows.empty:
        start = min_t
        end = min(max_t, min_t + 5.0)
        return (start, end)
    focus_t = float(fire_rows["t"].median())
    start = max(min_t, focus_t - before_s)
    end = min(max_t, focus_t + after_s)
    if end - start < 4.5:
        deficit = 4.5 - (end - start)
        start = max(min_t, start - deficit * 0.65)
        end = min(max_t, end + deficit * 0.35)
    return (float(start), float(end))


def build_timeline_figure(scores: pd.DataFrame, threshold: float, current_t: float | None = None):
    if current_t is None:
        observed = scores
        future = pd.DataFrame(columns=scores.columns)
    else:
        observed = scores[scores["t_end_s"] <= current_t]
        future = scores[scores["t_end_s"] > current_t]

    fig = go.Figure()
    if current_t is not None and not future.empty:
        fig.add_trace(
            go.Scatter(
                x=future["t_end_s"],
                y=future["gated_score"],
                mode="lines",
                name="Upcoming Score",
                line=dict(color="#d9d9d9", width=2, dash="dot"),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=observed["t_end_s"],
            y=observed["raw_score"],
            mode="lines+markers",
            name="Raw Suspicion",
            line=dict(color="#ff6b35", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=observed["t_end_s"],
            y=observed["gated_score"],
            mode="lines",
            name="After Causal Gate",
            line=dict(color="#1f77b4", width=3),
        )
    )
    fig.add_hline(y=threshold, line_dash="dash", line_color="#888", annotation_text="Flag threshold")
    cp_rows = scores[scores["change_detected"] == 1]
    if not cp_rows.empty:
        first_cp = cp_rows.iloc[0]
        fig.add_vline(
            x=first_cp["t_end_s"],
            line_dash="dot",
            line_color="#d62728",
            annotation_text="Change point",
        )
    if current_t is not None:
        fig.add_vline(
            x=current_t,
            line_dash="solid",
            line_color="#2ca02c",
            annotation_text="Playback",
        )
    fig.update_layout(
        title="Suspicion Timeline",
        xaxis_title="Time (s)",
        yaxis_title="Score",
        template="plotly_white",
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def render_timeline(scores: pd.DataFrame, threshold: float, current_t: float | None = None, chart_target=None):
    target = chart_target or st
    target.plotly_chart(build_timeline_figure(scores, threshold=threshold, current_t=current_t), use_container_width=True)


def _split_trail(frame_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame_df.empty:
        return frame_df, frame_df
    split_idx = max(1, int(len(frame_df) * 0.55))
    return frame_df.iloc[:split_idx], frame_df.iloc[split_idx:]


def _build_animation_controls(frame_names: list[str], frame_duration_ms: int) -> tuple[list[dict], list[dict]]:
    controls = [
        {
            "type": "buttons",
            "showactive": False,
            "x": 0.02,
            "y": 1.12,
            "direction": "left",
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [
                        None,
                        {
                            "frame": {"duration": frame_duration_ms, "redraw": True},
                            "transition": {"duration": 0},
                            "fromcurrent": True,
                            "mode": "immediate",
                        },
                    ],
                },
                {
                    "label": "Pause",
                    "method": "animate",
                    "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                },
            ],
        }
    ]
    slider = [
        {
            "active": 0,
            "x": 0.08,
            "len": 0.88,
            "pad": {"t": 40},
            "currentvalue": {"prefix": "Playback: "},
            "steps": [
                {
                    "label": name,
                    "method": "animate",
                    "args": [
                        [name],
                        {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}},
                    ],
                }
                for name in frame_names
            ],
        }
    ]
    return controls, slider


def build_combined_animation_figure(
    session_df: pd.DataFrame,
    scores: pd.DataFrame,
    start_t: float,
    end_t: float,
    threshold: float,
    frame_times: list[float],
    frame_duration_ms: int,
    replay_style: str = "FPS Overlay",
    trail_seconds: float = 2.5,
):
    playback_df = session_df[(session_df["t"] >= start_t) & (session_df["t"] <= end_t)].copy()
    score_df = scores[(scores["t_end_s"] >= start_t) & (scores["t_end_s"] <= end_t)].copy()
    if playback_df.empty or score_df.empty:
        return None

    x_col, y_col = _get_replay_columns(playback_df)
    frame_names = [f"{t:.1f}s" for t in frame_times]
    dark_mode = replay_style == "FPS Overlay"
    background = "#091014" if dark_mode else "#ffffff"
    paper = "#05090c" if dark_mode else "#ffffff"
    foreground = "#e8f3f8" if dark_mode else "#111111"
    first_time = frame_times[0]
    first_trail = playback_df[(playback_df["t"] >= max(start_t, first_time - trail_seconds)) & (playback_df["t"] <= first_time)]
    first_trail = first_trail if not first_trail.empty else playback_df.iloc[[0]]
    first_old_trail, first_recent_trail = _split_trail(first_trail)
    first_latest = first_trail.iloc[-1]
    first_scores = score_df[score_df["t_end_s"] <= first_time]
    first_score = first_scores.iloc[-1] if not first_scores.empty else score_df.iloc[0]
    first_clicks = first_trail[first_trail["fire_input"] > 0]
    y_max = max(float(score_df["raw_score"].max()), float(score_df["gated_score"].max()), float(threshold)) * 1.1

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,
        row_heights=[0.42, 0.26, 0.32],
        subplot_titles=("View-Angle Replay Feed", "Model Output", "Server Telemetry"),
    )

    fig.add_trace(
        go.Scatter(x=first_old_trail[x_col], y=first_old_trail[y_col], mode="lines", name="Older Aim Trace", line=dict(color="#7cc9ff", width=2), opacity=0.22),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=first_recent_trail[x_col], y=first_recent_trail[y_col], mode="lines", name="Recent Aim Trace", line=dict(color="#ff6b35", width=3)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=first_clicks[x_col],
            y=first_clicks[y_col],
            mode="markers",
            name="Fire Inputs",
            marker=dict(size=10, symbol="diamond", color="#f6c453", line=dict(width=1, color="#40330a")),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=[first_latest[x_col]], y=[first_latest[y_col]], mode="markers", name="Current Aim", marker=dict(size=18, symbol="cross", color="#ff6b35")),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=[first_score["t_end_s"]], y=[first_score["raw_score"]], mode="lines", name="Raw Suspicion", line=dict(color="#ff6b35", width=3)),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=[first_score["t_end_s"]], y=[first_score["gated_score"]], mode="lines", name="After Causal Gate", line=dict(color="#1f77b4", width=3)),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=[first_score["t_end_s"]], y=[first_score["raw_score"]], mode="markers", name="Current Raw", marker=dict(size=10, color="#ff6b35")),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=[first_score["t_end_s"]], y=[first_score["gated_score"]], mode="markers", name="Current Gated", marker=dict(size=10, color="#1f77b4")),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=[first_score["t_end_s"], first_score["t_end_s"]], y=[0.0, y_max], mode="lines", name="Playback Cursor", line=dict(color="#2ca02c", dash="dot", width=2)),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=[first_latest["t"]], y=[first_latest["ping_ms"]], mode="lines", name="Ping (ms)", line=dict(color="#1f77b4", width=2)),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=[first_latest["t"]], y=[first_latest["jitter_ms"]], mode="lines", name="Jitter (ms)", line=dict(color="#ff7f0e", width=2)),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=[first_latest["t"]], y=[first_latest["packet_loss_pct"] * 20.0], mode="lines", name="Packet Loss x20", line=dict(color="#d62728", width=2)),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=[first_latest["t"]], y=[first_latest["ping_ms"]], mode="markers", name="Current Ping", marker=dict(size=9, color="#1f77b4")),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=[first_latest["t"]], y=[first_latest["jitter_ms"]], mode="markers", name="Current Jitter", marker=dict(size=9, color="#ff7f0e")),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=[first_latest["t"]], y=[first_latest["packet_loss_pct"] * 20.0], mode="markers", name="Current Loss", marker=dict(size=9, color="#d62728")),
        row=3,
        col=1,
    )

    frames = []
    for name, current_t in zip(frame_names, frame_times):
        trail_start = max(start_t, current_t - trail_seconds)
        frame_trail = playback_df[(playback_df["t"] >= trail_start) & (playback_df["t"] <= current_t)]
        if frame_trail.empty:
            continue
        latest = frame_trail.iloc[-1]
        old_trail, recent_trail = _split_trail(frame_trail)
        click_points = frame_trail[frame_trail["fire_input"] > 0]
        observed_scores = score_df[score_df["t_end_s"] <= current_t]
        if observed_scores.empty:
            continue
        score_row = observed_scores.iloc[-1]
        observed_network = playback_df[playback_df["t"] <= current_t]
        frames.append(
            go.Frame(
                name=name,
                data=[
                    go.Scatter(x=old_trail[x_col], y=old_trail[y_col]),
                    go.Scatter(x=recent_trail[x_col], y=recent_trail[y_col]),
                    go.Scatter(x=click_points[x_col], y=click_points[y_col]),
                    go.Scatter(x=[latest[x_col]], y=[latest[y_col]]),
                    go.Scatter(x=observed_scores["t_end_s"], y=observed_scores["raw_score"]),
                    go.Scatter(x=observed_scores["t_end_s"], y=observed_scores["gated_score"]),
                    go.Scatter(x=[score_row["t_end_s"]], y=[score_row["raw_score"]]),
                    go.Scatter(x=[score_row["t_end_s"]], y=[score_row["gated_score"]]),
                    go.Scatter(x=[score_row["t_end_s"], score_row["t_end_s"]], y=[0.0, y_max]),
                    go.Scatter(x=observed_network["t"], y=observed_network["ping_ms"]),
                    go.Scatter(x=observed_network["t"], y=observed_network["jitter_ms"]),
                    go.Scatter(x=observed_network["t"], y=observed_network["packet_loss_pct"] * 20.0),
                    go.Scatter(x=[latest["t"]], y=[latest["ping_ms"]]),
                    go.Scatter(x=[latest["t"]], y=[latest["jitter_ms"]]),
                    go.Scatter(x=[latest["t"]], y=[latest["packet_loss_pct"] * 20.0]),
                ],
                traces=list(range(15)),
            )
        )
    fig.frames = frames

    cp_rows = score_df[score_df["change_detected"] == 1]
    if not cp_rows.empty:
        cp = cp_rows.iloc[0]
        fig.add_vline(x=cp["t_end_s"], line_dash="dot", line_color="#d62728", annotation_text="Change point", row=2, col=1)
    fig.add_hline(y=threshold, line_dash="dash", line_color="#888", annotation_text="Flag threshold", row=2, col=1)

    controls, slider = _build_animation_controls(frame_names, frame_duration_ms)
    fig.add_shape(type="circle", x0=-1.02, y0=-1.02, x1=1.02, y1=1.02, line=dict(color="#1f2e36", width=2), row=1, col=1)
    fig.add_annotation(
        x=-1.02,
        y=1.05,
        xanchor="left",
        yanchor="bottom",
        text="NO SCOPE-BIO OBSERVER FEED",
        showarrow=False,
        font=dict(color=foreground, size=12, family="Courier New"),
        row=1,
        col=1,
    )
    fig.update_xaxes(range=[-1.1, 1.1], visible=False, showgrid=False, zeroline=False, row=1, col=1)
    fig.update_yaxes(range=[-1.1, 1.1], visible=False, showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_xaxes(range=[start_t, end_t], title="Time (s)", row=2, col=1)
    fig.update_yaxes(range=[0.0, y_max], title="Score", row=2, col=1)
    fig.update_xaxes(range=[start_t, end_t], title="Time (s)", row=3, col=1)
    fig.update_yaxes(title="Scaled Value", row=3, col=1)
    fig.update_layout(
        title="Synchronized Live Animation",
        template="plotly_dark" if dark_mode else "plotly_white",
        height=980,
        margin=dict(l=20, r=20, t=80, b=20),
        plot_bgcolor=background,
        paper_bgcolor=paper,
        font=dict(color=foreground),
        legend=dict(bgcolor="rgba(0,0,0,0.2)", orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        updatemenus=controls,
        sliders=slider,
    )
    return fig


def build_animated_timeline_figure(scores: pd.DataFrame, threshold: float, frame_times: list[float], frame_duration_ms: int):
    y_max = max(float(scores["raw_score"].max()), float(scores["gated_score"].max()), float(threshold)) * 1.1
    first_time = frame_times[0] if frame_times else float(scores["t_end_s"].iloc[0])
    first = scores[scores["t_end_s"] <= first_time].iloc[-1]
    fig = go.Figure(
        data=[
            go.Scatter(x=[first["t_end_s"]], y=[first["raw_score"]], mode="lines", name="Raw Suspicion", line=dict(color="#ff6b35", width=3)),
            go.Scatter(x=[first["t_end_s"]], y=[first["gated_score"]], mode="lines", name="After Causal Gate", line=dict(color="#1f77b4", width=3)),
            go.Scatter(x=[first["t_end_s"]], y=[first["raw_score"]], mode="markers", name="Current Raw", marker=dict(size=11, color="#ff6b35")),
            go.Scatter(x=[first["t_end_s"]], y=[first["gated_score"]], mode="markers", name="Current Gated", marker=dict(size=11, color="#1f77b4")),
            go.Scatter(x=[first["t_end_s"], first["t_end_s"]], y=[0.0, y_max], mode="lines", name="Playback Cursor", line=dict(color="#2ca02c", dash="dot", width=2)),
        ]
    )
    cp_rows = scores[scores["change_detected"] == 1]
    if not cp_rows.empty:
        cp = cp_rows.iloc[0]
        fig.add_vline(x=cp["t_end_s"], line_dash="dot", line_color="#d62728", annotation_text="Change point")
    fig.add_hline(y=threshold, line_dash="dash", line_color="#888", annotation_text="Flag threshold")
    frame_names = [f"{t:.1f}s" for t in frame_times]
    frames = []
    for name, current_t in zip(frame_names, frame_times):
        row = scores[scores["t_end_s"] <= current_t].iloc[-1]
        observed = scores[scores["t_end_s"] <= current_t]
        frames.append(
            go.Frame(
                name=name,
                data=[
                    go.Scatter(x=observed["t_end_s"], y=observed["raw_score"]),
                    go.Scatter(x=observed["t_end_s"], y=observed["gated_score"]),
                    go.Scatter(x=[row["t_end_s"]], y=[row["raw_score"]]),
                    go.Scatter(x=[row["t_end_s"]], y=[row["gated_score"]]),
                    go.Scatter(x=[row["t_end_s"], row["t_end_s"]], y=[0.0, y_max]),
                ],
                traces=[0, 1, 2, 3, 4],
            )
        )
    fig.frames = frames
    controls, slider = _build_animation_controls(frame_names, frame_duration_ms)
    fig.update_layout(
        title="Playback Timeline",
        template="plotly_white",
        xaxis_title="Time (s)",
        yaxis_title="Score",
        height=360,
        margin=dict(l=20, r=20, t=60, b=20),
        updatemenus=controls,
        sliders=slider,
    )
    return fig


def build_replay_figure(
    session_df: pd.DataFrame,
    start_t: float,
    end_t: float,
    current_t: float | None = None,
    replay_style: str = "FPS Overlay",
):
    replay_df = session_df[(session_df["t"] >= start_t) & (session_df["t"] <= end_t)].copy()
    if replay_df.empty:
        return None

    x_col, y_col = _get_replay_columns(replay_df)
    sampled = replay_df.iloc[:: max(1, len(replay_df) // 120)].copy()
    sampled["frame_index"] = range(len(sampled))
    fig = go.Figure()
    old_trail, recent_trail = _split_trail(sampled)
    fig.add_trace(
        go.Scatter(
            x=old_trail[x_col],
            y=old_trail[y_col],
            mode="lines",
            name="Older Aim Trace",
            line=dict(color="#7cc9ff", width=2),
            opacity=0.22,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=recent_trail[x_col],
            y=recent_trail[y_col],
            mode="lines",
            name="Recent Aim Trace",
            line=dict(color="#ef553b", width=3),
        )
    )
    if current_t is not None:
        current_rows = replay_df[replay_df["t"] <= current_t]
        latest = current_rows.iloc[-1] if not current_rows.empty else sampled.iloc[-1]
        click_points = current_rows[current_rows["fire_input"] > 0].tail(10)
    else:
        latest = sampled.iloc[-1]
        click_points = replay_df[replay_df["fire_input"] > 0].tail(10)

    dark_mode = replay_style == "FPS Overlay"
    background = "#091014" if dark_mode else "#ffffff"
    paper = "#05090c" if dark_mode else "#ffffff"
    foreground = "#e8f3f8" if dark_mode else "#111111"

    fig.add_trace(
        go.Scatter(
            x=click_points[x_col],
            y=click_points[y_col],
            mode="markers",
            name="Recent Fire Inputs",
            marker=dict(size=10, symbol="diamond", color="#f6c453", line=dict(width=1, color="#40330a")),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[latest[x_col]],
            y=[latest[y_col]],
            mode="markers",
            name="Current Aim",
            marker=dict(size=18, symbol="cross", color="#ff6b35"),
        )
    )
    gap = 0.035
    arm = 0.08
    cx = float(latest[x_col])
    cy = float(latest[y_col])
    reticle_color = "#ff6b35"
    fig.add_shape(type="line", x0=cx - arm, y0=cy, x1=cx - gap, y1=cy, line=dict(color=reticle_color, width=2))
    fig.add_shape(type="line", x0=cx + gap, y0=cy, x1=cx + arm, y1=cy, line=dict(color=reticle_color, width=2))
    fig.add_shape(type="line", x0=cx, y0=cy - arm, x1=cx, y1=cy - gap, line=dict(color=reticle_color, width=2))
    fig.add_shape(type="line", x0=cx, y0=cy + gap, x1=cx, y1=cy + arm, line=dict(color=reticle_color, width=2))
    fig.add_shape(type="circle", x0=cx - 0.018, y0=cy - 0.018, x1=cx + 0.018, y1=cy + 0.018, line=dict(color=reticle_color, width=2))
    fig.add_shape(type="circle", x0=-1.02, y0=-1.02, x1=1.02, y1=1.02, line=dict(color="#1f2e36", width=2))
    fig.add_annotation(
        x=-1.02,
        y=1.05,
        xanchor="left",
        yanchor="bottom",
        text="NO SCOPE-BIO OBSERVER FEED",
        showarrow=False,
        font=dict(color=foreground, size=12, family="Courier New"),
    )
    fig.update_layout(
        title="View-Angle Replay View",
        template="plotly_dark" if dark_mode else "plotly_white",
        xaxis=dict(range=[-1.1, 1.1], title="", visible=False, showgrid=False, zeroline=False),
        yaxis=dict(range=[-1.1, 1.1], title="", visible=False, showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1),
        height=430,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor=background,
        paper_bgcolor=paper,
        font=dict(color=foreground),
        legend=dict(bgcolor="rgba(0,0,0,0.2)"),
    )
    return fig


def render_replay(
    session_df: pd.DataFrame,
    start_t: float,
    end_t: float,
    current_t: float | None = None,
    chart_target=None,
    replay_style: str = "FPS Overlay",
):
    fig = build_replay_figure(session_df, start_t, end_t, current_t=current_t, replay_style=replay_style)
    if fig is None:
        (chart_target or st).info("No replay frames in the selected range.")
        return
    (chart_target or st).plotly_chart(fig, use_container_width=True)


def build_animated_replay_figure(
    session_df: pd.DataFrame,
    start_t: float,
    end_t: float,
    frame_times: list[float],
    frame_duration_ms: int,
    replay_style: str = "FPS Overlay",
):
    replay_df = session_df[(session_df["t"] >= start_t) & (session_df["t"] <= end_t)].copy()
    if replay_df.empty:
        return None
    x_col, y_col = _get_replay_columns(replay_df)
    dark_mode = replay_style == "FPS Overlay"
    background = "#091014" if dark_mode else "#ffffff"
    paper = "#05090c" if dark_mode else "#ffffff"
    foreground = "#e8f3f8" if dark_mode else "#111111"
    first = replay_df.iloc[0]
    frame_names = [f"{t:.1f}s" for t in frame_times]

    fig = go.Figure(
        data=[
            go.Scatter(x=[first[x_col]], y=[first[y_col]], mode="lines", name="Older Aim Trace", line=dict(color="#7cc9ff", width=2), opacity=0.22),
            go.Scatter(x=[first[x_col]], y=[first[y_col]], mode="lines", name="Recent Aim Trace", line=dict(color="#ff6b35", width=2)),
            go.Scatter(x=[], y=[], mode="markers", name="Fire Inputs", marker=dict(size=10, symbol="diamond", color="#f6c453", line=dict(width=1, color="#40330a"))),
            go.Scatter(x=[first[x_col]], y=[first[y_col]], mode="markers", name="Current Aim", marker=dict(size=18, symbol="cross", color="#ff6b35")),
        ]
    )
    frames = []
    for name, current_t in zip(frame_names, frame_times):
        frame_df = replay_df[replay_df["t"] <= current_t]
        if frame_df.empty:
            continue
        latest = frame_df.iloc[-1]
        old_trail, recent_trail = _split_trail(frame_df)
        click_points = frame_df[frame_df["fire_input"] > 0]
        frames.append(
            go.Frame(
                name=name,
                data=[
                    go.Scatter(x=old_trail[x_col], y=old_trail[y_col]),
                    go.Scatter(x=recent_trail[x_col], y=recent_trail[y_col]),
                    go.Scatter(x=click_points[x_col], y=click_points[y_col]),
                    go.Scatter(x=[latest[x_col]], y=[latest[y_col]]),
                ],
                traces=[0, 1, 2, 3],
            )
        )
    fig.frames = frames
    controls, slider = _build_animation_controls(frame_names, frame_duration_ms)
    fig.add_shape(type="circle", x0=-1.02, y0=-1.02, x1=1.02, y1=1.02, line=dict(color="#1f2e36", width=2))
    fig.add_annotation(
        x=-1.02,
        y=1.05,
        xanchor="left",
        yanchor="bottom",
        text="NO SCOPE-BIO OBSERVER FEED",
        showarrow=False,
        font=dict(color=foreground, size=12, family="Courier New"),
    )
    fig.update_layout(
        title="Browser Playback View-Angle Replay",
        template="plotly_dark" if dark_mode else "plotly_white",
        xaxis=dict(range=[-1.1, 1.1], title="", visible=False, showgrid=False, zeroline=False),
        yaxis=dict(range=[-1.1, 1.1], title="", visible=False, showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1),
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor=background,
        paper_bgcolor=paper,
        font=dict(color=foreground),
        legend=dict(bgcolor="rgba(0,0,0,0.2)"),
        updatemenus=controls,
        sliders=slider,
    )
    return fig


def build_network_figure(session_df: pd.DataFrame, start_t: float, end_t: float, current_t: float | None = None):
    network_df = session_df[(session_df["t"] >= start_t) & (session_df["t"] <= end_t)].copy()
    if network_df.empty:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=network_df["t"],
            y=network_df["ping_ms"],
            mode="lines",
            name="Ping (ms)",
            line=dict(color="#1f77b4", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=network_df["t"],
            y=network_df["jitter_ms"],
            mode="lines",
            name="Jitter (ms)",
            line=dict(color="#ff7f0e", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=network_df["t"],
            y=network_df["packet_loss_pct"] * 20.0,
            mode="lines",
            name="Packet Loss x20",
            line=dict(color="#d62728", width=2),
        )
    )
    if current_t is not None:
        fig.add_vline(
            x=current_t,
            line_dash="solid",
            line_color="#2ca02c",
            annotation_text="Playback",
        )
    fig.update_layout(
        title="Server Telemetry",
        template="plotly_white",
        xaxis_title="Time (s)",
        yaxis_title="Scaled Value",
        height=280,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def render_network_figure(session_df: pd.DataFrame, start_t: float, end_t: float, current_t: float | None = None, chart_target=None):
    fig = build_network_figure(session_df, start_t, end_t, current_t=current_t)
    if fig is None:
        (chart_target or st).info("No network telemetry in the selected range.")
        return
    (chart_target or st).plotly_chart(fig, use_container_width=True)


def build_csgo_telemetry_figure(session_df: pd.DataFrame, start_t: float, end_t: float, current_t: float | None = None):
    telemetry_df = session_df[(session_df["t"] >= start_t) & (session_df["t"] <= end_t)].copy()
    if telemetry_df.empty:
        return None
    fire_scale = max(4.0, float(np.nanpercentile(telemetry_df["target_error"], 75)))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=telemetry_df["t"], y=telemetry_df["target_error"], mode="lines", name="Target Error", line=dict(color="#1f77b4", width=2))
    )
    fig.add_trace(
        go.Scatter(x=telemetry_df["t"], y=telemetry_df["angular_speed"], mode="lines", name="Angular Speed", line=dict(color="#ff7f0e", width=2))
    )
    fig.add_trace(
        go.Scatter(x=telemetry_df["t"], y=telemetry_df["fire_input"] * fire_scale, mode="lines", name="Firing x scale", line=dict(color="#d62728", width=2))
    )
    if current_t is not None:
        fig.add_vline(x=current_t, line_dash="solid", line_color="#2ca02c", annotation_text="Playback")
    fig.update_layout(
        title="CSGO Engagement Telemetry",
        template="plotly_white",
        xaxis_title="Time (s)",
        yaxis_title="Value / scaled firing",
        height=280,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def render_csgo_telemetry_figure(session_df: pd.DataFrame, start_t: float, end_t: float, current_t: float | None = None, chart_target=None):
    fig = build_csgo_telemetry_figure(session_df, start_t, end_t, current_t=current_t)
    if fig is None:
        (chart_target or st).info("No CSGO telemetry in the selected range.")
        return
    (chart_target or st).plotly_chart(fig, use_container_width=True)


def build_csgo_combined_animation_figure(
    session_df: pd.DataFrame,
    scores: pd.DataFrame,
    start_t: float,
    end_t: float,
    threshold: float,
    frame_times: list[float],
    frame_duration_ms: int,
    replay_style: str = "FPS Overlay",
    trail_seconds: float = 2.5,
):
    playback_df = session_df[(session_df["t"] >= start_t) & (session_df["t"] <= end_t)].copy()
    score_df = scores[(scores["t_end_s"] >= start_t) & (scores["t_end_s"] <= end_t)].copy()
    if playback_df.empty or score_df.empty:
        return None

    x_col, y_col = _get_replay_columns(playback_df)
    frame_names = [f"{t:.1f}s" for t in frame_times]
    dark_mode = replay_style == "FPS Overlay"
    background = "#091014" if dark_mode else "#ffffff"
    paper = "#05090c" if dark_mode else "#ffffff"
    foreground = "#e8f3f8" if dark_mode else "#111111"
    fire_scale = max(4.0, float(np.nanpercentile(playback_df["target_error"], 75)))

    first_time = frame_times[0]
    first_trail = playback_df[(playback_df["t"] >= max(start_t, first_time - trail_seconds)) & (playback_df["t"] <= first_time)]
    first_trail = first_trail if not first_trail.empty else playback_df.iloc[[0]]
    first_old_trail, first_recent_trail = _split_trail(first_trail)
    first_latest = first_trail.iloc[-1]
    first_scores = score_df[score_df["t_end_s"] <= first_time]
    first_score = first_scores.iloc[-1] if not first_scores.empty else score_df.iloc[0]
    first_clicks = first_trail[first_trail["fire_input"] > 0]
    y_max = max(float(score_df["raw_score"].max()), float(score_df["gated_score"].max()), float(threshold)) * 1.1

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,
        row_heights=[0.42, 0.26, 0.32],
        subplot_titles=("CSGO View-Angle Replay Feed", "Model Output", "Engagement Telemetry"),
    )

    fig.add_trace(go.Scatter(x=first_old_trail[x_col], y=first_old_trail[y_col], mode="lines", name="Older Aim Trace", line=dict(color="#7cc9ff", width=2), opacity=0.22), row=1, col=1)
    fig.add_trace(go.Scatter(x=first_recent_trail[x_col], y=first_recent_trail[y_col], mode="lines", name="Recent Aim Trace", line=dict(color="#ff6b35", width=3)), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=first_clicks[x_col], y=first_clicks[y_col], mode="markers", name="Fire Inputs", marker=dict(size=10, symbol="diamond", color="#f6c453", line=dict(width=1, color="#40330a"))),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=[first_latest[x_col]], y=[first_latest[y_col]], mode="markers", name="Current Aim", marker=dict(size=18, symbol="cross", color="#ff6b35")), row=1, col=1)

    fig.add_trace(go.Scatter(x=[first_score["t_end_s"]], y=[first_score["raw_score"]], mode="lines", name="Raw Suspicion", line=dict(color="#ff6b35", width=3)), row=2, col=1)
    fig.add_trace(go.Scatter(x=[first_score["t_end_s"]], y=[first_score["gated_score"]], mode="lines", name="After Gate", line=dict(color="#1f77b4", width=3)), row=2, col=1)
    fig.add_trace(go.Scatter(x=[first_score["t_end_s"]], y=[first_score["raw_score"]], mode="markers", name="Current Raw", marker=dict(size=10, color="#ff6b35")), row=2, col=1)
    fig.add_trace(go.Scatter(x=[first_score["t_end_s"]], y=[first_score["gated_score"]], mode="markers", name="Current Gated", marker=dict(size=10, color="#1f77b4")), row=2, col=1)
    fig.add_trace(go.Scatter(x=[first_score["t_end_s"], first_score["t_end_s"]], y=[0.0, y_max], mode="lines", name="Playback Cursor", line=dict(color="#2ca02c", dash="dot", width=2)), row=2, col=1)

    fig.add_trace(go.Scatter(x=[first_latest["t"]], y=[first_latest["target_error"]], mode="lines", name="Target Error", line=dict(color="#1f77b4", width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=[first_latest["t"]], y=[first_latest["angular_speed"]], mode="lines", name="Angular Speed", line=dict(color="#ff7f0e", width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=[first_latest["t"]], y=[first_latest["fire_input"] * fire_scale], mode="lines", name="Firing x scale", line=dict(color="#d62728", width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=[first_latest["t"]], y=[first_latest["target_error"]], mode="markers", name="Current Error", marker=dict(size=9, color="#1f77b4")), row=3, col=1)
    fig.add_trace(go.Scatter(x=[first_latest["t"]], y=[first_latest["angular_speed"]], mode="markers", name="Current Speed", marker=dict(size=9, color="#ff7f0e")), row=3, col=1)
    fig.add_trace(go.Scatter(x=[first_latest["t"]], y=[first_latest["fire_input"] * fire_scale], mode="markers", name="Current Fire", marker=dict(size=9, color="#d62728")), row=3, col=1)

    frames = []
    for name, current_t in zip(frame_names, frame_times):
        trail_start = max(start_t, current_t - trail_seconds)
        frame_trail = playback_df[(playback_df["t"] >= trail_start) & (playback_df["t"] <= current_t)]
        if frame_trail.empty:
            continue
        latest = frame_trail.iloc[-1]
        old_trail, recent_trail = _split_trail(frame_trail)
        click_points = frame_trail[frame_trail["fire_input"] > 0]
        observed_scores = score_df[score_df["t_end_s"] <= current_t]
        if observed_scores.empty:
            continue
        score_row = observed_scores.iloc[-1]
        observed_telemetry = playback_df[playback_df["t"] <= current_t]
        frames.append(
            go.Frame(
                name=name,
                data=[
                    go.Scatter(x=old_trail[x_col], y=old_trail[y_col]),
                    go.Scatter(x=recent_trail[x_col], y=recent_trail[y_col]),
                    go.Scatter(x=click_points[x_col], y=click_points[y_col]),
                    go.Scatter(x=[latest[x_col]], y=[latest[y_col]]),
                    go.Scatter(x=observed_scores["t_end_s"], y=observed_scores["raw_score"]),
                    go.Scatter(x=observed_scores["t_end_s"], y=observed_scores["gated_score"]),
                    go.Scatter(x=[score_row["t_end_s"]], y=[score_row["raw_score"]]),
                    go.Scatter(x=[score_row["t_end_s"]], y=[score_row["gated_score"]]),
                    go.Scatter(x=[score_row["t_end_s"], score_row["t_end_s"]], y=[0.0, y_max]),
                    go.Scatter(x=observed_telemetry["t"], y=observed_telemetry["target_error"]),
                    go.Scatter(x=observed_telemetry["t"], y=observed_telemetry["angular_speed"]),
                    go.Scatter(x=observed_telemetry["t"], y=observed_telemetry["fire_input"] * fire_scale),
                    go.Scatter(x=[latest["t"]], y=[latest["target_error"]]),
                    go.Scatter(x=[latest["t"]], y=[latest["angular_speed"]]),
                    go.Scatter(x=[latest["t"]], y=[latest["fire_input"] * fire_scale]),
                ],
                traces=list(range(15)),
            )
        )
    fig.frames = frames

    cp_rows = score_df[score_df["change_detected"] == 1]
    if not cp_rows.empty:
        cp = cp_rows.iloc[0]
        fig.add_vline(x=cp["t_end_s"], line_dash="dot", line_color="#d62728", annotation_text="Change point", row=2, col=1)
    fig.add_hline(y=threshold, line_dash="dash", line_color="#888", annotation_text="Flag threshold", row=2, col=1)

    controls, slider = _build_animation_controls(frame_names, frame_duration_ms)
    fig.add_shape(type="circle", x0=-1.02, y0=-1.02, x1=1.02, y1=1.02, line=dict(color="#1f2e36", width=2), row=1, col=1)
    fig.add_annotation(x=-1.02, y=1.05, xanchor="left", yanchor="bottom", text="NO SCOPE-BIO CSGO ARCHIVE FEED", showarrow=False, font=dict(color=foreground, size=12, family="Courier New"), row=1, col=1)
    fig.update_xaxes(range=[-1.1, 1.1], visible=False, showgrid=False, zeroline=False, row=1, col=1)
    fig.update_yaxes(range=[-1.1, 1.1], visible=False, showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_xaxes(range=[start_t, end_t], title="Time (s)", row=2, col=1)
    fig.update_yaxes(range=[0.0, y_max], title="Score", row=2, col=1)
    fig.update_xaxes(range=[start_t, end_t], title="Time (s)", row=3, col=1)
    fig.update_yaxes(title="Value / scaled firing", row=3, col=1)
    fig.update_layout(
        title="CSGO Archive Live Animation",
        template="plotly_dark" if dark_mode else "plotly_white",
        height=980,
        margin=dict(l=20, r=20, t=80, b=20),
        plot_bgcolor=background,
        paper_bgcolor=paper,
        font=dict(color=foreground),
        legend=dict(bgcolor="rgba(0,0,0,0.2)", orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        updatemenus=controls,
        sliders=slider,
    )
    return fig


def build_animated_network_figure(session_df: pd.DataFrame, start_t: float, end_t: float, frame_times: list[float], frame_duration_ms: int):
    network_df = session_df[(session_df["t"] >= start_t) & (session_df["t"] <= end_t)].copy()
    if network_df.empty:
        return None
    first = network_df.iloc[0]
    frame_names = [f"{t:.1f}s" for t in frame_times]
    fig = go.Figure(
        data=[
            go.Scatter(x=[first["t"]], y=[first["ping_ms"]], mode="lines", name="Ping (ms)", line=dict(color="#1f77b4", width=2)),
            go.Scatter(x=[first["t"]], y=[first["jitter_ms"]], mode="lines", name="Jitter (ms)", line=dict(color="#ff7f0e", width=2)),
            go.Scatter(x=[first["t"]], y=[first["packet_loss_pct"] * 20.0], mode="lines", name="Packet Loss x20", line=dict(color="#d62728", width=2)),
            go.Scatter(x=[first["t"]], y=[first["ping_ms"]], mode="markers", name="Current Ping", marker=dict(size=10, color="#1f77b4")),
            go.Scatter(x=[first["t"]], y=[first["jitter_ms"]], mode="markers", name="Current Jitter", marker=dict(size=10, color="#ff7f0e")),
            go.Scatter(x=[first["t"]], y=[first["packet_loss_pct"] * 20.0], mode="markers", name="Current Loss", marker=dict(size=10, color="#d62728")),
        ]
    )
    frames = []
    for name, current_t in zip(frame_names, frame_times):
        frame_df = network_df[network_df["t"] <= current_t]
        if frame_df.empty:
            continue
        latest = frame_df.iloc[-1]
        frames.append(
            go.Frame(
                name=name,
                data=[
                    go.Scatter(x=frame_df["t"], y=frame_df["ping_ms"]),
                    go.Scatter(x=frame_df["t"], y=frame_df["jitter_ms"]),
                    go.Scatter(x=frame_df["t"], y=frame_df["packet_loss_pct"] * 20.0),
                    go.Scatter(x=[latest["t"]], y=[latest["ping_ms"]]),
                    go.Scatter(x=[latest["t"]], y=[latest["jitter_ms"]]),
                    go.Scatter(x=[latest["t"]], y=[latest["packet_loss_pct"] * 20.0]),
                ],
                traces=[0, 1, 2, 3, 4, 5],
            )
        )
    fig.frames = frames
    controls, slider = _build_animation_controls(frame_names, frame_duration_ms)
    fig.update_layout(
        title="Browser Playback Server Telemetry",
        template="plotly_white",
        xaxis_title="Time (s)",
        yaxis_title="Scaled Value",
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        updatemenus=controls,
        sliders=slider,
    )
    return fig


def render_explanations(result: dict):
    st.subheader("What Changed?")
    for line in result["explanations"]:
        st.write(f"- {line}")
    if result["gate_reasons"]:
        st.subheader("Causal Gate Notes")
        for line in result["gate_reasons"]:
            st.write(f"- {line}")


def render_feature_shift(result: dict):
    df = pd.DataFrame(result["top_feature_deltas"])
    if df.empty:
        return
    fig = px.bar(
        df,
        x="delta",
        y="feature",
        orientation="h",
        color="direction",
        color_discrete_map={"up": "#d62728", "down": "#2ca02c"},
        title="Top Feature Shifts vs Baseline",
    )
    fig.update_layout(template="plotly_white", height=320, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)


def render_holdout_metrics(artifacts: dict, feature_schema: dict, caption_text: str):
    st.subheader("Test Results")
    eval_metrics = artifacts["evaluation_metrics"]
    headline_metrics = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "majority_baseline_accuracy",
    ]
    metric_cols = st.columns(len(headline_metrics))
    for col, name in zip(metric_cols, headline_metrics):
        value = eval_metrics[name]
        col.metric(name.replace("_", " ").title(), f"{value:.3f}")

    if {"tp", "fp", "tn", "fn"}.issubset(eval_metrics.keys()):
        c_tp, c_fp, c_tn, c_fn = st.columns(4)
        c_tp.metric("TP", f"{int(eval_metrics['tp'])}")
        c_fp.metric("FP", f"{int(eval_metrics['fp'])}")
        c_tn.metric("TN", f"{int(eval_metrics['tn'])}")
        c_fn.metric("FN", f"{int(eval_metrics['fn'])}")

    split_metrics_df = pd.DataFrame(artifacts["split_metrics"]).T.reset_index().rename(columns={"index": "split"})
    st.caption(caption_text)
    st.dataframe(split_metrics_df, use_container_width=True, hide_index=True)

    st.subheader("Bayesian Quality Check")
    st.caption(
        "Posterior cheat probabilities are computed from Bayes' theorem using the measured sensitivity and specificity. This helps show whether the detector is still useful when cheating is rare in deployment."
    )
    bayes_df = artifacts["bayes_reference"].copy()
    bayes_df["assumed_prevalence"] = bayes_df["assumed_prevalence"].map(lambda value: f"{value * 100:.2f}%")
    for column in ["posterior_cheat_given_positive", "posterior_legit_given_negative", "posterior_cheat_given_negative"]:
        bayes_df[column] = bayes_df[column].map(lambda value: f"{value * 100:.2f}%")
    st.dataframe(bayes_df, use_container_width=True, hide_index=True)

    with st.expander("Feature Schema"):
        st.write(feature_schema)


def build_offline_metric_source(persisted_summary: dict) -> dict:
    return {
        "evaluation_metrics": persisted_summary["test_metrics"],
        "split_metrics": persisted_summary["split_metrics"],
        "bayes_reference": pd.DataFrame(persisted_summary["bayes_reference"]),
    }


def _format_feature_label(name: str) -> str:
    return name.replace("_", " ").title()


def _first_event_timestamp(score_df: pd.DataFrame, threshold: float) -> float | None:
    if score_df.empty:
        return None
    if "change_detected" in score_df.columns:
        cp_rows = score_df[score_df["change_detected"] == 1]
        if not cp_rows.empty:
            return float(cp_rows.iloc[0]["t_end_s"])
    score_col = "gated_score" if "gated_score" in score_df.columns else "raw_score"
    crossing = score_df[score_df[score_col] >= threshold]
    if not crossing.empty:
        return float(crossing.iloc[0]["t_end_s"])
    return None


def _build_anomaly_feature_table(result: dict, limit: int = 5) -> pd.DataFrame:
    rows = []
    for item in result.get("top_feature_deltas", [])[:limit]:
        rows.append(
            {
                "Feature": _format_feature_label(str(item["feature"])),
                "Deviation": round(float(item["delta"]), 3),
                "Direction": str(item["direction"]).title(),
            }
        )
    return pd.DataFrame(rows)


def render_anomaly_summary(result: dict, threshold: float, confidence_score: float, detected_label: str):
    st.subheader("Anomaly Summary")
    score_df = result["window_scores"]
    event_t = _first_event_timestamp(score_df, threshold)
    feature_df = _build_anomaly_feature_table(result)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Detected Anomaly", "Yes" if detected_label == "Suspicious" else "No")
    c2.metric("Timestamp Of Event", f"{event_t:.2f} s" if event_t is not None else "No event")
    c3.metric("Confidence / Deviation", f"{confidence_score:.3f}")
    c4.metric("Affected Features", str(len(feature_df)))
    if not feature_df.empty:
        st.dataframe(feature_df, use_container_width=True, hide_index=True)


def _assign_showcase_difficulty(
    frame: pd.DataFrame,
    score_col: str,
    threshold: float,
    predicted_label_col: str,
    true_label_col: str,
    nuance_mask: pd.Series | None = None,
) -> pd.Series:
    working = frame.copy()
    working["_predicted_suspicious"] = working[predicted_label_col].astype(str).eq("Suspicious").astype(int)
    working["_true_cheat"] = working[true_label_col].astype(int)
    working["_margin"] = (working[score_col].astype(float) - float(threshold)).abs()
    working["_is_correct"] = working["_predicted_suspicious"] == working["_true_cheat"]
    correct_margins = working.loc[working["_is_correct"], "_margin"]
    if correct_margins.empty:
        low_cut = float(working["_margin"].quantile(0.33))
        high_cut = float(working["_margin"].quantile(0.67))
    else:
        low_cut = float(correct_margins.quantile(0.33))
        high_cut = float(correct_margins.quantile(0.67))
    nuance = nuance_mask if nuance_mask is not None else pd.Series(False, index=working.index)
    buckets = []
    for idx, row in working.iterrows():
        if not bool(row["_is_correct"]):
            buckets.append("hard")
            continue
        margin = float(row["_margin"])
        if margin >= high_cut:
            bucket = "easy"
        elif margin >= low_cut:
            bucket = "medium"
        else:
            bucket = "hard"
        if bool(nuance.loc[idx]) and bucket == "easy":
            bucket = "medium"
        buckets.append(bucket)
    return pd.Series(buckets, index=working.index)


def build_synthetic_showcase_manifest(artifacts: dict, session_ids: list[str]) -> pd.DataFrame:
    meta = artifacts["session_meta"].copy()
    meta = meta[(meta["split"] == "evaluation") & (meta["session_id"].isin(session_ids))].copy()
    meta["label_cheat"] = meta["mode"].isin({"aimbot", "triggerbot", "macro_consistency"}).astype(int)
    meta["predicted_label"] = meta["session_id"].map(lambda sid: artifacts["analysis_cache"][sid]["verdict"])
    meta["confidence_score"] = meta["session_id"].map(lambda sid: float(artifacts["analysis_cache"][sid]["peak_score"]))
    meta["change_detected_count"] = meta["session_id"].map(lambda sid: int(artifacts["analysis_cache"][sid]["change_detected_count"]))
    nuance_mask = meta["mode"].isin({"macro_consistency", "high_ping", "sensitivity_change", "patch_shift"})
    meta["difficulty"] = _assign_showcase_difficulty(
        meta,
        score_col="confidence_score",
        threshold=float(artifacts["decision_threshold"]),
        predicted_label_col="predicted_label",
        true_label_col="label_cheat",
        nuance_mask=nuance_mask,
    )
    meta["difficulty_rank"] = meta["difficulty"].map({"easy": 0, "medium": 1, "hard": 2}).fillna(3)
    meta["case_label"] = meta.apply(
        lambda row: f"{row['session_id']} | {row['mode']} | {row['difficulty'].title()} | {row['predicted_label']}",
        axis=1,
    )
    return meta.sort_values(["difficulty_rank", "label_cheat", "confidence_score"], ascending=[True, False, False]).reset_index(drop=True)


def build_csgo_showcase_manifest(csgo_artifacts: dict, persisted_model: dict | None) -> pd.DataFrame:
    manifest = csgo_artifacts["sample_manifest"].copy()
    if persisted_model is not None:
        session_table = csgo_artifacts["session_feature_table"]
        feature_rows = session_table[session_table["session_id"].isin(manifest["session_id"])].copy()
        if not feature_rows.empty:
            predictions = predict_with_persisted_csgo_model(feature_rows, persisted_model)
            manifest = manifest.merge(predictions, on="session_id", how="left")
            manifest["predicted_label"] = manifest["offline_prediction"].fillna(manifest["verdict"])
            manifest["confidence_score"] = manifest["offline_probability"].fillna(manifest["peak_score"])
            threshold = float(persisted_model["summary"]["threshold"])
        else:
            manifest["predicted_label"] = manifest["verdict"]
            manifest["confidence_score"] = manifest["peak_score"]
            threshold = float(csgo_artifacts["decision_threshold"])
    else:
        manifest["predicted_label"] = manifest["verdict"]
        manifest["confidence_score"] = manifest["peak_score"]
        threshold = float(csgo_artifacts["decision_threshold"])
    manifest["difficulty"] = _assign_showcase_difficulty(
        manifest,
        score_col="confidence_score",
        threshold=threshold,
        predicted_label_col="predicted_label",
        true_label_col="label_cheat",
    )
    manifest["difficulty_rank"] = manifest["difficulty"].map({"easy": 0, "medium": 1, "hard": 2}).fillna(3)
    manifest["case_label"] = manifest.apply(
        lambda row: f"{row['session_id']} | {row['source_label']} | {row['difficulty'].title()} | {row['predicted_label']}",
        axis=1,
    )
    return manifest.sort_values(["difficulty_rank", "label_cheat", "confidence_score"], ascending=[True, False, False]).reset_index(drop=True)


def render_synthetic_tab(artifacts: dict):
    st.sidebar.header("Synthetic Session Input")
    upload = st.sidebar.file_uploader("Upload telemetry CSV", type=["csv"])
    session_view_split = st.sidebar.selectbox("Bundled Session Split", ["test", "validation", "calibration_train", "all"], index=0)
    session_meta_df = artifacts["session_meta"]
    if session_view_split == "all":
        session_ids = sorted(artifacts["evaluation_sessions"].keys())
    else:
        session_ids = sorted(
            session_meta_df[(session_meta_df["split"] == "evaluation") & (session_meta_df["eval_split"] == session_view_split)]["session_id"].tolist()
        )
    if not session_ids:
        session_ids = sorted(artifacts["default_session_ids"])
    replay_style = st.sidebar.selectbox("Replay Style", ["FPS Overlay", "Clean Arena"], index=0)

    if upload is not None:
        selected = load_or_build_session_from_upload(upload)
        session_df = selected["session_df"]
        result = artifacts["analyzer"](session_df)
        session_meta = selected["meta"]
    else:
        showcase_manifest = build_synthetic_showcase_manifest(artifacts, session_ids)
        difficulty_filter = st.sidebar.selectbox("Showcase Difficulty", ["all", "easy", "medium", "hard"], index=0, key="synthetic_difficulty")
        if difficulty_filter != "all":
            showcase_manifest = showcase_manifest[showcase_manifest["difficulty"] == difficulty_filter]
        mode_options = ["all"] + sorted(showcase_manifest["mode"].unique().tolist()) if not showcase_manifest.empty else ["all"]
        mode_filter = st.sidebar.selectbox("Scenario Type", mode_options, index=0, key="synthetic_mode_filter")
        if mode_filter != "all":
            showcase_manifest = showcase_manifest[showcase_manifest["mode"] == mode_filter]
        if showcase_manifest.empty:
            st.sidebar.warning("No synthetic cases matched that difficulty filter. Falling back to all cases.")
            showcase_manifest = build_synthetic_showcase_manifest(artifacts, session_ids)
        selected_case = st.sidebar.selectbox("Bundled Demo Case", showcase_manifest["case_label"].tolist(), key="synthetic_case")
        selected_session_id = showcase_manifest.loc[showcase_manifest["case_label"] == selected_case, "session_id"].iloc[0]
        session_df = artifacts["evaluation_sessions"][selected_session_id]
        result = artifacts["analysis_cache"][selected_session_id]
        session_meta = showcase_manifest.set_index("session_id").loc[selected_session_id].to_dict()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted Verdict", result["verdict"])
    c2.metric("Mode", session_meta["mode"])
    c3.metric("Player", session_meta["player_id"])
    c4.metric("Peak Score", f"{result['peak_score']:.2f}")
    if "running_verdict" in result["window_scores"].columns:
        st.caption("Each point on the timeline is computed causally from the current window and past history only. The session verdict is the last running verdict.")
    render_anomaly_summary(result, artifacts["decision_threshold"], float(result["peak_score"]), str(result["verdict"]))

    render_timeline(result["window_scores"], threshold=artifacts["decision_threshold"])

    score_df = result["window_scores"]
    start_default = float(score_df["t_end_s"].min())
    end_default = float(score_df["t_end_s"].max())
    replay_range = st.slider(
        "Replay range (seconds)",
        min_value=float(session_df["t"].min()),
        max_value=float(session_df["t"].max()),
        value=(start_default, min(end_default, start_default + 12.0)),
        step=0.5,
        key="synthetic_replay_range",
    )
    render_replay(session_df, replay_range[0], replay_range[1], replay_style=replay_style)
    render_network_figure(session_df, replay_range[0], replay_range[1])

    st.subheader("Live Playback")
    animation_stride = st.slider("Animation frame stride", min_value=1, max_value=4, value=2, step=1, help="Higher values reduce the number of playback frames and make the animation lighter.", key="synthetic_stride")
    frame_duration_ms = st.select_slider("Animation speed", options=[120, 180, 260, 360, 520], value=260, format_func=lambda value: f"{value} ms/frame", help="This controls the built-in browser animation speed.", key="synthetic_speed")
    trail_seconds = st.slider("Replay trail length (seconds)", min_value=0.8, max_value=6.0, value=2.2, step=0.2, help="Older replay points drop out as the animation advances so the FPS view stays readable.", key="synthetic_trail")
    open_playback = st.button("Open Live Animation Studio", use_container_width=True, key="synthetic_playback")
    if open_playback:
        st.session_state["show_playback_studio"] = True

    if st.session_state.get("show_playback_studio", False):
        st.caption("One Play button now drives the view-angle replay, model score, and server telemetry together. The replay uses a trailing window so older points fall away.")
        score_times = [float(t) for t in result["window_scores"]["t_end_s"].tolist() if replay_range[0] <= float(t) <= replay_range[1]]
        frame_times = score_times[::animation_stride] if score_times else [float(replay_range[0]), float(replay_range[1])]
        if score_times and float(score_times[-1]) not in frame_times:
            frame_times.append(float(score_times[-1]))
        combined_animation = build_combined_animation_figure(
            session_df,
            result["window_scores"],
            replay_range[0],
            replay_range[1],
            artifacts["decision_threshold"],
            frame_times,
            frame_duration_ms,
            replay_style=replay_style,
            trail_seconds=trail_seconds,
        )
        if combined_animation is not None:
            st.plotly_chart(combined_animation, use_container_width=True)

    left, right = st.columns(2)
    with left:
        render_explanations(result)
    with right:
        render_feature_shift(result)

    render_holdout_metrics(
        artifacts,
        {"behavioral_features": DEMO_FEATURE_COLUMNS, "server_features": SERVER_TELEMETRY_COLUMNS},
        "These are held-out test metrics by default. The table below shows calibration-train, validation, and test splits, including class-balance-aware metrics.",
    )


def render_csgo_tab():
    csgo_artifacts = get_csgo_artifacts()
    if csgo_artifacts is None:
        st.info("Run `python3 scripts/export_csgo_bundle.py` first to build the CSGO archive bundle, then refresh this tab.")
        return
    persisted_model = get_persisted_csgo_model()

    st.caption("Archive benchmark mode for real CSGO engagements. This uses the uploaded `(players, 30, 192, 5)` archive with held-out player splits.")
    archive_summary = csgo_artifacts["archive_summary"]
    persisted_summary = persisted_model["summary"] if persisted_model is not None else None
    official_model_name = persisted_summary["best_model_name"] if persisted_summary is not None else "window_pipeline"
    official_threshold = float(persisted_summary["threshold"]) if persisted_summary is not None else float(csgo_artifacts["decision_threshold"])
    metrics_source = build_offline_metric_source(persisted_summary) if persisted_summary is not None else csgo_artifacts
    top_a, top_b, top_c, top_d = st.columns(4)
    top_a.metric("Legit Players In Archive", f"{int(archive_summary['legit_players_in_archive'])}")
    top_b.metric("Cheater Players In Archive", f"{int(archive_summary['cheater_players_in_archive'])}")
    top_c.metric("Legit Players In Current Export", f"{int(archive_summary['legit_players_used'])}")
    top_d.metric("Cheater Players In Current Export", f"{int(archive_summary['cheater_players_used'])}")
    st.caption("The archive itself contains 10,000 legit players and 2,000 cheater players. The current exported run only uses the smaller subset shown above so the demo rebuilds quickly; it is not the full archive size.")
    top_e, top_f, top_g, top_h = st.columns(4)
    top_e.metric("Official Model", _humanize_model_name(official_model_name))
    top_f.metric("Held-Out Balanced Acc.", f"{metrics_source['evaluation_metrics']['balanced_accuracy']:.3f}")
    if persisted_summary is not None:
        top_g.metric("Test Recall", f"{persisted_summary['test_metrics']['recall']:.3f}")
        top_h.metric("Test Precision", f"{persisted_summary['test_metrics']['precision']:.3f}")
        p1, p2 = st.columns(2)
        p1.metric("Session Threshold", f"{official_threshold:.2f}")
        p2.metric("Validation Bal. Acc.", f"{persisted_summary['validation_metrics']['balanced_accuracy']:.3f}")
        st.caption("The CSGO tab now uses the saved logistic regression session model as the official verdict because it had the best balanced-accuracy result in the offline model sweep. The live causal window timeline stays below as supporting evidence and explainability.")
    else:
        top_g.metric("Window Threshold", f"{csgo_artifacts['decision_threshold']:.2f}")
        top_h.metric("Test Recall", f"{metrics_source['evaluation_metrics']['recall']:.3f}")
        st.caption("No persisted CSGO session model was found, so this tab is falling back to the causal window pipeline for both verdicts and metrics.")
    st.caption("The default replay window is now centered around firing activity so you get more pre-shot context during review.")
    manifest = build_csgo_showcase_manifest(csgo_artifacts, persisted_model)
    group_filter = st.selectbox("Sample Group", ["all", "cheater", "legit"], index=0, key="csgo_group")
    difficulty_filter = st.selectbox("Showcase Difficulty", ["all", "easy", "medium", "hard"], index=0, key="csgo_difficulty")
    if group_filter != "all":
        manifest = manifest[manifest["source_label"] == group_filter]
    if difficulty_filter != "all":
        manifest = manifest[manifest["difficulty"] == difficulty_filter]
    if manifest.empty:
        st.warning("No sample sessions matched that filter.")
        return
    selected_case = st.selectbox("Choose CSGO archive engagement", manifest["case_label"].tolist(), key="csgo_session")
    selected_session_id = manifest.loc[manifest["case_label"] == selected_case, "session_id"].iloc[0]
    replay_style = st.selectbox("Replay Style", ["FPS Overlay", "Clean Arena"], index=0, key="csgo_replay_style")

    session_df = csgo_artifacts["sample_sessions"][selected_session_id]
    result = csgo_artifacts["sample_reports"][selected_session_id]
    session_meta = manifest.set_index("session_id").loc[selected_session_id].to_dict()
    offline_prediction = None
    if persisted_model is not None:
        session_feature_row = csgo_artifacts["session_feature_table"][csgo_artifacts["session_feature_table"]["session_id"] == selected_session_id].copy()
        if not session_feature_row.empty:
            offline_prediction = predict_with_persisted_csgo_model(session_feature_row, persisted_model).iloc[0].to_dict()

    c1, c2, c3, c4 = st.columns(4)
    official_verdict = offline_prediction["offline_prediction"] if offline_prediction is not None else result["verdict"]
    official_probability = float(offline_prediction["offline_probability"]) if offline_prediction is not None else float(result["peak_score"])
    c1.metric("Session Verdict", official_verdict)
    c2.metric("Ground Truth", "Cheater" if int(session_meta["label_cheat"]) == 1 else "Legit")
    c3.metric("Player", session_meta["player_id"])
    c4.metric("Session Probability", f"{official_probability:.3f}")
    if offline_prediction is not None:
        o1, o2 = st.columns(2)
        o1.metric("Window Evidence Verdict", result["verdict"])
        o2.metric("Window Peak Score", f"{result['peak_score']:.3f}")
    else:
        st.caption("This tab uses real CSGO engagement telemetry. The timeline is scored on trailing windows and the held-out metrics come from player-level train/validation/test splits.")
    if offline_prediction is not None:
        st.caption("The session verdict above comes from the saved logistic regression classifier over aggregated causal window features. The timeline below is the supporting causal window evidence stream.")
    render_anomaly_summary(result, csgo_artifacts["decision_threshold"], official_probability, official_verdict)
    if offline_prediction is not None:
        st.caption("The event timestamp above comes from the first causal window change point in the evidence stream. The confidence score above comes from the session-level logistic regression model.")

    render_timeline(result["window_scores"], threshold=csgo_artifacts["decision_threshold"])

    score_df = result["window_scores"]
    start_default, end_default = _suggest_csgo_replay_range(session_df)
    replay_range = st.slider(
        "Replay range (seconds)",
        min_value=float(session_df["t"].min()),
        max_value=float(session_df["t"].max()),
        value=(start_default, end_default),
        step=0.25,
        key="csgo_replay_range",
    )
    render_replay(session_df, replay_range[0], replay_range[1], replay_style=replay_style)
    render_csgo_telemetry_figure(session_df, replay_range[0], replay_range[1])

    st.subheader("Live Playback")
    animation_stride = st.slider("Animation frame stride", min_value=1, max_value=4, value=2, step=1, key="csgo_stride")
    frame_duration_ms = st.select_slider("Animation speed", options=[120, 180, 260, 360, 520], value=260, format_func=lambda value: f"{value} ms/frame", key="csgo_speed")
    trail_seconds = st.slider("Replay trail length (seconds)", min_value=0.6, max_value=3.5, value=1.6, step=0.1, key="csgo_trail")
    open_playback = st.button("Open CSGO Animation Studio", use_container_width=True, key="csgo_playback")
    if open_playback:
        st.session_state["show_csgo_playback_studio"] = True

    if st.session_state.get("show_csgo_playback_studio", False):
        score_times = [float(t) for t in result["window_scores"]["t_end_s"].tolist() if replay_range[0] <= float(t) <= replay_range[1]]
        frame_times = score_times[::animation_stride] if score_times else [float(replay_range[0]), float(replay_range[1])]
        if score_times and float(score_times[-1]) not in frame_times:
            frame_times.append(float(score_times[-1]))
        combined_animation = build_csgo_combined_animation_figure(
            session_df,
            result["window_scores"],
            replay_range[0],
            replay_range[1],
            csgo_artifacts["decision_threshold"],
            frame_times,
            frame_duration_ms,
            replay_style=replay_style,
            trail_seconds=trail_seconds,
        )
        if combined_animation is not None:
            st.plotly_chart(combined_animation, use_container_width=True)

    left, right = st.columns(2)
    with left:
        render_explanations(result)
    with right:
        render_feature_shift(result)

    render_holdout_metrics(
        metrics_source,
        {
            "archive_summary": csgo_artifacts["archive_summary"],
            "model_summary": csgo_artifacts.get("model_summary", {}),
            "saved_offline_model": persisted_model["summary"] if persisted_model is not None else "Run python3 scripts/train_csgo_session_models.py to persist an offline session model.",
            "raw_feature_names": csgo_artifacts["raw_feature_names"],
            "engineered_features": CSGO_ENGINEERED_FEATURE_COLUMNS,
            "telemetry_plot_features": CSGO_TELEMETRY_COLUMNS,
        },
        "These are held-out player-split metrics for the official CSGO session model shown in this tab. Logistic regression won the offline model sweep on balanced accuracy, so it is the deployed session classifier here; the causal timeline is kept as supporting evidence.",
    )


def main():
    st.title("NoScope-Bio")
    st.caption("Synthetic personalized demo plus a real CSGO archive benchmark")
    artifacts = get_artifacts()
    if "decision_threshold" not in artifacts:
        artifacts = normalize_artifacts(build_demo_pipeline())
    demo_tab, csgo_tab = st.tabs(["Synthetic Demo", "CSGO Archive"])
    with demo_tab:
        render_synthetic_tab(artifacts)
    with csgo_tab:
        render_csgo_tab()


if __name__ == "__main__":
    main()
