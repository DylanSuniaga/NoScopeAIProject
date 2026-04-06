from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
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
from noscope_bio.pipeline import build_demo_pipeline, load_or_build_session_from_upload


st.set_page_config(page_title="NoScope-Bio", page_icon=":dart:", layout="wide")

PIPELINE_CACHE_VERSION = "2026-04-06-03"


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
    first_clicks = first_trail[first_trail["click"] > 0]
    y_max = max(float(score_df["raw_score"].max()), float(score_df["gated_score"].max()), float(threshold)) * 1.1

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,
        row_heights=[0.42, 0.26, 0.32],
        subplot_titles=("Cursor Replay Feed", "Model Output", "Server Telemetry"),
    )

    fig.add_trace(
        go.Scatter(x=first_old_trail["cursor_x"], y=first_old_trail["cursor_y"], mode="lines", name="Older Cursor Trail", line=dict(color="#7cc9ff", width=2), opacity=0.22),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=first_recent_trail["cursor_x"], y=first_recent_trail["cursor_y"], mode="lines", name="Recent Cursor Trail", line=dict(color="#ff6b35", width=3)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=first_clicks["cursor_x"],
            y=first_clicks["cursor_y"],
            mode="markers",
            name="Clicks",
            marker=dict(size=10, symbol="diamond", color="#f6c453", line=dict(width=1, color="#40330a")),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=[first_latest["cursor_x"]], y=[first_latest["cursor_y"]], mode="markers", name="Cursor", marker=dict(size=18, symbol="cross", color="#ff6b35")),
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
        click_points = frame_trail[frame_trail["click"] > 0]
        observed_scores = score_df[score_df["t_end_s"] <= current_t]
        if observed_scores.empty:
            continue
        score_row = observed_scores.iloc[-1]
        observed_network = playback_df[playback_df["t"] <= current_t]
        frames.append(
            go.Frame(
                name=name,
                data=[
                    go.Scatter(x=old_trail["cursor_x"], y=old_trail["cursor_y"]),
                    go.Scatter(x=recent_trail["cursor_x"], y=recent_trail["cursor_y"]),
                    go.Scatter(x=click_points["cursor_x"], y=click_points["cursor_y"]),
                    go.Scatter(x=[latest["cursor_x"]], y=[latest["cursor_y"]]),
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

    sampled = replay_df.iloc[:: max(1, len(replay_df) // 120)].copy()
    sampled["frame_index"] = range(len(sampled))
    fig = go.Figure()
    old_trail, recent_trail = _split_trail(sampled)
    fig.add_trace(
        go.Scatter(
            x=old_trail["cursor_x"],
            y=old_trail["cursor_y"],
            mode="lines",
            name="Older Cursor Trail",
            line=dict(color="#7cc9ff", width=2),
            opacity=0.22,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=recent_trail["cursor_x"],
            y=recent_trail["cursor_y"],
            mode="lines",
            name="Recent Cursor Trail",
            line=dict(color="#ef553b", width=3),
        )
    )
    if current_t is not None:
        current_rows = replay_df[replay_df["t"] <= current_t]
        latest = current_rows.iloc[-1] if not current_rows.empty else sampled.iloc[-1]
        click_points = current_rows[current_rows["click"] > 0].tail(10)
    else:
        latest = sampled.iloc[-1]
        click_points = replay_df[replay_df["click"] > 0].tail(10)

    dark_mode = replay_style == "FPS Overlay"
    background = "#091014" if dark_mode else "#ffffff"
    paper = "#05090c" if dark_mode else "#ffffff"
    foreground = "#e8f3f8" if dark_mode else "#111111"

    fig.add_trace(
        go.Scatter(
            x=click_points["cursor_x"],
            y=click_points["cursor_y"],
            mode="markers",
            name="Recent Clicks",
            marker=dict(size=10, symbol="diamond", color="#f6c453", line=dict(width=1, color="#40330a")),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[latest["cursor_x"]],
            y=[latest["cursor_y"]],
            mode="markers",
            name="Current Cursor",
            marker=dict(size=18, symbol="cross", color="#ff6b35"),
        )
    )
    gap = 0.035
    arm = 0.08
    cx = float(latest["cursor_x"])
    cy = float(latest["cursor_y"])
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
        title="Cursor Replay View",
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
    dark_mode = replay_style == "FPS Overlay"
    background = "#091014" if dark_mode else "#ffffff"
    paper = "#05090c" if dark_mode else "#ffffff"
    foreground = "#e8f3f8" if dark_mode else "#111111"
    first = replay_df.iloc[0]
    frame_names = [f"{t:.1f}s" for t in frame_times]

    fig = go.Figure(
        data=[
            go.Scatter(x=[first["cursor_x"]], y=[first["cursor_y"]], mode="lines", name="Older Cursor Trail", line=dict(color="#7cc9ff", width=2), opacity=0.22),
            go.Scatter(x=[first["cursor_x"]], y=[first["cursor_y"]], mode="lines", name="Recent Cursor Trail", line=dict(color="#ff6b35", width=2)),
            go.Scatter(x=[], y=[], mode="markers", name="Clicks", marker=dict(size=10, symbol="diamond", color="#f6c453", line=dict(width=1, color="#40330a"))),
            go.Scatter(x=[first["cursor_x"]], y=[first["cursor_y"]], mode="markers", name="Cursor", marker=dict(size=18, symbol="cross", color="#ff6b35")),
        ]
    )
    frames = []
    for name, current_t in zip(frame_names, frame_times):
        frame_df = replay_df[replay_df["t"] <= current_t]
        if frame_df.empty:
            continue
        latest = frame_df.iloc[-1]
        old_trail, recent_trail = _split_trail(frame_df)
        click_points = frame_df[frame_df["click"] > 0]
        frames.append(
            go.Frame(
                name=name,
                data=[
                    go.Scatter(x=old_trail["cursor_x"], y=old_trail["cursor_y"]),
                    go.Scatter(x=recent_trail["cursor_x"], y=recent_trail["cursor_y"]),
                    go.Scatter(x=click_points["cursor_x"], y=click_points["cursor_y"]),
                    go.Scatter(x=[latest["cursor_x"]], y=[latest["cursor_y"]]),
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
        title="Browser Playback Replay",
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


def main():
    st.title("NoScope-Bio")
    st.caption("Personalized behavioral anti-cheat demo")

    artifacts = get_artifacts()
    if "decision_threshold" not in artifacts:
        artifacts = normalize_artifacts(build_demo_pipeline())
    st.sidebar.header("Session Input")

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
    selected_session_id = st.sidebar.selectbox("Or choose bundled demo session", session_ids)
    replay_style = st.sidebar.selectbox("Replay Style", ["FPS Overlay", "Clean Arena"], index=0)

    if upload is not None:
        selected = load_or_build_session_from_upload(upload)
        session_df = selected["session_df"]
        result = artifacts["analyzer"](session_df)
        session_meta = selected["meta"]
    else:
        session_df = artifacts["evaluation_sessions"][selected_session_id]
        result = artifacts["analysis_cache"][selected_session_id]
        session_meta = artifacts["session_meta"].set_index("session_id").loc[selected_session_id].to_dict()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted Verdict", result["verdict"])
    c2.metric("Mode", session_meta["mode"])
    c3.metric("Player", session_meta["player_id"])
    c4.metric("Peak Score", f"{result['peak_score']:.2f}")
    if "running_verdict" in result["window_scores"].columns:
        st.caption("Each point on the timeline is computed causally from the current window and past history only. The session verdict is the last running verdict.")

    render_timeline(result["window_scores"], threshold=artifacts["decision_threshold"])

    score_df = result["window_scores"]
    start_default = max(0.0, float(score_df["t_end_s"].min()))
    end_default = float(score_df["t_end_s"].max())
    replay_range = st.slider(
        "Replay range (seconds)",
        min_value=float(session_df["t"].min()),
        max_value=float(session_df["t"].max()),
        value=(start_default, min(end_default, start_default + 12.0)),
        step=0.5,
    )
    render_replay(session_df, replay_range[0], replay_range[1], replay_style=replay_style)
    render_network_figure(session_df, replay_range[0], replay_range[1])

    st.subheader("Live Playback")
    animation_stride = st.slider(
        "Animation frame stride",
        min_value=1,
        max_value=4,
        value=2,
        step=1,
        help="Higher values reduce the number of playback frames and make the animation lighter.",
    )
    frame_duration_ms = st.select_slider(
        "Animation speed",
        options=[120, 180, 260, 360, 520],
        value=260,
        format_func=lambda value: f"{value} ms/frame",
        help="This controls the built-in browser animation speed.",
    )
    trail_seconds = st.slider(
        "Replay trail length (seconds)",
        min_value=0.8,
        max_value=6.0,
        value=2.2,
        step=0.2,
        help="Older replay points drop out as the animation advances so the FPS view stays readable.",
    )
    open_playback = st.button("Open Live Animation Studio", use_container_width=True)
    if open_playback:
        st.session_state["show_playback_studio"] = True

    if st.session_state.get("show_playback_studio", False):
        st.caption("One Play button now drives replay, model score, and server telemetry together. The replay uses a trailing window so older points fall away.")
        score_times = [
            float(t)
            for t in result["window_scores"]["t_end_s"].tolist()
            if replay_range[0] <= float(t) <= replay_range[1]
        ]
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

    st.subheader("Test Results")
    eval_metrics = artifacts["evaluation_metrics"]
    metric_cols = st.columns(len(eval_metrics))
    for col, (name, value) in zip(metric_cols, eval_metrics.items()):
        col.metric(name.replace("_", " ").title(), f"{value:.3f}")

    split_metrics_df = pd.DataFrame(artifacts["split_metrics"]).T.reset_index().rename(columns={"index": "split"})
    st.caption("These are held-out test metrics by default. The table below shows calibration-train, validation, and test splits.")
    st.dataframe(split_metrics_df, use_container_width=True, hide_index=True)

    with st.expander("Feature Schema"):
        st.write({"behavioral_features": DEMO_FEATURE_COLUMNS, "server_features": SERVER_TELEMETRY_COLUMNS})


if __name__ == "__main__":
    main()
