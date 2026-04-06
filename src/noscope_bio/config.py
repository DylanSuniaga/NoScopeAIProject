from __future__ import annotations

from dataclasses import dataclass


DEMO_FEATURE_COLUMNS = [
    "dx",
    "dy",
    "speed",
    "acceleration",
    "jerk",
    "heading_sin",
    "heading_cos",
    "heading_change",
    "angular_velocity",
    "curvature",
    "motion_active",
    "pause_ms",
    "burst_progress",
    "burst_duration_ms",
    "local_straightness",
    "direction_entropy_short",
    "roughness_score",
    "speed_autocorr_short",
    "click",
    "time_since_click_ms",
    "click_motion_coupling",
    "last_click_interval_ms",
    "last_stabilization_delay_ms",
    "ping_ms",
    "jitter_ms",
    "packet_loss_pct",
    "command_age_ms",
    "packet_interarrival_ms",
    "input_burstiness",
    "server_correction_magnitude",
    "tick_desync_ms",
    "sensitivity",
]

SERVER_TELEMETRY_COLUMNS = [
    "ping_ms",
    "jitter_ms",
    "packet_loss_pct",
    "command_age_ms",
    "packet_interarrival_ms",
    "input_burstiness",
    "server_correction_magnitude",
    "tick_desync_ms",
]

FORBIDDEN_MODEL_COLUMNS = {
    "session_id",
    "player_id",
    "mode",
    "tick",
    "t",
    "label_cheat",
    "cheat_active",
    "confounder_active",
    "change_point_t",
    "lock_event_active",
    "pending_click_active",
    "window_cheat_fraction",
    "window_confounder_fraction",
}

SUMMARY_FEATURES = {
    "speed": ["mean", "std"],
    "acceleration": ["mean", "std"],
    "jerk": ["mean", "std"],
    "heading_change": ["mean", "std"],
    "angular_velocity": ["mean", "std"],
    "curvature": ["mean", "std"],
    "motion_active": ["mean"],
    "pause_ms": ["mean", "max"],
    "burst_progress": ["mean", "std"],
    "burst_duration_ms": ["mean", "std"],
    "local_straightness": ["mean", "std", "max"],
    "direction_entropy_short": ["mean", "std"],
    "roughness_score": ["mean", "std"],
    "speed_autocorr_short": ["mean", "std"],
    "click": ["mean"],
    "time_since_click_ms": ["mean", "std"],
    "click_motion_coupling": ["mean", "max"],
    "last_click_interval_ms": ["mean", "std"],
    "last_stabilization_delay_ms": ["mean", "std"],
    "ping_ms": ["mean", "std"],
    "jitter_ms": ["mean", "std", "max"],
    "packet_loss_pct": ["mean", "std", "max"],
    "command_age_ms": ["mean", "std", "max"],
    "packet_interarrival_ms": ["mean", "std"],
    "input_burstiness": ["mean", "std"],
    "server_correction_magnitude": ["mean", "max"],
    "tick_desync_ms": ["mean", "std", "max"],
    "sensitivity": ["mean", "std"],
}

CHEAT_MODES = {"aimbot", "triggerbot", "macro_consistency"}
CONFOUNDER_MODES = {"high_ping", "sensitivity_change", "patch_shift"}


@dataclass(frozen=True)
class SimulationConfig:
    ticks: int = 480
    dt: float = 0.05
    cheat_start_fraction: float = 0.55
    motion_threshold: float = 0.018
    straightness_window: int = 10
    entropy_window: int = 12
    autocorr_window: int = 10
