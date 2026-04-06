from __future__ import annotations

from dataclasses import dataclass


DEMO_FEATURE_COLUMNS = [
    "view_yaw",
    "view_pitch",
    "yaw_delta",
    "pitch_delta",
    "angular_speed",
    "angular_energy",
    "angular_acceleration",
    "angular_jerk",
    "aim_vector_x",
    "aim_vector_y",
    "aim_vector_z",
    "heading_change",
    "angular_velocity",
    "curvature",
    "motion_active",
    "pause_ms",
    "burst_progress",
    "burst_duration_ms",
    "view_straightness",
    "stability_score",
    "yaw_reversal",
    "pitch_reversal",
    "direction_entropy_short",
    "micro_correction_score",
    "angular_speed_autocorr_short",
    "fire_input",
    "time_since_fire_ms",
    "fire_motion_coupling",
    "last_fire_interval_ms",
    "last_stabilization_to_fire_ms",
    "flick_event",
    "flick_magnitude",
    "time_since_flick_ms",
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
    "view_yaw": ["mean", "std"],
    "view_pitch": ["mean", "std"],
    "angular_speed": ["mean", "std"],
    "angular_energy": ["mean", "std"],
    "angular_acceleration": ["mean", "std"],
    "angular_jerk": ["mean", "std"],
    "heading_change": ["mean", "std"],
    "angular_velocity": ["mean", "std"],
    "curvature": ["mean", "std"],
    "motion_active": ["mean"],
    "pause_ms": ["mean", "max"],
    "burst_progress": ["mean", "std"],
    "burst_duration_ms": ["mean", "std"],
    "view_straightness": ["mean", "std", "max"],
    "stability_score": ["mean", "std", "max"],
    "yaw_reversal": ["mean"],
    "pitch_reversal": ["mean"],
    "direction_entropy_short": ["mean", "std"],
    "micro_correction_score": ["mean", "std"],
    "angular_speed_autocorr_short": ["mean", "std"],
    "fire_input": ["mean"],
    "time_since_fire_ms": ["mean", "std"],
    "fire_motion_coupling": ["mean", "max"],
    "last_fire_interval_ms": ["mean", "std"],
    "last_stabilization_to_fire_ms": ["mean", "std"],
    "flick_event": ["mean"],
    "flick_magnitude": ["mean", "std", "max"],
    "time_since_flick_ms": ["mean", "std"],
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
