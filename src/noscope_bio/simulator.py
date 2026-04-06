from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CHEAT_MODES, CONFOUNDER_MODES, SimulationConfig


@dataclass(frozen=True)
class PlayerProfile:
    player_id: str
    burst_amp_mean: float
    burst_amp_std: float
    burst_duration_mean_ms: float
    burst_duration_std_ms: float
    pause_mean_ms: float
    pause_std_ms: float
    correction_prob: float
    correction_scale: float
    micro_jitter: float
    noise_memory: float
    angular_noise: float
    direction_persistence: float
    click_latency_mean_ms: float
    click_latency_std_ms: float
    click_gap_mean_ms: float
    click_gap_std_ms: float
    stabilization_bias: float
    sensitivity: float
    base_ping_ms: float
    base_jitter_ms: float
    base_packet_loss_pct: float
    base_command_rate_hz: float


def _clip(vec: np.ndarray) -> np.ndarray:
    return np.clip(vec, -1.0, 1.0)


def _angle_diff(a: float, b: float) -> float:
    return float(np.arctan2(np.sin(a - b), np.cos(a - b)))


def _minimum_jerk_derivative(u: float) -> float:
    if u <= 0.0 or u >= 1.0:
        return 0.0
    return float(30.0 * u * u - 60.0 * u * u * u + 30.0 * u * u * u * u)


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


def generate_player_profiles(num_players: int = 14, seed: int = 7) -> list[PlayerProfile]:
    rng = np.random.default_rng(seed)
    profiles = []
    for idx in range(num_players):
        profiles.append(
            PlayerProfile(
                player_id=f"P{idx + 1:02d}",
                burst_amp_mean=float(rng.uniform(0.06, 0.18)),
                burst_amp_std=float(rng.uniform(0.015, 0.045)),
                burst_duration_mean_ms=float(rng.uniform(220.0, 520.0)),
                burst_duration_std_ms=float(rng.uniform(60.0, 140.0)),
                pause_mean_ms=float(rng.uniform(80.0, 260.0)),
                pause_std_ms=float(rng.uniform(24.0, 90.0)),
                correction_prob=float(rng.uniform(0.25, 0.70)),
                correction_scale=float(rng.uniform(0.12, 0.48)),
                micro_jitter=float(rng.uniform(0.004, 0.018)),
                noise_memory=float(rng.uniform(0.45, 0.88)),
                angular_noise=float(rng.uniform(0.18, 0.75)),
                direction_persistence=float(rng.uniform(0.45, 0.88)),
                click_latency_mean_ms=float(rng.uniform(120.0, 260.0)),
                click_latency_std_ms=float(rng.uniform(24.0, 80.0)),
                click_gap_mean_ms=float(rng.uniform(190.0, 420.0)),
                click_gap_std_ms=float(rng.uniform(25.0, 90.0)),
                stabilization_bias=float(rng.uniform(0.45, 1.15)),
                sensitivity=float(rng.uniform(0.85, 1.25)),
                base_ping_ms=float(rng.uniform(18.0, 55.0)),
                base_jitter_ms=float(rng.uniform(0.8, 4.8)),
                base_packet_loss_pct=float(rng.uniform(0.02, 0.40)),
                base_command_rate_hz=float(rng.uniform(18.0, 32.0)),
            )
        )
    return profiles


def simulate_session(
    profile: PlayerProfile,
    session_id: str,
    mode: str = "clean",
    seed: int = 0,
    config: SimulationConfig | None = None,
) -> pd.DataFrame:
    config = config or SimulationConfig()
    rng = np.random.default_rng(seed)
    cheat_start_tick = int(config.ticks * config.cheat_start_fraction)

    session_amp_scale = float(np.clip(rng.normal(1.0, 0.16), 0.70, 1.38))
    session_tempo_scale = float(np.clip(rng.normal(1.0, 0.18), 0.72, 1.38))
    session_jitter_scale = float(np.clip(rng.normal(1.0, 0.18), 0.65, 1.45))
    session_pause_scale = float(np.clip(rng.normal(1.0, 0.20), 0.60, 1.55))
    session_direction_scale = float(np.clip(rng.normal(1.0, 0.22), 0.65, 1.45))
    session_network_load = float(np.clip(abs(rng.normal(0.0, 0.34)), 0.0, 1.35))
    session_focus = float(np.clip(rng.normal(1.0, 0.14), 0.70, 1.35))
    cheat_intensity = float(np.clip(rng.normal(1.0, 0.16), 0.72, 1.35))
    confounder_intensity = float(np.clip(rng.normal(1.0, 0.18), 0.75, 1.45))

    cursor = rng.uniform(-0.25, 0.25, size=2)
    current_heading = float(rng.uniform(-np.pi, np.pi))
    prev_delta = np.zeros(2, dtype=np.float64)
    prev_speed = 0.0
    prev_accel = 0.0
    prev_heading = current_heading
    noise_state = np.zeros(2, dtype=np.float64)

    active_moves: list[dict] = []
    pending_moves: list[dict] = []
    next_primary_tick = 0
    burst_counter = 0
    current_primary_id = -1
    pause_ms = 0.0
    click_motion_coupling = 0.0
    time_since_click_ms = 10_000.0
    last_click_tick = -10_000
    last_click_interval_ms = profile.click_gap_mean_ms
    last_stabilization_delay_ms = profile.click_latency_mean_ms
    pending_click_tick: int | None = None
    pending_stabilization_tick: int | None = None
    lock_event_until = -1
    time_since_flick_ms = 10_000.0

    position_history: list[np.ndarray] = [cursor.copy()]
    heading_history: list[float] = []
    speed_history: list[float] = []
    heading_change_history: list[float] = []
    rows = []

    def spawn_primary_move(tick: int, active_cheat: bool, sensitivity: float) -> None:
        nonlocal burst_counter, current_heading, next_primary_tick, current_primary_id, lock_event_until

        burst_counter += 1
        is_macro = mode == "macro_consistency" and active_cheat
        is_lock = mode == "aimbot" and active_cheat and rng.random() < (0.30 + 0.18 * cheat_intensity)

        if is_macro:
            macro_period = max(6, int((profile.burst_duration_mean_ms * 0.60) / (config.dt * 1000.0)))
            macro_index = (tick // macro_period) % 4
            angle = float((macro_index * (np.pi / 2.0)) + rng.normal(0.0, 0.05))
            amplitude = float(np.clip(profile.burst_amp_mean * (1.05 + 0.22 * cheat_intensity), 0.05, 0.26))
            duration_ticks = max(4, int((profile.burst_duration_mean_ms * 0.78) / (config.dt * 1000.0)))
            pause_ticks = max(2, int((profile.pause_mean_ms * 0.55) / (config.dt * 1000.0)))
            correction_prob = 0.06
        elif is_lock:
            angle = float(current_heading + rng.normal(0.0, 0.018))
            amplitude = float(np.clip(profile.burst_amp_mean * (1.20 + 0.30 * cheat_intensity), 0.05, 0.30))
            duration_ticks = max(4, int((profile.burst_duration_mean_ms * (0.62 + 0.05 * rng.normal())) / (config.dt * 1000.0)))
            pause_ticks = max(2, int((profile.pause_mean_ms * 0.42) / (config.dt * 1000.0)))
            correction_prob = 0.03
            lock_event_until = tick + duration_ticks
        else:
            angle_jitter = profile.angular_noise * session_direction_scale
            if rng.random() < 0.18:
                current_heading = float(rng.uniform(-np.pi, np.pi))
            angle = float(current_heading + rng.normal(0.0, angle_jitter))
            amplitude = float(max(0.03, rng.normal(profile.burst_amp_mean, profile.burst_amp_std) * session_amp_scale * session_focus))
            duration_ms = max(120.0, rng.normal(profile.burst_duration_mean_ms, profile.burst_duration_std_ms) / session_tempo_scale)
            duration_ticks = max(4, int(duration_ms / (config.dt * 1000.0)))
            pause_ms_local = max(35.0, rng.normal(profile.pause_mean_ms, profile.pause_std_ms) * session_pause_scale)
            pause_ticks = max(1, int(pause_ms_local / (config.dt * 1000.0)))
            correction_prob = profile.correction_prob

        vector = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64) * amplitude * sensitivity
        primary_move = {
            "id": burst_counter,
            "start_tick": tick,
            "duration_ticks": duration_ticks,
            "vector": vector,
            "kind": "primary",
            "lock_like": int(is_lock),
        }
        active_moves.append(primary_move)
        current_primary_id = burst_counter
        current_heading = angle * profile.direction_persistence + current_heading * (1.0 - profile.direction_persistence)
        next_primary_tick = tick + duration_ticks + pause_ticks

        if rng.random() < correction_prob:
            correction_delay = max(2, duration_ticks // 2)
            rotate = rng.normal(0.0, 0.35)
            correction_vec = -vector * (profile.correction_scale * session_focus) + np.array(
                [np.cos(angle + rotate), np.sin(angle + rotate)],
                dtype=np.float64,
            ) * amplitude * 0.18
            pending_moves.append(
                {
                    "id": burst_counter,
                    "start_tick": tick + correction_delay,
                    "duration_ticks": max(3, duration_ticks // 2),
                    "vector": correction_vec,
                    "kind": "correction",
                    "lock_like": 0,
                }
            )

    for tick in range(config.ticks):
        t = tick * config.dt
        active = tick >= cheat_start_tick

        sensitivity = profile.sensitivity * float(np.clip(rng.normal(1.0, 0.03), 0.92, 1.08))
        ping_ms = profile.base_ping_ms + 4.0 * session_network_load + rng.normal(0.0, 2.8)
        jitter_ms = max(0.15, profile.base_jitter_ms * (1.0 + 0.18 * session_network_load) + abs(rng.normal(0.0, 0.85)))
        packet_loss_pct = max(0.0, profile.base_packet_loss_pct + 0.05 * session_network_load + abs(rng.normal(0.0, 0.05)))
        command_rate_hz = max(8.0, profile.base_command_rate_hz + rng.normal(0.0, 0.8) - session_network_load * 0.8)
        patch_multiplier = 1.0

        if mode == "high_ping" and active:
            ping_ms += (55.0 + 24.0 * np.sin(t * 1.7)) * confounder_intensity
            jitter_ms += (7.0 + 4.0 * abs(np.sin(t * 2.5))) * confounder_intensity
            packet_loss_pct += (0.7 + 0.9 * abs(np.sin(t * 2.9))) * confounder_intensity
            command_rate_hz = max(8.0, command_rate_hz - 3.5 * confounder_intensity + rng.normal(0.0, 0.8))
        if mode == "sensitivity_change" and active:
            sensitivity *= 1.16 + 0.18 * confounder_intensity
        if mode == "patch_shift" and active:
            patch_multiplier = 1.06 + 0.10 * confounder_intensity
            jitter_ms += 0.6 + 0.7 * confounder_intensity
            command_rate_hz += 0.5

        for move in pending_moves[:]:
            if move["start_tick"] <= tick:
                active_moves.append(move)
                pending_moves.remove(move)

        if tick >= next_primary_tick:
            spawn_primary_move(tick, active, sensitivity * patch_multiplier)

        total_delta = np.zeros(2, dtype=np.float64)
        completed_primary_lock = False
        completed_primary = False
        current_primary_progress = 0.0
        current_burst_duration_ms = 0.0

        for move in active_moves[:]:
            elapsed = tick - move["start_tick"]
            if elapsed >= move["duration_ticks"]:
                active_moves.remove(move)
                if move["kind"] == "primary" and move["id"] == current_primary_id:
                    completed_primary = True
                    completed_primary_lock = bool(move["lock_like"])
                continue

            u = (elapsed + 0.5) / max(move["duration_ticks"], 1)
            contribution = move["vector"] * (_minimum_jerk_derivative(u) / max(move["duration_ticks"], 1))
            total_delta += contribution

            if move["kind"] == "primary" and move["id"] == current_primary_id:
                current_primary_progress = float(np.clip(u, 0.0, 1.0))
                current_burst_duration_ms = float(move["duration_ticks"] * config.dt * 1000.0)

        if completed_primary and pending_click_tick is None:
            pending_stabilization_tick = tick
            base_latency = max(25.0, rng.normal(profile.click_latency_mean_ms, profile.click_latency_std_ms))
            base_gap = max(80.0, rng.normal(profile.click_gap_mean_ms, profile.click_gap_std_ms))
            fire_prob = 0.66
            if mode == "triggerbot" and active:
                base_latency = 42.0 + abs(rng.normal(0.0, 6.0))
                base_gap *= 0.95
                fire_prob = 0.88
            elif mode == "macro_consistency" and active:
                base_latency = 68.0 + abs(rng.normal(0.0, 4.0))
                base_gap = 175.0 + abs(rng.normal(0.0, 8.0))
                fire_prob = 0.82
            elif mode == "aimbot" and active and completed_primary_lock:
                base_latency = 56.0 + abs(rng.normal(0.0, 8.0))
                fire_prob = 0.86

            delay_ticks = max(1, int(base_latency / (config.dt * 1000.0)))
            min_gap_ticks = max(2, int(base_gap / (config.dt * 1000.0)))
            if tick - last_click_tick >= min_gap_ticks and rng.random() < fire_prob:
                pending_click_tick = tick + delay_ticks

        lock_like_active = int(mode == "aimbot" and active and tick <= lock_event_until)

        speed_scale = 1.0 + min(0.5, max(0.0, ping_ms - profile.base_ping_ms) / 160.0)
        if mode == "macro_consistency" and active:
            speed_scale *= 0.92
        jitter_scale = profile.micro_jitter * session_jitter_scale * speed_scale
        if lock_like_active:
            jitter_scale *= max(0.18, 0.34 - 0.10 * cheat_intensity)
        if mode == "macro_consistency" and active:
            jitter_scale *= 0.45
        if mode == "high_ping" and active:
            jitter_scale *= 1.25 + 0.18 * confounder_intensity

        noise_state = (
            profile.noise_memory * noise_state
            + rng.normal(0.0, jitter_scale * (1.0 + 0.40 * np.linalg.norm(total_delta)), size=2)
        )
        delta = (total_delta + noise_state) * patch_multiplier
        if mode == "macro_consistency" and active:
            periodic_angle = float((tick % 12) * (np.pi / 6.0))
            delta = 0.82 * delta + np.array([np.cos(periodic_angle), np.sin(periodic_angle)]) * 0.006
        cursor = _clip(cursor + delta)

        speed = float(np.linalg.norm(delta))
        acceleration = float(speed - prev_speed)
        jerk = float(acceleration - prev_accel)
        heading = prev_heading if speed < 1e-6 else float(np.arctan2(delta[1], delta[0]))
        heading_change = float(abs(_angle_diff(heading, prev_heading)))
        angular_velocity = float(heading_change / config.dt)
        curvature = float(heading_change / max(speed, 1e-4))
        angular_energy = float(speed * speed)
        motion_active = float(speed > config.motion_threshold)
        pause_ms = 0.0 if motion_active else pause_ms + config.dt * 1000.0
        yaw_reversal = int(abs(delta[0]) > 0.01 and abs(prev_delta[0]) > 0.01 and np.sign(delta[0]) != np.sign(prev_delta[0]))
        pitch_reversal = int(abs(delta[1]) > 0.01 and abs(prev_delta[1]) > 0.01 and np.sign(delta[1]) != np.sign(prev_delta[1]))

        position_history.append(cursor.copy())
        if len(position_history) > config.straightness_window:
            position_history.pop(0)
        if motion_active:
            heading_history.append(heading)
            if len(heading_history) > config.entropy_window:
                heading_history.pop(0)
        speed_history.append(speed)
        if len(speed_history) > config.autocorr_window:
            speed_history.pop(0)
        heading_change_history.append(heading_change)
        if len(heading_change_history) > config.entropy_window:
            heading_change_history.pop(0)

        local_straightness = _local_straightness(position_history)
        direction_entropy_short = _normalized_entropy(heading_history)
        roughness_score = float(
            np.std(heading_change_history) + 0.35 * np.std(np.diff(speed_history)) if len(speed_history) > 2 else 0.1
        )
        speed_autocorr_short = _lag1_autocorr(speed_history)
        stability_score = float(
            np.clip(local_straightness * (1.0 - min(speed / 0.09, 1.0)) * (1.0 - min(roughness_score / 1.2, 1.0)), 0.0, 1.0)
        )
        yaw_angle = float(cursor[0] * np.pi)
        pitch_angle = float(cursor[1] * (np.pi / 3.0))
        aim_vector_x = float(np.cos(pitch_angle) * np.cos(yaw_angle))
        aim_vector_y = float(np.sin(pitch_angle))
        aim_vector_z = float(np.cos(pitch_angle) * np.sin(yaw_angle))

        recent_peak_speed = float(max(speed_history) if speed_history else speed)
        stabilization_signal = max(0.0, recent_peak_speed - speed) * (0.65 + 0.35 * local_straightness) * (1.35 - direction_entropy_short)
        click_motion_coupling *= 0.88
        click = 0
        if pending_click_tick is not None and tick >= pending_click_tick:
            click = 1
            pending_click_tick = None
            interval_ms = max(0.0, (tick - last_click_tick) * config.dt * 1000.0)
            last_click_interval_ms = interval_ms if last_click_tick > 0 else profile.click_gap_mean_ms
            if pending_stabilization_tick is not None:
                last_stabilization_delay_ms = max(0.0, (tick - pending_stabilization_tick) * config.dt * 1000.0)
            click_motion_coupling = float(np.clip(stabilization_signal * 8.0 + profile.stabilization_bias, 0.0, 6.0))
            last_click_tick = tick
            time_since_click_ms = 0.0
            pending_stabilization_tick = None
        else:
            time_since_click_ms += config.dt * 1000.0

        if mode == "triggerbot" and active:
            click_motion_coupling = float(min(6.0, click_motion_coupling * 1.08 + 0.10 * click))
        if mode == "aimbot" and active and lock_like_active:
            click_motion_coupling = float(min(6.0, click_motion_coupling * 1.03 + 0.02))

        flick_magnitude = float(heading_change * speed * 24.0)
        flick_event = int(speed > 0.055 and heading_change > 0.32)
        if flick_event:
            time_since_flick_ms = 0.0
        else:
            time_since_flick_ms += config.dt * 1000.0
        packet_interarrival_ms = max(4.0, 1000.0 / max(command_rate_hz, 1.0) + rng.normal(0.0, 0.35 + jitter_ms * 0.12))
        input_burstiness = max(0.0, abs(acceleration) * 1.8 + abs(jerk) * 0.95 + 0.25 * click + abs(rng.normal(0.0, 0.04)))
        server_correction_magnitude = max(
            0.0,
            abs(acceleration) * 1.3
            + max(0.0, 1.0 - local_straightness) * 0.35
            + max(0.0, jitter_ms - profile.base_jitter_ms) * 0.08
            + packet_loss_pct * 0.55
            + abs(rng.normal(0.0, 0.05)),
        )
        tick_desync_ms = max(0.0, 0.35 * jitter_ms + 2.1 * packet_loss_pct + abs(rng.normal(0.0, 0.5)))
        command_age_ms = max(
            1.0,
            ping_ms * 0.48 + jitter_ms * 1.65 + packet_loss_pct * 8.5 + abs(rng.normal(0.0, 2.0)),
        )

        if lock_like_active:
            input_burstiness *= max(0.25, 0.62 - 0.12 * cheat_intensity)
            server_correction_magnitude *= max(0.45, 0.78 - 0.10 * cheat_intensity)
        if mode == "macro_consistency" and active:
            input_burstiness *= 0.55
            tick_desync_ms *= 0.75
        if mode == "patch_shift" and active:
            server_correction_magnitude += 0.18 + 0.15 * confounder_intensity
            tick_desync_ms += 0.55 + 0.75 * confounder_intensity

        cheat_active = int(mode in CHEAT_MODES and active)
        confounder_active = int(mode in CONFOUNDER_MODES and active)

        rows.append(
            {
                "session_id": session_id,
                "player_id": profile.player_id,
                "mode": mode,
                "tick": tick,
                "t": t,
                "view_yaw": float(cursor[0]),
                "view_pitch": float(cursor[1]),
                "yaw_delta": float(delta[0]),
                "pitch_delta": float(delta[1]),
                "angular_speed": speed,
                "angular_energy": angular_energy,
                "angular_acceleration": acceleration,
                "angular_jerk": jerk,
                "aim_vector_x": aim_vector_x,
                "aim_vector_y": aim_vector_y,
                "aim_vector_z": aim_vector_z,
                "heading_change": heading_change,
                "angular_velocity": angular_velocity,
                "curvature": curvature,
                "motion_active": motion_active,
                "pause_ms": float(pause_ms),
                "burst_progress": current_primary_progress,
                "burst_duration_ms": current_burst_duration_ms,
                "view_straightness": local_straightness,
                "stability_score": stability_score,
                "yaw_reversal": yaw_reversal,
                "pitch_reversal": pitch_reversal,
                "direction_entropy_short": direction_entropy_short,
                "micro_correction_score": roughness_score,
                "angular_speed_autocorr_short": speed_autocorr_short,
                "fire_input": click,
                "time_since_fire_ms": float(time_since_click_ms),
                "fire_motion_coupling": float(click_motion_coupling),
                "last_fire_interval_ms": float(last_click_interval_ms),
                "last_stabilization_to_fire_ms": float(last_stabilization_delay_ms),
                "flick_event": flick_event,
                "flick_magnitude": flick_magnitude,
                "time_since_flick_ms": float(time_since_flick_ms),
                "ping_ms": float(ping_ms),
                "jitter_ms": float(jitter_ms),
                "packet_loss_pct": float(packet_loss_pct),
                "command_age_ms": float(command_age_ms),
                "packet_interarrival_ms": float(packet_interarrival_ms),
                "input_burstiness": float(input_burstiness),
                "server_correction_magnitude": float(server_correction_magnitude),
                "tick_desync_ms": float(tick_desync_ms),
                "sensitivity": float(sensitivity),
                "cheat_active": cheat_active,
                "confounder_active": confounder_active,
                "label_cheat": int(mode in CHEAT_MODES),
                "change_point_t": cheat_start_tick * config.dt,
                "lock_event_active": lock_like_active,
                "pending_click_active": int(pending_click_tick is not None),
            }
        )

        prev_delta = delta
        prev_speed = speed
        prev_accel = acceleration
        prev_heading = heading

    return pd.DataFrame(rows)
