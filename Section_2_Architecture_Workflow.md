# Section 2: Overall Architecture, Workflow, and Main Functional Components

## System Overview

`NoScope-Bio` is a personalized behavioral anti-cheat prototype built around **target-agnostic cursor motor modeling**.

Instead of assuming we know where an enemy target was, the system uses only:

- cursor motion
- clicks
- timestamps
- server-side telemetry

The goal is to learn each player's normal motor signature and then detect statistically abnormal shifts in that signature over time.

The system is organized as a six-stage pipeline:

1. session telemetry input
2. feature preprocessing and online segmentation
3. behavioral fingerprint encoding
4. personalized baseline comparison
5. online change detection
6. causal-style validation and moderator-facing output

```mermaid
flowchart LR
    A["Session Telemetry CSV"] --> B["Feature Preprocessing"]
    B --> C["Online Segmentation / Windowing"]
    C --> D["Causal Sequence Encoder"]
    D --> E["Player Baseline Comparison"]
    E --> F["Online Change Detection"]
    F --> G["Causal Plausibility Gate"]
    G --> H["Moderator Output + Timeline + Metrics"]
```

## Main Functional Components

### 1. Synthetic Session Generator

This component produces baseline and evaluation sessions for multiple synthetic players.

Each player has a stable individual style across:

- cursor delta and cursor velocity
- pause and burst timing
- correction style
- micro-jitter amplitude
- curvature tendencies
- click cadence
- server-visible network conditions

The generator is designed to simulate more realistic human motor behavior using:

- smooth overlapping submovements
- signal-dependent noise
- session-level tempo variation
- pauses and hesitations
- corrective micro-motions

It also injects controlled session types:

- `clean`
- `aimbot_like_locking`
- `triggerbot_like_clicking`
- `macro_consistency`
- `high_ping`
- `sensitivity_change`
- `patch_shift`

### 2. Feature Preprocessing And Online Segmentation

The raw cursor stream is converted into a causal windowed sequence representation so the model compares recent behavioral intervals rather than full-session summaries.

Online segmentation is based on movement structure such as:

- velocity threshold crossings
- pauses
- jerk minima
- direction-change boundaries

### 3. Fingerprint Encoder

The main ML component is a causal temporal fingerprint model.

It is trained on clean telemetry windows to predict player identity. The hidden representation of that model becomes the player's behavioral embedding.

This is the main non-trivial ML contribution because it performs:

- sequence representation learning
- player-specific behavioral fingerprinting
- personalized anomaly detection support

### 4. Personalized Baseline Comparison

For each player, the system stores baseline embedding statistics and clean feature distributions. Each new session window is compared against that player's historical normal behavior.

### 5. Online Shift Detector

During evaluation, each new session is processed in causal time order. For each window, the system measures:

- embedding drift from baseline
- movement regularity shifts
- direction-entropy changes
- curvature and jerk changes
- click-motion coupling changes

These scores are streamed into an online change detector to identify abrupt sustained transitions.

### 6. Causal-Style Validation Gate

Before labeling a session as suspicious, the system checks whether the observed shift is more plausibly explained by confounders such as:

- ping spikes
- jitter increases
- packet loss
- command-age inflation
- sensitivity changes
- environment or patch shifts

### 7. Review UI

The Streamlit UI supports:

- selecting or uploading a session
- viewing a suspicion timeline
- viewing a cursor-only replay
- running a live timeline animation
- viewing server telemetry over time
- viewing explanation text for why the session was flagged or suppressed

## Input To The Algorithm / Model

### Input Format

The main input is a telemetry session CSV. Each row represents one causal time-step in the session.

### Input Examples

Representative cursor and behavioral fields:

- `cursor_x`, `cursor_y`
- `dx`, `dy`
- `speed`
- `acceleration`
- `jerk`
- `heading`
- `heading_change`
- `angular_velocity`
- `curvature`
- `burst_id`
- `burst_progress`
- `pause_ms`
- `click`
- `time_since_click_ms`
- `path_tortuosity`
- `direction_entropy_short`
- `roughness_score`

Representative server and environment fields:

- `ping_ms`
- `jitter_ms`
- `packet_loss_pct`
- `command_age_ms`
- `packet_interarrival_ms`
- `input_burstiness`
- `server_correction_magnitude`
- `tick_desync_ms`
- `sensitivity`

### Example Input Row

```text
session_id=P07_locking_01, player_id=P07, tick=418, t=20.90,
cursor_x=0.215, cursor_y=-0.041, dx=0.071, dy=-0.012,
speed=0.188, acceleration=0.041, jerk=0.013,
heading_change=0.022, curvature=0.018, pause_ms=0,
click=1, time_since_click_ms=142,
ping_ms=31.7, jitter_ms=2.3, packet_loss_pct=0.08, command_age_ms=19.5
```

## Intermediate Processing Steps

### Step 1. Session Validation

The CSV is checked for required columns, session metadata, and causal input integrity.

### Step 2. Feature Scaling

Behavioral and network features are normalized using clean baseline sessions only, so suspicious sessions are evaluated relative to each player's historical normal behavior.

### Step 3. Online Segmentation And Windowing

The time series is split into overlapping trailing windows. Each window contains only current and past information, never future values.

### Step 4. Fingerprint Embedding

Each window is passed through the fingerprint encoder to produce an embedding vector representing the player's local behavioral identity.

### Step 5. Baseline Comparison

The current embedding and summary features are compared against that player's stored baseline statistics.

### Step 6. Online Change Detection

The system computes rolling suspicion scores and feeds them into an online change detector to find abrupt sustained behavioral shifts.

### Step 7. Causal Plausibility Gate

If the same time window also shows server-side instability such as elevated jitter, packet loss, or command age, the suspicion score is discounted unless the cursor behavior still looks abnormally low-entropy or over-regular.

## Final Output Of The Algorithm / Model

### Data Output

For each analyzed session, the system produces:

- session verdict: `Suspicious` or `Likely Legit`
- peak suspicion score
- change-point timestamp
- per-window suspicion timeline
- top shifted features vs baseline
- causal gate notes

### Measurement Output

The project exports split-aware evaluation metrics:

- accuracy
- precision
- recall
- false positive rate
- false negative rate

Evaluation is separated into:

- calibration-train sessions for fitting the cheat scorer
- validation sessions for threshold selection
- held-out test sessions for final reporting in the UI

Latest held-out test snapshot from the current synthetic evaluation:

- accuracy: `0.986`
- precision: `0.968`
- recall: `1.000`
- false positive rate: `0.025`
- false negative rate: `0.000`

Important note:

- these values are from the current synthetic evaluation environment and should be presented as demo results, not as proof of real-world deployment readiness

### Moderator-Facing Output

The UI is designed to show:

- when the behavioral drift began
- what changed in the cursor dynamics
- whether network or environment confounders were present
- whether the final decision should be escalation or suppression

## Where The ML Enters Non-Trivially

The ML contribution is not a simple thresholding system.

1. A causal sequence model learns a compact behavioral embedding from raw cursor-motion windows.
2. The system learns player-specific baseline distributions in embedding space.
3. A second learned layer maps anomaly evidence and confounders into a calibrated cheat probability.

This means the project combines:

- sequence representation learning
- personalized anomaly detection
- probability calibration

while still keeping the online change detector and confounder gate interpretable.

## Deliverable Language You Can Use In Class

> Our model takes cursor motion, click timing, and server-side session signals as input, learns an individualized cursor-motor fingerprint from clean sessions, monitors new sessions for abrupt identity drift using only causal windows, and uses a confounder-aware validation gate to distinguish suspicious automation-like behavior from plausible external causes such as lag or sensitivity changes.
