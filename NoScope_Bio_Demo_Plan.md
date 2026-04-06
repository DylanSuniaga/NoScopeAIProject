# NoScope-Bio Demo Plan

## Project Direction

The strongest version of this project is a polished **behavioral anti-cheat lab** built around **target-agnostic cursor motor signatures**.

That means the demo does **not** assume we know where the enemy was on screen. Instead, it asks:

> Can we detect suspicious shifts in a player's cursor-motion behavior using only cursor movement, clicks, timestamps, and server-side telemetry?

With the available time, the best path is:

- synthetic telemetry with more realistic motor structure
- a believable replay UI driven by cursor-only data
- a non-trivial ML fingerprint model
- online change detection
- strong explainability

## Research-Grounded Framing

The core modeling change is important:

- human cursor motion should **not** be treated as pure Brownian motion end-to-end
- intentional movement is better modeled as **smooth submovements**
- local residual noise is stochastic, but often **structured**
- motor variability is **signal-dependent**
- suspicious automation should look less like "perfect aim at a known target" and more like **overly regular, low-entropy, low-correction cursor behavior**

So the project should be framed around:

- smooth bursts
- pauses and hesitations
- corrective micro-motions
- curvature and direction changes
- timing consistency
- network confounders

## Core Decisions

### 1. Do not use raw gameplay video as the main input

Recovering true player telemetry from random FPS clips with OpenCV in two days is too risky and too lossy. Video can still be a nice supporting visual, but it should not be the scientific core of the project.

Recommended approach:

- primary input: synthetic session telemetry CSVs
- optional visual layer: generated replay synced to telemetry
- demo interaction: upload a session package and watch the system analyze it

This gives us:

- controlled data generation
- ground-truth labels
- clean evaluation splits
- believable class-demo outputs

### 2. Use per-player behavioral baselines

This still matches the proposal best.

Pros:

- aligns with personalized behavioral biometrics
- reduces false positives from naturally skilled players
- makes explanations much stronger
- separates identity drift from cheat suspicion

Cons:

- requires clean baseline sessions per player
- introduces a cold-start issue
- genuine improvement can still look anomalous

Best demo fix:

- use multiple clean baseline sessions per player
- freeze those baselines for the demo
- add a confounder gate for ping, jitter, sensitivity changes, and patch-style environment shifts

## Where The ML Lives

This project still contains clear, non-trivial ML.

### 1. Behavioral Fingerprint Encoder

The first ML model takes a **causal trailing window** of cursor telemetry and learns a compact embedding of that player's motor behavior.

Example inputs per window:

- `dx`, `dy`
- speed
- acceleration
- jerk
- heading change
- angular velocity
- curvature
- pause timing
- click timing
- server telemetry such as ping and jitter

Best practical sequence models:

- small TCN
- small GRU
- small 1D temporal CNN

The training task is: given a clean telemetry window, predict which player generated it. The hidden representation becomes the player's **behavioral fingerprint embedding**.

### 2. Player-Specific Baseline Learning

For each player, the system learns what their normal embedding distribution looks like across clean sessions.

This means the system does not ask:

> Does this look suspicious in general?

It asks:

> Does this look unlike this player's normal motor behavior?

### 3. Cheat-Likelihood Calibration

A second learned model maps online anomaly signals into a cheat probability using features such as:

- embedding drift from baseline
- direction-entropy collapse
- curvature drop
- unusually stable click-motion coupling
- burst regularity
- server confounders

This is what turns raw anomaly evidence into a calibrated moderation score.

### 4. What Is Not ML

Some parts should stay non-ML:

- online change-point detection
- the causal/confounder gate

That gives the project a stronger story:

> ML learns the player's motor fingerprint and anomaly structure, while the causal gate reduces false positives from known external factors.

## Recommended System

Build a **NoScope-Bio Demo** with six components.

### 1. Synthetic Motor-Telemetry Generator

Generate telemetry for about 20 synthetic players.

Each player should have stable individual tendencies across:

- cursor burst length
- correction style
- micro-jitter amplitude
- curvature tendencies
- pause timing
- click cadence
- movement tempo
- response to network instability

Human-like sessions should be built from:

- overlapping smooth submovements
- signal-dependent noise
- session-specific jitter and tempo variation
- occasional pauses and hesitations
- corrective micro-motions
- optional colored or long-range correlated micro-noise

Generate:

- clean baseline sessions
- clean evaluation sessions
- suspicious sessions with `aimbot_like_locking`
- suspicious sessions with `triggerbot_like_clicking`
- suspicious sessions with `macro_consistency`
- confounder sessions with `high_ping`
- confounder sessions with `sensitivity_change`
- confounder sessions with `patch_shift`

### 2. Feature Pipeline And Online Segmentation

Use only target-agnostic live features.

Core features:

- `dx`, `dy`
- speed
- acceleration
- jerk
- heading change
- angular velocity
- curvature
- path tortuosity or straightness
- movement burst duration
- pause duration
- click-to-motion coupling
- entropy of direction changes
- roughness or autocorrelation summaries
- ping
- jitter
- packet loss
- command age

Segment the motion stream online using:

- velocity threshold crossings
- pause boundaries
- jerk minima
- direction-change boundaries

### 3. Fingerprint Model

Train a lightweight causal temporal model on clean windows to identify player ID. Use the penultimate layer as the behavioral fingerprint embedding.

Why this works:

- it is real representation learning
- it is easy to explain in class
- it naturally supports personalized anomaly detection

### 4. Online Shift Detector

Stream windows from a session and compare them against that player's historical baseline.

Compute:

- embedding distance from baseline
- feature-level anomaly scores
- rolling suspicion score

Then run:

- Page-Hinkley, or
- CUSUM

Output:

- change-point timestamp
- suspicion score over time
- flagged windows

### 5. Causal Plausibility Gate

Do not let every anomaly become cheating.

Use a causal-style validation layer with explicit confounders:

- if ping and jitter spikes explain degraded smoothness or delayed response, reduce suspicion
- if sensitivity changed, allow broad cursor-dynamics changes but not suspicious click locking
- if a patch-style environment shift is active, tolerate moderate movement-style drift unless it is paired with low-entropy motor behavior

This can honestly be presented as a **causal-inspired validation layer** even if it is rule-backed rather than full causal discovery.

### 6. UI

Use Streamlit and Plotly.

The demo UI should support:

- uploading a session package
- displaying a cursor-only replay pane
- showing a suspicion timeline
- showing a change-point marker
- showing a before-vs-after comparison
- showing a "what changed?" explanation panel
- showing server telemetry below the score
- showing a final moderator verdict card

## The "Wow" Features

These will make the demo memorable:

- replay timeline with the exact flag moment marked
- before-vs-after comparison around the change point
- explanation card showing which cursor features shifted
- confounder panel showing why a lag spike was not flagged as cheating
- embedding visualization showing drift away from the player's normal cluster

## Example Evidence The System Should Show

For a flagged session, the report should explain things like:

- direction entropy collapsed sharply
- curvature dropped and movement became too straight
- jerk variability fell during suspicious lock-like bursts
- click timing became tightly coupled to abrupt cursor stabilization
- embedding distance from the player's clean baseline spiked

## Two-Day Build Plan

### Day 1

1. Lock the stack:
   - Python
   - PyTorch if stable, otherwise a lightweight sklearn or numpy fallback
   - Streamlit
   - Plotly
   - pandas
   - numpy
   - scikit-learn
2. Rewrite the synthetic generator around cursor-only motor dynamics.
3. Generate baseline and evaluation datasets.
4. Train the player fingerprint encoder.
5. Validate that embeddings separate players and drift during suspicious sessions.

### Day 2

1. Build the shift detector.
2. Add the causal plausibility gate.
3. Build the Streamlit UI.
4. Create three polished demo uploads:
   - clean session
   - lock-like suspicious session
   - laggy but legitimate session
5. Export evaluation metrics, charts, and screenshots for the presentation.

## Demo Script

Present the system in this order:

1. Upload a clean session and show that the model remains stable.
2. Upload a suspicious cursor-only session and show the score spike at the change point.
3. Show the replay timeline and "what changed?" panel.
4. Upload a high-ping or sensitivity-change session and show that the causal gate suppresses the final cheat flag.
5. End with evaluation metrics, a confusion matrix, and a short ethics/privacy note.

## Recommended Tech Stack

- `PyTorch` for the sequence encoder if available
- `pandas`, `numpy`, and `scikit-learn` for data handling and evaluation
- `Streamlit` and `Plotly` for the UI
- `River` only if installation is easy
- `DoWhy` only if it does not slow development

If `River` or `DoWhy` become a time sink, implement simple versions of the detector and causal gate directly and present them honestly.

## Best Framing For Class

We should present this as:

> A personalized behavioral anti-cheat prototype that learns each player's cursor-motion fingerprint, detects statistically improbable shifts in that fingerprint over time, and validates those shifts against plausible external confounders before producing a moderator-facing explanation.

That framing is more scientifically defensible, better aligned with causal online inference, and still ambitious enough for a strong class demo.
