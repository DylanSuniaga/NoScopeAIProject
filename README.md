# NoScope-Bio

Behavioral anti-cheat demo for `CAI 4002`.

This project is scoped as a polished class demo built around **target-agnostic aim telemetry**:

- synthetic FPS-style view-angle telemetry
- per-player behavioral fingerprints
- online shift detection
- causal-style confounder gating
- a simple review UI

## Quick Start

1. Generate example sessions and baseline artifacts:

```bash
python3 scripts/export_demo_bundle.py
```

2. Launch the UI:

```bash
streamlit run app.py
```

## What The Demo Shows

- clean sessions that stay near a player's normal aim baseline
- suspicious sessions with locking-like, triggerbot-like, and `macro_consistency` behavior
- confounder sessions with `high_ping` and `sensitivity_change`
- a replay-style view of yaw/pitch motion over time
- an explanation panel for why the system flagged or suppressed a session

## Where ML Enters

The main ML component is a fingerprint model trained on **causal windowed aim telemetry** that learns a compact view-angle embedding for each player from clean sessions. A second learned layer calibrates drift, feature anomalies, and confounders into a cheat-likelihood score.

That means the project uses:

- sequence representation learning
- personalized anomaly detection
- probability calibration

## Current Design Choice

The main input is a telemetry session CSV, not raw gameplay video. That keeps the project honest, controllable, and demo-ready in the short time available. In a real FPS pipeline this would come from view angles such as `yaw` and `pitch`, angular deltas, aim vectors, fire-input timing, flick events, and server telemetry. The demo models those signals directly in a normalized angle space instead of a full game engine.

## Evaluation Quality

The project now reports more than raw accuracy. It includes:

- balanced accuracy
- specificity
- majority-class baseline accuracy
- positive and negative predictive value at the observed test prevalence
- Bayes-theorem posterior cheat probabilities under lower deployment prevalences such as 10%, 5%, 1%, and 0.1%

That helps show the detector is not just benefiting from class imbalance.
