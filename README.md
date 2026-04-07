# NoScope-Bio

Behavioral anti-cheat demo for `CAI 4002`.

This project now has two connected modes:

- a **synthetic personalized demo** built around FPS-style view-angle telemetry and per-player baselines
- a **real CSGO archive benchmark** built from engagement tensors under `/real_data/archive`

Together they let you show:

- a polished end-to-end anti-cheat demo with live replay and explainability
- a harder real-data benchmark using held-out CSGO players

## Design Rationale

The core design choice in this project is that aim behavior should **not** be modeled as either:

- pure deterministic motion
- pure random noise

Instead, we model it as a combination of:

- **smooth intentional movement**
- **stochastic micro-variation**

That is the reasoning behind the project’s “behavioral” framing.

In practice, that means we assume a real player’s view-angle stream is made of:

- purposeful turns, flicks, corrections, and settle phases
- small noisy adjustments caused by motor variability and timing noise

This is why the feature design is split across two families:

- **smooth / structured features** that describe controlled motion
- **stochastic / irregularity features** that describe human variability

This is a better anti-cheat framing than looking only for impossible aim snaps, because it asks whether the player’s motion still looks like a human motor process rather than a perfectly stabilized assistive system.

## Why The Motion Is Split Into Smooth + Stochastic

For the class project, the cleanest scientific story is:

- human aim has intentional structure
- human aim also has noise
- suspicious automation often removes too much of the noise or makes the structure too regular

So the project treats behavior as:

`behavior = smooth control signal + stochastic residual variation`

That affects both the synthetic generator and the feature engineering.

### Smooth / structured side

These features try to capture the deliberate part of movement:

- yaw / pitch deltas
- angular speed
- angular acceleration
- angular jerk
- heading change
- curvature
- straightness
- settling ratio
- target-error reduction
- snap-like transitions
- fire alignment and lock-like dwell

Why:

- these tell us how efficiently and directly the aim moves
- they capture flicks, corrections, and settle phases
- they are where aimbot-like behavior often becomes unnaturally clean or overly efficient

### Stochastic / variability side

These features try to capture the human irregularity that normally remains on top of movement:

- direction entropy
- reversal rate
- micro-correction score
- variance / standard deviation terms
- short-lag autocorrelation
- fire interval regularity
- stabilization-to-fire timing regularity

Why:

- human motor behavior is noisy, but not purely random
- real players usually keep some inconsistency and micro-correction
- automation often collapses this variability into unusually stable, low-entropy motion

## Why These Feature Choices Make Sense

The feature set was chosen to match what the system can realistically observe and what cheating would likely change.

### Synthetic demo features

The synthetic path uses:

- view angles
- angular motion features
- timing features
- fire-coupling features
- server/network confounders

Why:

- this supports the personalized fingerprint story
- it gives the causal gate real confounders to reason about
- it lets the app show both suspicious behavior and “not cheating, just lag” cases

### CSGO archive features

The real-data path starts from the 5 archive fields:

- `AttackerDeltaYaw`
- `AttackerDeltaPitch`
- `CrosshairToVictimYaw`
- `CrosshairToVictimPitch`
- `Firing`

Those are expanded into:

- target-error and target-error-improvement features
- angular speed / acceleration / jerk
- lock-like stability features
- snap-power and settling features
- entropy and curvature drops
- fire-alignment and fire-stability coupling

Why:

- they are directly grounded in the available archive telemetry
- they let us measure both control efficiency and human irregularity
- they are good candidates for detecting assistive aiming without pretending we have full hardware-side input logs

## Quick Start

1. Generate the synthetic demo bundle:

```bash
python3 scripts/export_demo_bundle.py
```

2. Generate the real CSGO archive bundle:

```bash
python3 scripts/export_csgo_bundle.py
```

For a faster smoke test, you can export a smaller subset:

```bash
python3 scripts/export_csgo_bundle.py --legit-limit 60 --cheat-limit 30
```

3. Train and save the best offline CSGO session model:

```bash
python3 scripts/train_csgo_session_models.py
```

This currently saves `baseline_logistic`, which is the model the CSGO tab uses as its official session classifier.

4. Launch the UI:

```bash
streamlit run app.py
```

## What The Demo Shows

- clean synthetic sessions that stay near a player's normal aim baseline
- suspicious synthetic sessions with locking-like, triggerbot-like, and `macro_consistency` behavior
- confounder sessions with `high_ping` and `sensitivity_change`
- a second Streamlit tab for real CSGO archive engagements
- replay-style view-angle traces, suspicion timelines, and explanations in both tabs
- held-out metrics and Bayes-theorem quality checks for both datasets

## Where ML Enters

The synthetic demo and the CSGO archive both use a two-stage ML setup:

- a first model that learns a compact embedding from **causal windowed aim telemetry**
- a second learned layer that calibrates embedding drift and engineered anomalies into a cheat-likelihood score

In the synthetic demo, the first model learns **player fingerprints** from clean sessions.

In the CSGO archive benchmark, the first model learns **window-level cheat-vs-legit structure** from real engagement windows, then the second layer combines that encoder signal with engineered aim-alignment, lock-like, snap-like, and fire-coupling features.

The current exported CSGO benchmark metrics in the repo come from the causal window pipeline:

- window encoder: `sklearn` `MLPClassifier` over flattened causal windows
- cheat scorer: `sklearn` `LogisticRegression` over engineered window-shift features

The real-data modeling notebooks also compare full session models:

- a baseline session-level logistic regression
- a medium gradient-boosting model
- three ordered-window LSTMs: `tiny_lstm`, `small_lstm`, and `medium_lstm`

That means the project uses:

- sequence representation learning
- personalized anomaly detection
- probability calibration

## Current CSGO Deployment Choice

The official CSGO classifier shown in Streamlit is the saved **session-level logistic regression** model.

We kept it over the LSTM sweep because it had the best balanced-accuracy result in the notebook comparison:

- `baseline_logistic`: accuracy `0.7119`, balanced accuracy `0.7107`, precision `0.5792`, recall `0.7067`, F1 `0.6366`, MCC `0.4073`
- `medium_lstm`: accuracy `0.6929`, balanced accuracy `0.6930`, precision `0.5562`, recall `0.6933`, F1 `0.6172`, MCC `0.3721`
- `tiny_lstm`: accuracy `0.7381`, balanced accuracy `0.6881`, precision `0.6754`, recall `0.5133`, F1 `0.5833`, MCC `0.4055`
- `small_lstm`: accuracy `0.7167`, balanced accuracy `0.6804`, precision `0.6148`, recall `0.5533`, F1 `0.5825`, MCC `0.3701`
- `medium_hgbt`: accuracy `0.7071`, balanced accuracy `0.6463`, precision `0.6311`, recall `0.4333`, F1 `0.5138`, MCC `0.3259`

So even though `tiny_lstm` had the highest raw accuracy, logistic regression was the best legit-vs-cheater balance on the held-out CSGO players. The app now uses logistic regression for the final CSGO session verdict and probability.

The causal window timeline is still shown underneath it because it provides:

- time-localized evidence
- replay synchronization
- explanation of when the suspicious behavior emerged

## Current Design Choice

The main synthetic input is a telemetry session CSV, not raw gameplay video. That keeps the project honest, controllable, and demo-ready in the short time available.

For the real benchmark, the project uses a CSGO archive stored as numpy arrays with shape `(players, 30, 192, 5)` where the raw features are:

- `AttackerDeltaYaw`
- `AttackerDeltaPitch`
- `CrosshairToVictimYaw`
- `CrosshairToVictimPitch`
- `Firing`

Those raw fields are expanded into engineered features like target error, target-error improvement, on-target dwell, lock-like stability, angular jerk, directional entropy, snap indicators, and fire-motion coupling.

## Evaluation Quality

The project reports more than raw accuracy. It includes:

- balanced accuracy
- specificity
- majority-class baseline accuracy
- positive and negative predictive value at the observed test prevalence
- Bayes-theorem posterior cheat probabilities under lower deployment prevalences such as 10%, 5%, 1%, and 0.1%

That helps show the detector is not just benefiting from class imbalance.

## CSGO Notebooks

There are now notebook paths for the real-data workflow:

- [notebooks/csgo_eda.ipynb](/Users/dsuniaga/Desktop/cai4002%20proj/notebooks/csgo_eda.ipynb)
- [notebooks/csgo_modeling.ipynb](/Users/dsuniaga/Desktop/cai4002%20proj/notebooks/csgo_modeling.ipynb)
- [notebooks/csgo_feature_engineering.ipynb](/Users/dsuniaga/Desktop/cai4002%20proj/notebooks/csgo_feature_engineering.ipynb)

They read the exported files under [data/csgo_generated](/Users/dsuniaga/Desktop/cai4002%20proj/data/csgo_generated), including:

- `window_features.csv`
- `classifier_windows.csv`
- `session_feature_table.csv`

The persisted offline session model is saved under [artifacts/csgo_models](/Users/dsuniaga/Desktop/cai4002%20proj/artifacts/csgo_models) and is loaded automatically by the CSGO Streamlit tab if present.
