# NoScope-Bio: Teammate Report and PPT Notes

## Small Report

### What We Built

`NoScope-Bio` is now a working anti-cheat demo with two connected parts:

- a **synthetic personalized demo** that models player-specific aim behavior and confounders
- a **real CSGO benchmark tab** that evaluates the same project idea on held-out player data from the uploaded archive

The Streamlit app shows:

- replay-style yaw/pitch visualization
- synchronized model timeline playback
- explanation text for suspicious sessions
- held-out test metrics
- a real-data CSGO tab with an official saved classifier

### Main Technical Finding

For the real CSGO benchmark, the best model was **session-level logistic regression**, not the LSTM variants.

Why that matters:

- it gave the best **balanced accuracy**, which is more important than raw accuracy for this problem
- it had stronger cheat recall than most alternatives
- it is much lighter and more stable to deploy in the app

Held-out CSGO test results for the deployed model:

- accuracy: `0.712`
- balanced accuracy: `0.711`
- precision: `0.579`
- recall: `0.707`
- specificity: `0.715`
- F1: `0.637`
- MCC: `0.407`

### Why Logistic Regression Won

Model comparison showed:

- `baseline_logistic` had the best balanced accuracy: `0.7107`
- `medium_lstm` was close but lower: `0.6930`
- `tiny_lstm` had higher raw accuracy, but lower balanced accuracy and much worse recall behavior than logistic overall for deployment framing

So the final CSGO tab now uses:

- **logistic regression** for the official session verdict
- the **causal window timeline** as supporting evidence and explainability

### Scientific / Project Framing

The project now makes a more realistic claim:

- we do not assume full raw mouse telemetry in real FPS systems
- instead, we model **view-angle telemetry**, firing, and target-relative CSGO archive fields when available
- the model uses **causal windows only**, so at time `t` it does not use future information

### What The Causal Gate Is

The causal gate is the part of the system that tries to answer:

> Is this suspicious shift actually cheating, or could it be explained by outside conditions?

In the synthetic pipeline, the gate reduces suspicion when the same time window also shows plausible confounders such as:

- ping spikes
- jitter increases
- packet loss
- sensitivity changes
- patch or environment shifts

So the gate does **not** learn cheats directly. It acts as a correction layer on top of the anomaly score to reduce false positives when there is a legitimate external explanation.

### What The Current Behavioral Fingerprint Encoder Is

Right now, the encoder is a lightweight **causal window encoder** implemented with an `sklearn` `MLPClassifier` over flattened trailing windows.

Synthetic path:

- input: causal windows of view-angle telemetry, timing, and server-side features
- training target: player identity
- output use: the hidden representation becomes each player's behavioral fingerprint embedding

CSGO path:

- input: causal windows of CSGO engagement telemetry
- training target: window-level legit vs cheater structure
- output use: the encoder signal becomes part of the evidence used by the session classifier

Why we defined it this way:

- it gives us a real learned representation, not just hand-made thresholds
- it stays causal
- it is lightweight enough to train on this machine
- it still supports the “behavioral fingerprint” story from the proposal

### What The Current Change-Point Detection Is

The change-point detector is currently **Page-Hinkley**.

What it does:

- reads the running suspiciousness score over time
- tracks whether the score has shifted upward in a sustained way
- marks the first time a change is statistically meaningful enough to count as an event

Why we use it:

- it works online
- it is lightweight
- it matches the proposal’s “behavioral shift detection” requirement
- it gives us a timestamp we can show in the UI

So in the app, the “timestamp of event” comes from the first detected change point in the causal timeline.

### Main Caveat

The synthetic tab is still the polished demo path.

The CSGO archive is the more realistic benchmark, but it is harder because:

- it has only 5 raw input fields
- it does not include server confounders like ping or jitter
- it does not provide clean baseline history for each cheater identity

Also, the current Streamlit demo export is a **smaller subset** of the full archive for speed:

- full archive: `10,000` legit players and `2,000` cheater players
- current exported demo run: `60` legit players and `30` cheater players

That smaller number is just the current export/build size for the demo, not the real archive size.

That means we should present the synthetic results as the polished demo and the CSGO results as the tougher real-data addendum.

## What To Include In The PPTX

### Slide 1. Problem and Motivation

- cheating hurts fairness and trust in competitive FPS games
- signature-based anti-cheat misses adaptive tools and behavior-mimicking cheats
- our idea: detect suspicious behavioral shifts instead of only known cheat signatures

### Slide 2. System Overview

Show the pipeline:

- telemetry input
- causal windowing
- learned encoder / feature extraction
- session classifier
- change-point timeline
- explanation layer

Key sentence:

> We learn behavioral signatures from aim telemetry and flag improbable shifts consistent with assistive aiming or scripted behavior.

Make sure this slide also covers the architecture/workflow items your professor asked for:

- **input to the model**
- **intermediate processing steps**
- **final output of the model**

### Slide 3. Inputs and Features

For the real CSGO archive:

- `AttackerDeltaYaw`
- `AttackerDeltaPitch`
- `CrosshairToVictimYaw`
- `CrosshairToVictimPitch`
- `Firing`

Engineered features:

- target error and error-drop signals
- angular speed, acceleration, jerk
- lock-like stability
- snap-like features
- entropy and curvature reductions
- fire-alignment and fire-stability coupling

This slide is your **input examples** slide for Section 2.

### Slide 4. Where ML Enters

- causal window encoder learns window-level structure
- aggregated session features go into the final classifier
- best deployed real-data model: **logistic regression**
- timeline remains as causal supporting evidence

Use one line for the encoder:

> The behavioral fingerprint encoder is a lightweight MLP over causal telemetry windows; it learns compact aim-behavior representations that feed the final classifier.

Use one line for the change-point detector:

> We use Page-Hinkley to detect the first sustained upward shift in suspiciousness over time.

Use one line for the causal gate:

> The causal gate suppresses suspicious scores when lag, sensitivity shifts, or other confounders provide a more plausible explanation.

Important line:

> We tested larger models too, but logistic regression produced the best balanced-accuracy result on held-out CSGO players.

### Slide 5. Results

Use these numbers:

- accuracy `0.712`
- balanced accuracy `0.711`
- precision `0.579`
- recall `0.707`
- specificity `0.715`
- MCC `0.407`

Also mention:

- logistic regression beat the LSTM sweep on balanced accuracy
- this is why it became the deployed model in the app

This slide is your **measurement output** slide for Section 2.

### Slide 6. Demo Story

Show:

- synthetic tab: polished live replay and explanation
- CSGO tab: real benchmark and official classifier
- session verdict + timeline + explanation together

Point out that the app now explicitly shows:

- detected anomaly
- timestamp of event
- affected features
- confidence / deviation score

### Slide 7. Limitations and Future Work

- add more real telemetry sources
- incorporate real server/network signals
- collect cleaner per-player baseline histories
- improve triggerbot and macro detection
- test on more games or larger replay archives

## One Short Script You Can Say

> Our final system combines a polished synthetic anti-cheat demo with a real CSGO benchmark. The most important result is that on held-out CSGO players, a session-level logistic regression model outperformed the larger LSTM variants on balanced accuracy, so we used it as the deployed classifier in the app. The live timeline is still shown because it gives causal, replay-aligned evidence for when suspicious behavior emerges.

## Section 2 Checklist

If your teammates are writing the architecture/workflow section, make sure they explicitly include these three things:

### 1. Input To The Algorithm / Model

- synthetic telemetry CSVs or CSGO archive tensors
- examples of raw fields and engineered features
- short example of one session / one engagement input

### 2. Intermediate Processing Steps

- feature engineering
- causal trailing-window construction
- behavioral fingerprint encoding
- session-level classification
- change-point detection
- causal gate filtering

### 3. Final Output

Data output:

- session verdict
- anomaly timestamp
- affected features
- confidence / deviation score
- replay-aligned explanation

Measurement output:

- accuracy
- balanced accuracy
- precision
- recall
- specificity
- MCC
