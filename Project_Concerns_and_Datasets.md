# NoScope-Bio: Concerns, Risks, and Dataset Options

## What Currently Concerns Me

### 1. The scientific framing is stronger than the current implementation in a few places

The right story for this project is now clear:

- target-agnostic cursor modeling
- causal online windows
- player-specific motor fingerprints
- confounder-aware validation

That framing is stronger than a target-relative anti-cheat story, because in a real server-side or replay-derived setting we often do **not** know the exact target state needed to compute features like `aim_error` or `aligned`.

Main takeaway:

- the project direction is now more defensible
- the documentation should follow that framing
- the codebase should continue moving away from target-relative live features

### 2. The detector is split-calibrated for the demo, but the evaluation is still synthetic

The current prototype works end-to-end with a learned calibration layer on top of the behavioral fingerprint and causal gate. The latest exported synthetic demo snapshot is:

- accuracy: `0.986`
- precision: `0.968`
- recall: `1.000`
- false positive rate: `0.025`
- false negative rate: `0.000`

These results are useful for the class demo, but they are still synthetic holdout results.

### 3. The earlier overfitting concern was real

The earlier perfect metrics came from a setup that was too easy and too tightly coupled to the evaluation data. The project now uses:

- more baseline sessions per player
- multiple replicated evaluation sessions per mode
- session-to-session variability
- a calibration-train split
- a validation split
- a held-out test split shown in Streamlit

That is much better, but the project should still be presented honestly as a prototype.

### 4. Synthetic realism is the biggest scientific weakness

The demo data is good for showing architecture, personalized baselines, and explainability, but it does not yet prove real-world anti-cheat performance.

The synthetic generator should aim to mimic:

- smooth submovements
- pauses and hesitations
- corrective micro-motions
- signal-dependent noise
- session-to-session motor variation

not just simplistic target chasing.

### 5. There is no perfect public FPS anti-cheat benchmark

I did not find a strong open benchmark that already contains:

- per-player baseline cursor telemetry
- labeled aimbot-like or triggerbot-like sessions
- server-side ping, jitter, and packet-loss context
- clean anti-cheat evaluation splits

That means synthetic generation and augmentation are still necessary.

### 6. Server-visible features help the demo a lot

The project becomes more believable when it uses not just cursor and click behavior, but also server-style signals such as:

- ping
- jitter
- packet loss
- command age
- packet inter-arrival timing
- input burstiness
- correction and desync signals

These give the causal gate something concrete to reason about.

### 7. The environment forced one implementation change

`torch` crashes on this machine, so the current fingerprint encoder uses a lighter fallback stack instead of a full PyTorch sequence model. That is acceptable for the class demo, but it is worth knowing when describing the current prototype.

## Motor-Control Framing We Should Use

The best scientific framing is:

- full cursor trajectories are **not** pure Brownian motion
- intentional motion is better modeled as **smooth submovements**
- residual jitter is stochastic but often structured
- motor noise can be signal-dependent
- suspicious automation may appear as low-entropy, low-correction, overly straight, or overly regular micro-trajectories

That supports the shift to:

- cursor-only inputs
- target-agnostic anomaly scoring
- causal online windows

## What Is Already In Place

- synthetic multi-player session generation
- per-player baselines
- cheat modes for `aimbot`, `triggerbot`, and `macro_consistency`
- confounder modes for `high_ping`, `sensitivity_change`, and `patch_shift`
- live timeline animation in the UI
- replay-style visualization
- server telemetry plots
- exported CSV sessions and evaluation metrics
- section-2 architecture writeup

## Best Public Dataset Options

### Option 1. ESTA / awpy Counter-Strike demo data

Best use:

- realistic game-state and event telemetry
- player positions, actions, trajectories, and frame-level esports data
- useful for replacing parts of the synthetic movement and action layer

Why it helps:

- closest thing to real game-log telemetry in an FPS-like setting
- useful for player movement and action structure
- good source for realistic temporal patterns

Limitations:

- not labeled for cheating
- may not include every anti-cheat feature you want
- still needs synthetic cheat injection or augmentation

Links:

- [ESTA dataset repository](https://github.com/pnxenopoulos/esta)
- [awpy on PyPI](https://pypi.org/project/awpy/)
- [ESTA paper summary](https://deepai.com/publication/esta-an-esports-trajectory-and-action-dataset)

### Option 2. Minecraft Mouse Dynamics Dataset

Best use:

- gaming-context mouse trajectories
- user-specific mouse behavior for identity modeling

Why it helps:

- closer to gameplay mouse behavior than generic office mouse datasets
- useful for training or validating fingerprint-style mouse embeddings

Limitations:

- small compared to modern ML standards
- not FPS aiming data
- not server telemetry

Link:

- [GitHub repository](https://github.com/NyleSiddiqui/Minecraft-Mouse-Dynamics-Dataset)

### Option 3. Boğaziçi University Mouse Dynamics Dataset

Best use:

- continuous mouse behavior
- user identity verification
- anomaly and impostor-style experiments

Why it helps:

- useful for pretraining mouse-dynamics feature extractors
- can help regularize synthetic cursor data generation

Limitations:

- not game-specific
- not cheat-labeled
- no FPS target or shooting context

Links:

- [Mendeley Data dataset page](https://data.mendeley.com/datasets/w6cxr8yc7p/2)
- [Paper summary on PubMed](https://pubmed.ncbi.nlm.nih.gov/34041314/)

### Option 4. KVC-onGoing / Aalto Keystroke resources

Best use:

- timing-based behavioral biometrics
- keyboard cadence and consistency modeling
- account-sharing or identity-shift style experiments

Why it helps:

- extremely large-scale timing data
- strong source for learning general biometric-timing embeddings

Limitations:

- typing data, not gameplay
- no aiming or shooting context
- best used as auxiliary pretraining or augmentation inspiration

Link:

- [KVC-onGoing paper](https://www.sciencedirect.com/science/article/pii/S0031320324010380)

### Option 5. Berkeley VR motion data

Best use:

- large-scale behavioral identity modeling from motion traces
- framing support for the behavioral-fingerprint idea

Why it helps:

- closest academic precedent to the identity-conditioned story
- useful as related-work support and augmentation inspiration

Limitations:

- VR motion, not mouse or FPS telemetry
- very large and not anti-cheat labeled

Link:

- [Berkeley VR motion dataset page](https://rdi.berkeley.edu/metaverse/identification/)

## Related Research Pointers For The Writeup

These are useful for justifying the target-agnostic cursor-motor framing:

- [Flash and Hogan (minimum-jerk movement)](https://pubmed.ncbi.nlm.nih.gov/4020415/)
- [Flash and Hogan overview on PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6565116/)
- [Harris and Wolpert (signal-dependent noise)](https://pubmed.ncbi.nlm.nih.gov/9723616/)
- [Plamondon kinematic theory, Part I](https://pubmed.ncbi.nlm.nih.gov/7748959/)
- [Plamondon kinematic theory, Part IV](https://pubmed.ncbi.nlm.nih.gov/12905041/)
- [Strokes of insight: cursor trajectories](https://www.sciencedirect.com/science/article/abs/pii/S0306457316300723)
- [Torre and Wagenmakers on structured movement noise](https://pubmed.ncbi.nlm.nih.gov/19403189/)

## My Honest Recommendation

Use a hybrid strategy:

1. keep the current synthetic anti-cheat generator as the primary labeled dataset
2. borrow distributional ideas from public mouse or esports datasets
3. optionally import one public mouse dataset to regularize the clean baseline distribution
4. inject suspicious automation-like behavior synthetically on top of those clean traces

That lets you say something defensible:

> We used public real-user behavioral datasets where possible to inform cursor-motion distributions, then augmented them with synthetic suspicious and confounder scenarios to create a controlled anti-cheat evaluation environment.

## Most Practical Next Steps

1. Refactor the live feature set so it is fully target-agnostic.
2. Rewrite the synthetic generator around cursor-motor submovements.
3. Freeze three demo sessions:
   - clean
   - suspicious locking-like behavior
   - laggy but legitimate
4. Freeze screenshots and charts for the slide deck.
5. If time allows, test one public dataset for augmentation instead of replacing the whole pipeline.
