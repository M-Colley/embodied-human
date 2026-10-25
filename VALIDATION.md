# Validation

## 2026-10-24, first PC (Windows 11, CPU)

Python 3.12.13, NumPy 2.5.4, SciPy 1.17.1 and MuJoCo 3.17.0, with these
optional packages:

- embodied-ui 0.1.0 (commit `aa08385`);
- myosuite-carla 0.1.0 (commit `85983b1`);
- MyoSuite 3.0.0 from its `dev` branch, with myo-sim 0.2.3 (`myofullbody`).

### Tests

```bash
python -m pytest tests -q
```

The suite passes 30 tests ([JUnit](evidence/tests.xml)), including 4
integrated driving tests on myosuite-carla's fitted vehicle surrogate.
Modules are checked against their equations:

- the main sequence and minimum-jerk profile; saccadic suppression and
  queueing;
- foveation blurring only the periphery;
- noise scaling with eccentricity;
- uncertainty growth with speed;
- sampler thresholds, SEEV probabilities and AttenD timing per Kircher and
  Ahlström (2009);
- ACT-R activation, retrieval and latency;
- Fitts timing and endpoint spread;
- spindle rates per Prochazka and Gorassini (1998), with delays, on a
  MuJoCo muscle;
- the SuRT, DRT (ISO 17488 classification), n-back and ISO tapping formulas;
- the registry, module replacement and benchmark scoring.

### Benchmark

```bash
python -m embodied_human.bench --out evidence/benchmark-baseline.json
```

The full run (seed 0) took about 90 s; see the
[report](evidence/benchmark-baseline.json). Seven of 8 benchmarks pass. Module parameters by source: 8 measured, 11 fitted,
3 standard, 1 derived and 10 assumed. The driver agent adds 15 assumed and
2 measured ([docs/parameters.md](docs/parameters.md)).

| Benchmark | Result | Human reference | Kind |
| --- | --- | --- | --- |
| eyes.main_sequence | 2.20 ms/deg, 21.0 ms | 2.0–3.7 ms/deg; 13–37 ms | consistency |
| eyes.latency | median 201 ms | 150–250 ms | consistency |
| objects.peripheral_acuity | E2 1.98° | 1.0–2.6° | consistency |
| vision.foveation | E2 2.30° | 1.0–3.0° | consistency |
| memory.span | 4 | 3–5 | consistency |
| motor.touch_tapping | 6.28 bits/s; 104 ms/bit | 5.5–8.5 bits/s; 75–135 ms/bit | physical contact on an embodied-ui touch surface |
| driving.surt_glances | mean off-road glance 1.45 s (85th percentile 1.52 s); 0 of 34 glances over 2 s | NHTSA criteria ≤ 2 s and ≤ 15%: pass. Human data, mean 0.8–1.2 s and 3–13% over 2 s: fail (score 0.77) | emergent |
| driving.takeover | 1.66 s mean (SD 0.18 s, 8 runs) | 2.72 s, mean ± 1 SD across 520 conditions | emergent |

Further measurements from the same runs:

- **Glances while doing SuRT.** Over 60 s at 20 m/s on a straight road, the
  driver kept a lane-offset SD of 0.21 m (maximum 0.48 m). It completed 14
  SuRT screens at 86% accuracy, 4.2 s per screen, with 3.5 s eyes-off-road
  per screen. The AttenD buffer reached zero, so continuous SuRT engagement
  is classified as distracted at times.
- **Take-over.** Every request was taken over, in 1.42–1.96 s.

**The benchmark responds to module changes.** With a laxer attention
threshold (`--replace attention=uncertainty_sampler:return_sd=0.8,maximum_glance=4.0`,
[report](evidence/benchmark-return-sd-0.8.json)), the mean off-road glance
rose to 2.25 s and every glance exceeded 2 s. `driving.surt_glances` failed
(score 0.22, against 0.77 for the baseline), while the take-over time
stayed at 1.68 s.

### Interpretation and limits

- **What the passes show.** Five benchmarks only confirm that the modules
  implement their cited parameters. The tapping, glance and take-over
  results arise from interacting modules.
- **Glances.** Meeting the NHTSA design criteria shows plausibility only.
  Against human in-vehicle search glances, the baseline's glances are too
  long and too regular:
  - simulator means are 1.06–1.20 s with 5.7–10% over 2 s (Kujala and
    Saariluoma, 2011);
  - about 1.0 s per in-car glance in Jokinen, Kujala and Oulasvirta's
    (2021) human data;
  - naturalistic phone tasks have a median of 3.4% over 2 s and a mean of
    12.8% (Tivesten and Dozza, 2014).

  Their tasks and speeds differ from SuRT at 72 km/h. Jokinen et al. found
  no significant effect of speed on glance duration.
- **Take-over.** The baseline is faster and far less variable than the
  meta-analysis. It models one simulated participant without
  inter-individual variation. Its take-over steps (response latency,
  assessment fixations, hand return) use assumed parameters.
- **What was not validated:** pixel-level vision in the driving loop, a
  musculoskeletal reaching controller, and comparison with individual-level
  human data.

## 2026-10-24, evening: simulated participants, inference and studies

Same PC and versions, with embodied-ui at commit `5ce2df5`. These changes
follow the requirements for computational user models discussed at
Dagstuhl Seminar 26432 ([mapping](docs/seminar-requirements.md)).

### Tests

The suite passes 43 tests ([JUnit](evidence/tests.xml)), up from 30. New
tests cover:
- recovery of a known parameter by ABC;
- priors and distances;
- populations: reproducibility, spreads, profiles, and variations applied
  only to modules that have them;
- bootstrap intervals, Kendall's tau and elasticity;
- reasons for gaze returns, belief growth off the road, and shorter
  glances at higher speed;
- CSV traces;
- the take-over components;
- the glance and key-size studies.

### Calibration

```bash
python examples/calibrate_glances.py --workers 6 --out evidence/glance-calibration.json
```

**Setup.**
- **Fitted:** the attention module's return threshold (`return_sd`) and its
  glance-to-glance spread (`threshold_spread`).
- **Targets:** two summaries of Kujala and Saariluoma (2011, Table 2), the
  mean in-vehicle glance (1.12 s) and the mean within-person SD (0.57 s).
- **Simulation:** at 50 km/h on `StraightRoad`, with two 300 s runs per
  parameter set.
- **Method:** rejection ABC with 512 scrambled Sobol draws, keeping the
  closest 5% (26).
- **Wall time:** about 3.5 minutes on six processes, alongside other jobs.

**First attempt, rejected.** Narrower priors (`return_sd` 0.1–0.6 m,
`threshold_spread` 0–1, one run per draw) gave posteriors pressed against
both prior bounds. Checks on a fresh random stream also fell short of the
fitted SD (0.44 against 0.57 s): the rejection had selected draws whose
single run happened to match. Wider priors (0.03–0.6 m, 0–2) and two pooled
runs fixed both problems.

| Parameter | Posterior median | 90% interval | Default now |
| --- | --- | --- | --- |
| `return_sd` | 0.110 m | 0.060–0.181 m | 0.11 m (was 0.4, assumed) |
| `threshold_spread` | 1.01 | 0.64–1.34 | 1.0 (was 0, assumed) |

**Posterior predictive checks** (fresh random streams; 5th, 50th and 95th
percentiles over the accepted parameter sets):

| Quantity | Predicted | Human | |
| --- | --- | --- | --- |
| Mean glance | 1.01, 1.10, 1.23 s | 1.06–1.20 s | fitted |
| Within-person SD | 0.45, 0.60, 0.70 s | 0.48–0.66 s | fitted |
| Share over 2 s | 4.1, 7.5, 10.5% | 5.7–10% | held out: agrees |
| Longest glance | 3.75, 4.15, 4.28 s | 2.42–3.70 s | held out: too long |

### Benchmark

```bash
python -m embodied_human.bench --out evidence/benchmark-baseline.json
```

The run took about 2 minutes ([report](evidence/benchmark-baseline.json)); 9 of 11
benchmarks pass. Module parameters by source: 8 measured, 13 fitted,
2 standard, 1 derived and 10 assumed. The 4 s maximum glance is now
labelled assumed: it is a bound chosen above the measured longest glances.
The six module benchmarks are unchanged. The new and changed integrated
results:

| Benchmark | Result | Human reference | Kind |
| --- | --- | --- | --- |
| driving.surt_glances (MyoSuite full body, 72 km/h, 60 s) | mean 0.93 s, SD 0.47 s, 2.2% over 2 s; lane-offset SD 0.18 m; 13 SuRT screens at 85% | 0.8–1.2 s: pass; 3–13% over 2 s: just below (score 0.98); NHTSA criteria: pass | held out |
| driving.takeover | 1.61 s mean, SD 0.27 s; median response 0.30 s, eyes on road 0.60 s, hand on wheel 0.71 s, driving 1.42 s | 1.27–4.17 s | emergent |
| driving.glance_population (20 participants, 50 km/h) | mean 1.13 s (95% CI 1.08–1.19), within-person SD 0.58 s (0.54–0.64), between-person SD of means 0.12 s, 7.9% over 2 s, longest 4.2 s | 1.06–1.20 s; 0.48–0.66 s; 5.7–10% | fitted (mean, SD) and held out (share) |
| driving.glance_speed (40–120 km/h) | elasticity −0.45; 1.22 s at 40 km/h, 0.74 s at 120 km/h | occlusion −0.72 to −0.53; in-car search 0 ± 0.15: both fail | counterfactual |
| design.touch_key_size (30 participants, 7.5–27.5 mm) | errors 18.5, 13.5, 10.9, 12.4 and 12.1%; only 7.5 mm differs from 27.5 mm (p < 0.05), so gains stop at 12.5 mm | 17.5 mm (12.5–22.5) | application |

With the uncalibrated defaults (`return_sd` 0.4, no spread), the same
studies gave the following, all failing the human data:
- a mean glance of 1.76 s;
- a within-person SD of 0.06 s;
- 17% of glances over 2 s.

### Interpretation and limits

- **Consistency checks.** Mean and SD in `driving.glance_population` were
  fitted, so they only check consistency.
- **Held-out results.**
  - The long-glance share and the full-body drive at a different speed
    are genuine out-of-sample results.
  - The longest glance and the speed effect miss the human data. The
    threshold's log-normal tail is too heavy.
  - Road uncertainty alone cannot reproduce both voluntary occlusion
    (shorter glances at speed) and in-car search (no speed effect).
- **Key size.** The simulated participants tapped without driving, so
  Kim et al.'s speed conditions were not reproduced. The test statistics
  depend on the simulated sample size, as they do in the study.
- **Still not validated:** individual-level comparison with human data,
  and between-person spreads other than joint position sense.

## 2026-10-25: senses, responses, fatigue, emotion, risk and trust

Same PC; additionally scikit-learn 1.9.1 and MoSQITo 1.2.1. Eight module
categories were added, each reimplementing a published model from its
equations. Parameters and human reference data were collected from the
primary sources where reachable (in a few cases from secondary sources,
labelled so in `docs/parameters.md` and `bench/references.py`).

### Tests

`python -m pytest tests -q` passes 57 tests ([JUnit](evidence/tests.xml)).
New tests check:

- hearing thresholds, masking by cabin noise, Piéron latency, ISO 7731 and
  MoSQITo loudness (60 dB at 1 kHz gives 4.0 sone);
- vibration thresholds, which apply only with contact;
- TWIN means (a single signal gives latency + 217 ms, redundant signals
  less) and visual eccentricity;
- the conflict model at rest, the Wf weighting (1.012 at 0.167 Hz, 0.224 at
  0.5 Hz) and the ISO dose of a sinusoid;
- fatigue endurance (shoulder 275, 58 and 14 s at 20, 50 and 80% MVC, as an
  independent reimplementation found) and recovery at rest;
- trust inertia, hysteresis and drift without information;
- the risk field's closed form against grid integration (within 2%);
- the appraisal formulas and the boredom profile (0, 0, 0.5, 0.526);
- a take-over perceived by touch, with its trust drop and head motion.

### Checks against independent reimplementations

- **Emotion.** Retraining the three tasks of Zhang et al. (2024) gives
  appraisal profiles within the run-to-run spread of the published values:
  - happiness 0.20 / 0.74 / 0.87 / 0.82 (mean of 10 runs);
  - boredom exactly 0 / 0 / 0.5 / 0.526;
  - irritation 0.20 / 0.92 / 0.04 / 0.16.
- **Motion sickness.** The conflict model's equations follow the
  state-space form of Wada and Bos (2025) with the parameters of Wada et
  al. (2018), and the ISO weighting matches a published table.

### Benchmark

```bash
python -m embodied_human.bench --out evidence/benchmark-baseline.json
python -m embodied_human.bench --replace vestibular=iso_2631 --only vestibular.vertical_sickness --out evidence/benchmark-vestibular-iso2631.json
```

The full run took 217 s ([report](evidence/benchmark-baseline.json)).
**11 of 18 benchmarks pass.** Module parameters by source: 50 fitted,
10 measured, 3 standard, 4 derived and 18 assumed. New results:

| Benchmark | Result | Human reference | Kind |
| --- | --- | --- | --- |
| response.modality_rt | tone 252 < light 319 < vibration 335 ms | same order, 132 < 163 < 177 ms (Diederich & Colonius, 2004) | order only |
| response.drt | visual 349 ms; tactile 352 ms; older 467 ms (1.33×) | 326–362 ms: pass; 526 ms (482–570): **fail**; 1.215× (1.07–1.36): pass | emergent |
| driving.takeover_modality | median response 0.27 s sound, 0.31 s touch, 0.33 s light; wheel grabbed after 0.65 s | sound and touch faster than light: pass; 1.87 s (1.36–2.38): **fail** | emergent |
| driving.takeover (MyoSuite) | 1.51 s, SD 0.18 s, 8 runs | 1.27–4.17 s: pass | emergent |
| vestibular.vertical_sickness | RMS error 19.7 % points, r = 0.92 | 14 conditions of O'Hanlon & McCauley: ≤ 10: **fail** | held out |
| vestibular.vertical_sickness, `iso_2631` | RMS error 6.1 | same | consistency (ISO derived partly from these data) |
| fatigue.endurance | median absolute log ratio 0.16 | ≤ 0.25 | consistency (rates fitted to these curves) |
| risk.lane_width | 41, 47, 60, 77 (fitted), 97 km/h at 2.85–3.75 m | 56, 61, 65, 77, 82 km/h; error ≤ 9.4: **fail** (12.3) | four held out |
| affect.emotion_study | RMSE 0.098, R² 0.84 | ≤ 0.13 (the original model's RMSE) | reproduction |

### Assumptions to know

- **Tactile latency.** The mapping of the vibration intensities of
  Diederich and Colonius (2004) to 10 and 30 dB above threshold is
  assumed, as are the DRT light's eccentricity (17°) and the DRT
  vibration's level (20 dB).
- **Take-over request.** The chime level (75 dB SPL at 1 kHz) follows ISO
  7731. The seat vibration (1 m/s² at 60 Hz) and the icon's position are
  assumed.
- **Trust.** A request lowers trust by the amount Kraus et al. (2020)
  measured after a first request. This is one data point; trust dynamics
  are not yet benchmarked.
- **Risk field.** The wheelbase is assumed. The threshold is refitted
  because the published value depends on an unreported grid spacing.
- **Unreachable sources.** Petermeijer et al.'s per-modality times, tactile
  TWIN parameters, the steering-wheel and backrest vibration thresholds,
  and Kolekar et al.'s code (4TU timed out) were out of reach.
