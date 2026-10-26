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

## 2026-10-25 (later): load, take-over decisions, refits, trust, body sensing

Same PC and versions. These changes address the failures listed above.

### Tests

`python -m pytest tests -q` passes 63 tests ([JUnit](evidence/tests.xml)).
New tests check:

- the shared central stage: a random arrival waits d²/2T, reproduced to
  within 2 ms;
- the DRT under driving and 2-back load;
- the take-over decision stage: the hand starts only after the eyes have
  been on the road for the decision time;
- the lane-width refit and the trust event sequences;
- the MyoSuite body's head motion and muscle fatigue (gravity felt as
  9.81 m/s² along the head's z axis).

A fresh Python 3.10 environment, installed as the CI workflow installs it,
passed the suite without MyoSuite: 52 passed, 2 skipped.

### Fits

```bash
python examples/fit_motion_sickness.py --out evidence/motion-sickness-fit.json
python examples/fit_drt_load.py --out evidence/drt-load-fit.json
```

- **Motion sickness.**
  - **Fit:** the conflict scale only, as Bos and Bles (1998) fitted one
    gain, to the 14 two-hour conditions. It went from 0.5 to 0.196 m/s²
    (SciPy bounded search, 9 evaluations, 229 s).
  - **Fit set:** RMSE 19.7 → 7.5.
  - **Held out:** 43 time-course points at nine other conditions. RMSE
    28.6 → 14.8, r = 0.76; McCauley et al.'s own descriptive model scores
    6.3 there.
- **DRT load.**
  - **Fit:** chunk lengths fitted to Conti et al.'s pooled visual DRT
    increases. The 0-back takes 0.743 s per digit, the 2-back 1.115 s and
    SuRT 1.771 s per screen.
  - **SuRT gaze:** the attentive driver had its eyes on the display 56% of
    the time during SuRT.
- **Take-over decision.**
  - **Fit:** a decision dwell of 0.91 s reproduces Ko et al.'s 1.87 s to
    the wheel; each 0.1 s of dwell adds 0.1 s.
- **Risk field.**
  - **Fit:** threshold and width growth, by least squares on the 3.0 and
    3.5 m lanes.
  - **Result:** width growth 0.005 per metre, against 0.001 published.
- **Trust.**
  - **Fit:** capabilities revealed by an overtake, a request and a
    malfunction, and the opacity, all on Study 1 (24 points). The best fit
    gives the malfunction no dip (capability 0.89 against 0.82 for an
    overtake).

### Benchmark

```bash
python -m embodied_human.bench --out evidence/benchmark-baseline.json
python -m embodied_human.bench --replace vestibular=iso_2631 --only vestibular.vertical_sickness vestibular.sickness_time_course --out evidence/benchmark-vestibular-iso2631.json
```

The run took 151 s; **13 of 21 benchmarks pass**. Changes from the
previous run:

| Benchmark | Before | Now | Human reference |
| --- | --- | --- | --- |
| driving.takeover (MyoSuite) | 1.51 s | 1.94 s, SD 0.13; hand 1.85 s | 2.72 ± 1.45 s; Gold et al. 2.10–2.89 s |
| driving.takeover_modality | hands 0.65 s: fail | hands 1.85 s (fitted): pass | 1.87 s |
| response.drt_load (new) | — | driving +28 ms; 2-back +284 ms; SuRT +382 ms | +51 ms (31–70); +234 ms (152–316): pass; +177 ms (104–250): **fail** |
| vestibular.vertical_sickness | 19.7: fail | 7.5: pass (fitted) | ≤ 10 |
| vestibular.sickness_time_course (new) | — | 14.8: **fail** (ISO swap 12.4) | ≤ 10 |
| risk.lane_width | 12.3 km/h: fail | 2.1 km/h: pass | ≤ 9.4 |
| trust.kraus_dynamics (new) | — | RMSE 0.29: pass; dip +0.20: **fail** | ≤ 0.3; −0.47 |

**Body.** The full-body SuRT drive now reports what the body felt over
60 s at 72 km/h:
- motion sickness incidence 0.005%;
- 1.3% of elbow motor units fatigued, 2–4% of hand motor units, and 0.2%
  of ankle motor units.

### Interpretation

- **Driving cost.** It is a real prediction: threaded cognition's
  published driver gives +28 ms against +51 ms measured.
- **Cooper et al.** Their 2-back cost is predicted. Their SuRT cost is
  not, but their SuRT, stimuli and scoring differ from Conti et al.'s (see
  the README).
- **Trust.** The failure is structural. Gao and Lee's model has inertia in
  both of its updates, so it cannot show a sharp dip that recovers by the
  next event.

## 2026-10-25 (evening): thresholds, saccades and take-over components from four papers

Same PC and versions. Four papers were read in full:

- Morioka and Griffin (2005, 2008), for vibration thresholds;
- Diederich and Colonius (2007), for visual-tactile saccades;
- Gold et al. (2013), for take-over components.

Their numbers replace values that were read from figures or quoted from
secondary sources.

### Changes

- **Touch.** The thresholds now come from two sources:
  - Morioka and Griffin's (2008) Table 2: the hand on a handle, the seat
    and the foot, each in three axes;
  - their 2005 Fig. 2: fingertip, palm and grip, read from the PDF's
    vector data.

  They replace values read from figures in Brisben et al. (1999), Miwa
  (1967) and Morioka and Griffin (2006). The seat's vertical threshold at
  10 Hz rises from 0.0078 to 0.022 m/s² r.m.s.
- **Take-over.** The take-over reaction now waits for the central stage
  held by SuRT. This is the same threaded-cognition load as in
  `response.drt_load`, with SuRT's chunk fitted to Conti et al. The driver
  also logs Gold et al.'s gaze reaction: the onset of the first saccade
  from the task display to anywhere else.

### Tests

`python -m pytest tests -q` passes 65 tests ([JUnit](evidence/tests.xml)).
New tests check:

- the threshold orderings both papers report: the seat is most sensitive
  vertically at 8–80 Hz and the hand above 100 Hz; the fingertip is the
  most sensitive site at 8 Hz and the least sensitive at 125 Hz;
- the central-stage wait and the gaze reaction in take-overs;
- the eccentricity benchmark.

The fresh Python 3.10 environment without MyoSuite passed 59, skipped 2,
and mypy found no issues.

### Fit

```bash
python examples/fit_takeover.py --out evidence/takeover-fit.json
```

With the wait added, a decision time of 0.78 s (down from 0.91 s)
reproduces Ko et al.'s median 1.87 s to the wheel (model 1.87 s, 12 runs,
10 ms grid).

### Benchmark

```bash
python -m embodied_human.bench --out evidence/benchmark-baseline.json
```

The run took 141 s; **16 of 23 benchmarks pass**. The previous section
reported 13 of 21, but its report shows 15 of 21. Changes:

| Benchmark | Before | Now | Human reference |
| --- | --- | --- | --- |
| response.eccentricity (new) | — | +40 and +59 ms at 50° and 70°: RMS 1 ms, pass | +39, +58 ms (Diederich & Colonius, 2007, Table 8), ≤ 10 ms |
| driving.takeover_components (new) | — | gaze 0.48 s, road 0.73 s: pass; hands 1.94 s, input 1.99 s: **fail** | Gold et al. (2013) 5–7 s budgets: 0.41–0.54, 0.69–0.94, 1.45–1.79, 2.10–2.89 s |
| driving.takeover (MyoSuite) | 1.94 s, SD 0.13 | 2.24 s, SD 0.62 | 2.72 ± 1.45 s |
| driving.takeover_modality | hands 1.85 s | hands 1.87 s (refitted) | 1.87 s |

The previous model, without the wait and with a decision time of 0.91 s,
scores gaze 0.39 s, road 0.43 s, hands 1.76 s and input 1.80 s on
`driving.takeover_components`. Only its hands-on time falls in Gold et
al.'s range.

### Interpretation

- **Eccentricity.** Osaka's (1976) manual reaction times set the increase
  up to 50°, and the model extrapolates beyond. The increase carries over
  to saccades, as TWIN's response-independent first stage implies.
- **Take-over.** Nothing in the model was fitted to Gold et al. Load
  fitted on the DRT predicts when the eyes leave SuRT and reach the road.
  The model then reaches for the wheel late and acts at once. Gold et
  al.'s drivers held the wheel 0.65–1.10 s before choosing to brake or
  swerve, and the straight road gives the model no hazard to choose for.
  The model has no time budget, so it cannot reproduce faster reactions
  with less time.
- **Tactile latency.** Diederich and Colonius give their vibration in volts,
  so their tactile estimates (29 ms at 50 Hz, 111 ms at 5 Hz, against
  16 ms for a light at 10°) cannot calibrate the latency curve.
  - At 50 Hz they agree with the model if the vibration was about 18 dB
    above threshold.
  - No felt vibration in the model is as slow as their 5 Hz one: the
    model's slowest is 66 ms behind the light, theirs 95 ms.

## 2026-10-26: DRT benchmark measured while driving

Same PC and versions. `response.drt` compared a model without any driving
load against DRT times both studies measured while driving. It also
placed the head-mounted light at 17°, where `response.drt_load` uses ISO
17488's 20° left and 10° up (22.4°). The benchmark now uses
`response.drt_load`'s procedure: driving holds the central stage, the
light is at the ISO position, and Cooper et al.'s vibration and windshield
light are pooled, as they pooled them. The reference
`response.drt_tactile` is renamed `response.drt_on_road`, because
Cooper et al.'s 526 ms is not tactile-only.

`python -m pytest tests -q` passes 66 tests. The new one checks that the
benchmark includes the driving load.

```bash
python -m embodied_human.bench --out evidence/benchmark-baseline.json
```

The run took 146 s; **16 of 23 benchmarks pass**, as before.
`response.drt` was already failing. It now fails on both absolute times:

| Measure | Before | Now | Human reference |
| --- | --- | --- | --- |
| Head-mounted light while driving | 350 ms: pass | 385 ms: **fail** (353 ms without driving) | 344 ms (326–362); 293 ms without driving (Conti et al., 2014) |
| On the road, vibration and light pooled | 348 ms (tactile only): fail | 368 ms: **fail** | 526 ms (482–570; Cooper et al., 2020) |
| Older over younger | 1.34: pass | 1.32: pass | 1.215 (1.07–1.36) |

The visual pass came from leaving out the driving load and using a
smaller eccentricity. The cost of driving itself is close (+32 ms against
+51 ms), but the model answers Conti et al.'s light about 60 ms too slowly
even without driving. Cooper et al.'s on-road times are 158 ms slower than
the model's; their colour-changing light reflected in the windshield and
real traffic are not modelled.

## 2026-10-26 (later): parallel motor readiness, Zeeb et al., Kamiji et al., steering wheel

Same PC and versions. Five more papers were uploaded:

- **Zeeb, Buchner and Schrauf (2015):** used.
- **Kamiji et al. (2007):** used.
- **Morioka and Griffin (2007), steering wheel:** used.
- **Morioka and Griffin (2008, JSV):** a second copy of a paper already in
  use. All 165 table values agree with the stored thresholds.
- **Bharitkar et al. (2004, JASA):** a room-equalisation paper, not one of
  the requested sources. The intended one was probably Morioka and Griffin
  (2008, JASA), doi:10.1121/1.2935002.

### Changes

- **Take-over structure.** Zeeb et al. argue, and Gold et al.'s data show,
  that the hands return to the wheel reflexively and in parallel with the
  gaze, and that the first input waits for the situation to be processed.
  The driver had the hand wait for a decision dwell on the road scene.
  - **Now:** the hand leaves the task `hand_release` seconds after the
    reaction. Driving still waits for the hand and for the assessment of
    the road.
  - **Refit:** `hand_release` replaces `decision_dwell`. It is fitted to
    the midpoint of Gold et al.'s hands-on times (1.62 s) on their SuRT
    task, which is the simulated driver's task. This gives 0.64 s (10 ms
    grid, 16 runs at 120 km/h).
  - **Ko et al.:** their laptop game is a different task, so their
    hands-on time is now held out.
- **Vestibular.**
  - **Sources:** the parameters now cite Kamiji et al. (2007), Table 1,
    whose values are those already used.
  - **New method:** `SubjectiveVertical.conflict` returns the conflict
    over time.
- **Touch.** A `steering_wheel` site comes from the median thresholds in
  Morioka and Griffin (2007), Fig. 3:
  - read from the PDF's vector data;
  - the three panels draw the same curve and agree within 3%.

### Tests

`python -m pytest tests -q` passes 67 tests. New or changed tests:

- the hand starts one release time after the reaction, whatever the eyes
  are doing, and driving waits for both hand and assessment;
- with rotation and the internal acceleration model switched off, the
  vertical conflict follows Kamiji et al.'s closed form
  s/(τs² + s + K_vc)·a. The match is within 0.5% at 0.05, 0.2 and 0.5 Hz;
- the steering wheel is most sensitive near 125 Hz.

### Fit

```bash
python examples/fit_takeover.py --out evidence/takeover-fit.json
```

`hand_release` = 0.64 s gives a mean hands-on time of 1.63 s against the
1.62 s target.

### Benchmark

```bash
python -m embodied_human.bench --out evidence/benchmark-baseline.json
```

The run took 146 s; **16 of 25 benchmarks pass**. Both new benchmarks fail.

| Benchmark | Before | Now | Human reference |
| --- | --- | --- | --- |
| driving.takeover_components | gaze 0.48, road 0.73, hands 1.94, input 1.99 s | gaze 0.48, road 0.73, hands 1.63 (fitted), input 1.81 s: **fail** on input | 0.41–0.54, 0.69–0.94, 1.45–1.79, 2.10–2.89 s |
| driving.takeover_modality | hands 1.87 s (fitted) | hands 1.52 s (held out): pass | 1.87 s (1.36–2.38) |
| driving.takeover_motor_readiness (new) | — | road 0.74 s: pass; hands 0.97 s: **fail** | 0.69 s (0.64–0.74); 1.14 s (1.05–1.23) (Zeeb et al.) |
| driving.automation_monitoring (new) | — | longest look away 1.8 s; road 36%: **fail** | 8.6–44.4 s; 2.2–12.5% (Zeeb et al., Table 1) |
| driving.takeover (MyoSuite) | 2.24 s, SD 0.62 | 2.03 s, SD 0.69: pass | 2.72 ± 1.45 s |

### Interpretation

- **Motor readiness.** With the hand released at the reaction, hands-on
  times line up across tasks by manual load: Zeeb et al.'s knob (1.14 s),
  Gold et al.'s SuRT (1.45–1.79 s) and Ko et al.'s laptop (1.87 s). Even
  with no release time, the model's knob take-over is 0.17 s faster than
  Zeeb et al.'s. They detected contact from steering-wheel motion, which
  probably lags the touch.
- **Processing.** The first input comes 1.08 s after the eyes reach the
  road; Gold et al.'s drivers needed 1.4–1.95 s. Zeeb et al.'s groups
  braked at 1.63, 1.86 and 2.31 s after longest looks away of 8.6, 20.8
  and 44.4 s. That is about 19 ms more per second away from the road
  (derived; it is linear across the three groups). Linking the number of
  assessment fixations to what memory lost while the eyes were away would
  turn this into a prediction.
- **Monitoring.** During automation the model glances as if it were
  steering. Zeeb et al.'s drivers watched the road far less, and how much
  they did predicted their braking time.
- **Not used.**
  - Kamiji et al.'s improved parameter set (Table 2) adds a second-order
    learning filter. It also changes predictions in a way the authors
    describe as worse at low and high frequencies.
  - The steering-wheel paper's rate of growth of sensation (Fig. 2) and
    its equivalent comfort contours are not used; the touch module models
    detection only. The journal version (Morioka and Griffin, 2009,
    Applied Ergonomics, doi:10.1016/j.apergo.2008.06.001) may tabulate
    them.

## 2026-10-26 (evening): tutorial, README, module swaps

Same PC and versions.

- **Bug.** The driver read its threshold for taking over control from the
  baseline attention module's `leave_sd`. Any other attention module, such
  as SEEV, therefore crashed during a take-over. The driver now has its
  own `engage_sd` (0.15 m, the same value), so baseline results are
  unchanged. A test runs a take-over with SEEV.
- **Command line.** `python -m embodied_human.bench --import module` loads
  modules that register new implementations before `--replace` names
  them.
- **Tutorial.** `examples/tutorial_modules.ipynb` changes parameters, swaps
  implementations, changes an agent parameter, and writes, registers,
  tests and benchmarks a new attention module. It also adds a reference
  and a benchmark.
  - **Test:** `tests/test_tutorial.py` executes it with nbclient. It needs
    the `test` extra and is skipped without it.
  - **Run:** in a fresh Python 3.10 environment it ran in 27 s. That
    environment passed 63 tests and skipped 2 (MyoSuite).
- **README.** Two new sections, *What is included* and *What was
  replicated*.
  - **Models:** 15 reimplemented models, each with the check against its
    publication.
  - **Human results:** 27 published results, each marked replicated,
    fitted or not replicated.

## 2026-10-26: Python 3.14

Fresh Python 3.14.4 environment (uv): NumPy 2.5.4, SciPy 1.18.1,
MuJoCo 3.17.0.

- **As in CI.** With embodied-ui from GitHub and
  `pip install -e ".[test,lint,experience]"`, tests, ruff, mypy and the
  build all pass. Only the MyoSuite tests are skipped.
- **With MyoSuite.** With the local MyoSuite `dev` checkout and
  myosuite-carla added, all 69 tests pass and none are skipped. The
  warnings are the same MuJoCo spec-prefix warnings as on 3.12.
- **Hidden skip.** MoSQITo imports matplotlib without declaring it, so
  the loudness test had been skipped in every environment, CI included.
  - **Fix:** matplotlib is now part of the `experience` and `test` extras.
  - **Test:** it now skips only when MoSQITo is absent, so a broken
    install fails.
- **CI.** The workflow now also runs on Python 3.14.
