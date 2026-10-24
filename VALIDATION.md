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
