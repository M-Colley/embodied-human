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
