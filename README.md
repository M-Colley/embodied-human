# embodied-human

A modular simulated human for interaction and driving research. It covers
the senses (eyes, vision, hearing, touch, the vestibular sense and
proprioception), attention, memory, state estimation and responses, motor
control and fatigue, and experience and motivation (emotion, perceived
risk and trust). Each is a replaceable module with cited parameters, and a
benchmark scores the modules, alone and together, against published human
data.

The aim is convergence. Groups that model eye movements, visual attention,
memory, motor control or driver behaviour can improve one module, keep
everything else fixed, and show what changes in the integrated behaviour.

```
             scene / engine camera                 in-vehicle display (SuRT)
                      │                                       │
      ┌───────────────▼───────────────┐         ┌─────────────▼────────────┐
      │ eyes: latency, main sequence, │  gaze   │ vision: foveated image   │
      │ saccadic suppression          ├────────►│ objects: noise ∝ 1+e/E2  │
      └───────────────▲───────────────┘         └─────────────┬────────────┘
                      │ where to look                         │ observations
      ┌───────────────┴───────────────┐         ┌─────────────▼────────────┐
      │ attention: uncertainty sampler│◄────────┤ estimator: lane Kalman   │
      │ (Senders), SEEV, AttenD metric│ belief  │ (belief and uncertainty) │
      └───────────────┬───────────────┘ and SD  └─────────────┬────────────┘
                      │ task switching                        │ believed state
      ┌───────────────▼───────────────┐         ┌─────────────▼────────────┐
      │ memory: focus of 4 + ACT-R    │         │ motor: Fitts-timed       │
      │ (task progress across glances)│         │ minimum-jerk reaches;    │
      └───────────────────────────────┘         │ MyoSuite full-body driver│
                                                │ (via myosuite-carla)     │
      proprioception: spindle Ia rates,         └──────────────────────────┘
      joint sense, reflex delay (MuJoCo muscles)
```

## Install

```bash
pip install -e .                       # numpy, scipy: every module and module benchmark
pip install -e ".[experience]"         # scikit-learn (emotion), MoSQITo (loudness)
pip install -e ../embodied-ui          # touch and button tasks with physical contact
pip install -e ../myosuite-carla       # driving with the MyoSuite full body
```

## Use

```python
import embodied_human as eh

human = eh.build_human(seed=0)                          # the cited baseline
human = eh.build_human(eh.HumanConfig().replace("attention", "seev"))
for row in human.provenance():                          # every parameter's source
    print(row["parameter"], row["value"], row["kind"], row["citation"])
```

Each category has an abstract interface. Implement it and register the
implementation:

```python
from embodied_human import register
from embodied_human.memory import Memory

@register("memory", "interference")
class InterferenceMemory(Memory):
    def encode(self, key, value, time): ...
    def retrieve(self, key, time): ...
```

## Baseline modules

| Category | Baseline | Model and parameters | Sources |
| --- | --- | --- | --- |
| `eyes` | `main_sequence` | Log-normal latency, median 200 ms; duration 2.2 ms/deg + 21 ms; minimum-jerk velocity profile (peak 1.875 × mean); no vision during saccades | Carpenter (1988); Baloh et al. (1975); Saslow (1967) |
| `vision` | `foveated` | Per-pixel blur from the contrast-threshold cutoff f_c(e) = e2 ln(1/CT0) / (α(e + e2)), α = 0.106, e2 = 2.3°, CT0 = 1/64, blended across blur levels | Geisler & Perry (1998) |
| `objects` | `eccentricity_noise` | Observation SD ∝ 1 + e/E2, E2 = 2° | Strasburger, Rentschler & Jüttner (2011) |
| `attention` | `uncertainty_sampler` | Look back at the road when the believed lateral SD exceeds a threshold drawn anew for each glance (log-normal, median 0.11 m, log-SD 1.0, fitted by ABC); leave after 0.3 s on the road; at most 4 s away. Records why gaze returned. Alternatives: `seev` (Wickens et al., 2003) and the AttenD buffer as a metric (Kircher & Ahlström, 2009) | Senders et al. (1967); fitted to Kujala & Saariluoma (2011) |
| `memory` | `act_r` | Focus of 4 chunks over ACT-R base-level memory (d = 0.5, τ = 0, s = 0.25, F = 1) | Cowan (2001); Anderson et al. (2004) |
| `proprioception` | `spindle` | Ia rate 4.3 v^0.6 + 2 d + r0 (r0 = 10 imp/s); 3° joint position error; 25 ms delay | Prochazka & Gorassini (1998); Malik et al. (2016); Talis & Levik (2016); Pruszynski et al. (2009) |
| `motor` | `ffitts_min_jerk` | MT = 147.7 + 105.0 log2(A/W + 1) ms; endpoint SD² = (W/4.133)² + σa², σa = 1.5 mm; minimum-jerk path | Bi, Li & Zhai (2013); Soukoreff & MacKenzie (2004); Flash & Hogan (1985) |
| `estimator` | `lane_kalman` | Kalman filter on lateral offset and heading with known speed and yaw rate; heading process noise sets how fast uncertainty grows | Senders et al. (1967) account; parameters assumed |
| `hearing` | `psychoacoustic` | Threshold in quiet (ISO 226); masking by noise in the critical band, K = 0.4; car cabin noise by speed; latency by Piéron's law (k = 7.31 ms, n = −0.33); ISO 7731 audibility; loudness of waveforms with MoSQITo | ISO 226:2003/2023; Zwicker & Terhardt (1980); Scharf (1970); Kohfeld et al. (1981) |
| `touch` | `vibrotactile` | Vibration thresholds by frequency for gripping hands, hands on handles and plates, and seats; felt only through contact | Brisben et al. (1999); Morioka & Griffin (2006); Miwa (1967) |
| `vestibular` | `subjective_vertical` | 6-DOF subjective vertical conflict model: otoliths, canals and their internal model; conflict to motion sickness incidence. Alternative: `iso_2631` (Wf-weighted motion sickness dose) | Bos & Bles (1998); Wada, Fujisawa & Doi (2018); ISO 2631-1 |
| `response` | `twin` | Time-window-of-integration model: channels race, signals within 359 ms integrate and save 20 ms; visual latency grows with eccentricity | Diederich & Colonius (2015); Osaka (1976) |
| `fatigue` | `three_compartment` | Active, fatigued and resting motor units per joint; recovery 15-30× faster at rest; Consumed Endurance for comparison | Xia & Frey-Law (2008); Frey-Law et al. (2012); Looft et al. (2018); Hincapié-Ramos et al. (2014) |
| `affect` | `appraisal` | Q-learning values appraised (suddenness, relevance, conduciveness, power) and recognised as emotions by linear SVMs | Zhang, Hilpert, Broekens & Jokinen (2024); Scherer (2001) |
| `risk` | `risk_field` | Driver's risk field along the predicted path; speed and lane keeping keep perceived risk below a threshold | Kolekar, de Winter & Abbink (2020) |
| `trust` | `decision_field` | Beliefs in automation and own capability with inertia; take-over and hand-back with hysteresis | Gao & Lee (2006) |

[docs/parameters.md](docs/parameters.md) lists every parameter as measured,
standard, fitted, derived or assumed. Assumed values are modelling choices
to be fitted or replaced, and the benchmark tells you which matter.

## Tasks

| Task | Notes |
| --- | --- |
| **SuRT** (ISO/TS 14198:2019) | Visual-manual search task. It runs as an embodied-ui engine that draws its display and scores touches. The standard's geometry is not public, so all dimensions are parameters |
| **DRT** (ISO 17488:2016) | 3–5 s intervals, 1 s stimuli, hits 100–2500 ms after onset, misses, premature and repeated responses. The lamp is an embodied-ui `Indicator`; responses are presses of a `PushButton` |
| **Delayed digit recall** (n-back) | One digit every 2.25 s at levels 0/1/2, answered from the memory module |
| **ISO 9241-411 tapping** | Effective width, amplitude, IDe and throughput after Soukoreff & MacKenzie (2004) |

### Integrated agents

- **`agents.pointer`:** a touchscreen user. The eyes saccade to each target,
  and an embodied-ui fingertip follows the motor module's plans. Every
  selection is a physical contact on an embodied-ui touch surface.
- **`agents.driver`:** the attentive driver, which drives myosuite-carla's
  MyoSuite full body through its muscles.
  - It sees the road only through eyes, objects and estimator. Its two-point
    steering and IDM controller act on the belief, not the true state.
  - Attention shares gaze with SuRT. Search progress survives road glances
    only if memory retrieves it.
  - A take-over request is a chime, a seat vibration and/or an icon. The
    chime is heard over the cabin noise at the current speed, the vibration
    felt through the seat, and the icon seen at its eccentricity from where
    the driver is looking; the response module combines them. The driver
    then drops the task, returns gaze and hand, and assesses the scene; the
    request also lowers trust. The take-over time emerges from these steps.
  - The log keeps the vehicle's motion for the vestibular module, so the
    same run predicts motion sickness.

## Simulated participants, inference and studies

```python
from embodied_human.population import BASELINE_VARIATIONS, OLDER_ADULTS, Population
from embodied_human.inference import Prior, abc_rejection
from embodied_human.agents.road import simulate_glances

people = Population(variations=BASELINE_VARIATIONS).participants(20, seed=0)
older = Population(profile=OLDER_ADULTS).typical()      # a group shift, with its source
log = simulate_glances(eh.build_human(people[0], seed=0), speed=13.9, seconds=120)
log.glance_metrics()["returns_by_reason"]               # why gaze went back to the road
log.write_csv("trace.csv")                              # gaze, belief and truth per step
```

- **Individual differences.** `Population` draws participants by varying
  module parameters between people and shifting whole groups. Every
  spread carries a source; only joint position sense is measured so far,
  and the other spreads are labelled assumptions.
- **Inversion.** `abc_rejection` fits any module's parameters to human
  summary data by approximate Bayesian computation with Sobol prior draws.
  It returns a posterior, not just a point estimate.
  `examples/calibrate_glances.py` fits the attention module.
- **Explanations and traces.** The driver records why each glance ended,
  splits take-overs into response, hand, eyes and situation assessment,
  and exports its full trace.
- **Fast glance studies.** `agents.road.StraightRoad` simulates glances
  without vehicle dynamics or MyoSuite, about 0.5 s per simulated
  minute. This is possible because the belief's uncertainty depends only
  on speed and on when the road was last seen.

[docs/seminar-requirements.md](docs/seminar-requirements.md) maps the
requirements for computational user models discussed at Dagstuhl Seminar
26432 to these features, and lists what is still missing.

## Benchmark

```bash
python -m embodied_human.bench --out results.json
python -m embodied_human.bench --replace attention=seev --out seev.json
```

Results of the baseline (seed 0; [report](evidence/benchmark-baseline.json)).
Eleven of eighteen benchmarks pass:

| Benchmark | Measured | Human reference (acceptance range) | Kind of check |
| --- | --- | --- | --- |
| `eyes.main_sequence` | 2.2 ms/deg, 21 ms | 2.2–2.7 ms/deg (2.0–3.7); 21 ms (13–37) | consistency |
| `eyes.latency` | median 201 ms | 150–250 ms (Saslow, 1967) | consistency |
| `objects.peripheral_acuity` | E2 = 1.98° | 1.0–2.6° (letters, Landolt C) | consistency |
| `vision.foveation` | E2 = 2.3° | 1.0–3.0° | consistency |
| `memory.span` | 4 items | 3–5 chunks (Cowan, 2001) | consistency |
| `motor.touch_tapping` | 6.28 bits/s; 104 ms/bit | 6.95 bits/s (5.5–8.5; MacKenzie, 2015); 105 ms/bit (75–135; Bi et al., 2013) | **through physical contact** |
| `driving.surt_glances` | Full body at 72 km/h: mean off-road glance 0.93 s; 2.2% over 2 s | NHTSA (2013) criteria: pass. Human in-vehicle glances, mean 0.8–1.2 s: pass; 3–13% over 2 s: just below (score 0.98) | **held out** (fitted on a straight road at 50 km/h) |
| `driving.takeover` | chime and icon: mean 1.51 s, SD 0.18 s (8 runs); median response 0.22 s, eyes 0.52 s, hand 0.63 s, driving 1.47 s | 2.72 ± 1.45 s across 520 conditions (Zhang et al., 2019) | **emergent** |
| `driving.takeover_modality` | median response: sound 0.27 s, vibration 0.31 s, light 0.33 s; hand on the wheel after chime and icon 0.65 s | sound and touch faster than light (Petermeijer et al., 2017): pass. Grabbing the wheel 1.87 s (1.36–2.38; Ko et al., 2022): **fail** | **emergent** |
| `response.modality_rt` | tone 252 ms < light 319 ms < vibration 335 ms | same order (Diederich & Colonius, 2004) | order only |
| `response.drt` | visual DRT 349 ms; tactile 352 ms; older drivers 1.33× slower | visual 326–362 ms (Conti et al., 2014): pass; tactile 526 ms (482–570; Cooper et al., 2020): **fail**; ageing 1.07–1.36×: pass | **emergent** |
| `vestibular.vertical_sickness` | RMS error 19.7 % points; r = 0.92 over 14 conditions | 2 h vertical oscillation (O'Hanlon & McCauley, 1974): error ≤ 10: **fail**. With `--replace vestibular=iso_2631`: 6.1 (consistency) | held out |
| `fatigue.endurance` | median error 17% | endurance times per joint (Frey-Law & Avin, 2010) | consistency |
| `risk.lane_width` | 41–97 km/h for 2.85–3.75 m lanes; order right (tau 1.0); mean error 12.3 km/h | 56–82 km/h (Liu et al., 2016), error ≤ 9.4: **fail** | threshold fitted on one lane, four held out |
| `affect.emotion_study` | RMSE 0.098, R² 0.84 | normalised emotion ratings in three tasks (Zhang et al., 2024); the original model: 0.13, 0.78 | **reproduction** |
| `driving.glance_population` | 20 participants: mean 1.13 s (95% CI 1.08–1.19), within-person SD 0.58 s (0.54–0.64), 7.9% over 2 s | Kujala & Saariluoma (2011): 1.06–1.20 s, 0.48–0.66 s, 5.7–10% | fitted (mean, SD); held out (long glances) |
| `driving.glance_speed` | elasticity of glance duration on speed −0.45 (40–120 km/h) | occlusion −0.72 to −0.53 (Senders et al., 1967); in-car search 0 (Jokinen et al., 2021): **fail both** | **counterfactual** |
| `design.touch_key_size` | gains stop at 12.5 mm (30 simulated participants, paired t-tests as in the study) | 17.5 mm (12.5–22.5; Kim et al., 2014) | **application**, physical contact |

**How to read these results.** A *consistency* check confirms that a module
behaves as its cited parameters say. It becomes informative when a module is
replaced. *Emergent* results come from the interplay of modules and are the
real test:

- **Glances.** The attention module was fitted to two numbers: the mean
  and the within-person SD of in-vehicle glances (Kujala & Saariluoma,
  2011). Held-out checks:
  - **Long-glance share:** matches.
  - **Full-body drive at 72 km/h:** passes.
  - **Longest glances:** too long, 4.2 s against 2.4–3.7 s in humans.
  - **Speed:** the effect of speed lies between two conflicting human
    results. Glance timing by road uncertainty alone fits neither the
    occlusion data nor in-car search, which looks unaffected by speed.
    [docs/seminar-requirements.md](docs/seminar-requirements.md) discusses
    what this implies for alternative attention modules.
- **Take-over.** The take-over time lies within one SD of the meta-analytic
  mean, but it is shorter than that mean. The order of the senses is right,
  but the model grabs the wheel almost three times faster than Ko et al.'s
  drivers. In the model the hand starts back as soon as the request is
  noticed; people evidently first stop the game, look and decide.
- **Senses and responses.** The visual DRT and the slowing of older
  drivers match. The tactile DRT is 170 ms too fast: the model has no cost
  of driving itself on the central stage, which Conti et al. measured as
  about 50 ms for the visual DRT.
- **Motion sickness.** The subjective vertical conflict model, with
  parameters fitted to car passengers, ranks the 14 vertical conditions
  well (r = 0.92) but predicts a third of the observed incidence. The ISO
  2631-1 dose was partly derived from these data, so its fit (6.1) is a
  consistency check.
- **Perceived risk.** The risk field orders the lane widths correctly but
  reacts about twice as strongly as drivers. Its parameters were fitted to
  one driver; the paper's threshold could not be used directly because it
  depends on an unreported grid spacing.
- **Emotion.** The reimplemented appraisal model reproduces the published
  appraisal profiles and fits the human ratings at least as well as the
  original.
- **Design.** The key-size study puts the point of no further gain at
  12.5 mm, one size below the human 17.5 mm and within the acceptance
  range. The simulated participants were not driving.

Many of the integrated agent's parameters are assumed (see docs/parameters.md)
and are the first candidates for fitting.

## Relation to other work

The new senses, responses, fatigue, emotion, risk and trust modules
reimplement published models; none of their code is copied. Code that was
released without a licence (Zhang et al.'s emotion model, TouchSim, Jang et
al.'s fatigue model) was not used. MIMo (MIT) models full-body touch in
MuJoCo, but its pinned dependencies (MuJoCo 3.3, NumPy 2.0) conflict with
this package, so its approach is a candidate for a `touch` module rather
than a dependency.

- **User-in-the-Box** (Ikkala et al., 2022) couples MuJoCo biomechanics,
  perception and RL for interaction tasks.
- **Computational rationality** (Oulasvirta et al.) underlies the model of
  Jokinen, Kujala & Oulasvirta (2021). It keeps a belief over lane position
  that spreads while the eyes are on an in-car display, and learns when to
  switch with hierarchical reinforcement learning. Its human data (about
  1.0 s per in-car glance) are part of this benchmark. Its supervisor is a
  natural replacement for the `uncertainty_sampler`.
- **atari-cr and SugaRL** add human-like active vision to RL agents.
  `FoveatedVision` with `MainSequenceEyes` provides that sensor for any
  image source, and embodied-ui's CARLA and Unity cameras supply driver
  views.

No code from these projects is included.

## Limitations and next steps

- **Motor:** touch reaches use an idealized fingertip; a MyoSuite arm
  reaching controller is the obvious replacement. The full-body driver
  operates only the controls of its scripted motor programs.
- **Driving perception:** the attentive driver uses object-level vision.
  Pixel-level vision in the loop (a CARLA or Unity camera through
  `FoveatedVision` into a learned perception module) is supported by the
  interfaces but not yet trained.
- **Individual differences:** only one between-person spread (joint
  position sense) is measured. The older-adult profile also slows the
  central response stage and widens the integration window (Diederich et
  al., 2008); other shifts are still missing.
- **Senses:** touch has thresholds for vibration only, at five sites; full
  skin with receptor types (TouchSim, once licensed) or sensor density per
  body part (as in MIMo) would replace it. Hearing handles tones in
  broadband noise, not speech or spatial sound.
- **Load:** the response module has no cost of driving or of a secondary
  task, so DRT times under load are not yet predicted (Conti et al.: about
  +290 ms under 2-back, +400 ms under SuRT).
- **Next benchmarks:**
  - Occlusion distance on highways: mean 15.5 m (Kujala et al., 2016).
  - Take-over times by time budget, which still need a verified source.
  - Trust dynamics after take-overs and malfunctions (Kraus et al., 2020),
    now that the trust module exists.

## Development

```bash
python -m pytest tests -q
python -m ruff check src tests examples
python -m mypy src/embodied_human --ignore-missing-imports
python examples/calibrate_glances.py --out evidence/glance-calibration.json
```

Apache-2.0.
