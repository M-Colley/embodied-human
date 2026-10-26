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

## What is included

- **Modules.** Sixteen categories:
  - senses: `eyes`, `vision`, `objects`, `hearing`, `touch`, `vestibular`
    and `proprioception`;
  - cognition: `attention`, `memory`, `estimator` and `response`;
  - action: `motor` and `fatigue`;
  - experience and motivation: `affect`, `risk` and `trust`.

  Each has an abstract interface and a cited baseline. Two have
  alternatives: SEEV for attention and the ISO 2631-1 dose for motion
  sickness.
- **Sources for every parameter.** 110 parameters: 12 measured, 5 set by
  standards, 52 fitted, 4 derived and 37 assumed. See
  [docs/parameters.md](docs/parameters.md).
- **Tasks.** SuRT (ISO/TS 14198), the detection-response task (ISO
  17488), delayed digit recall and ISO 9241-411 tapping.
- **Agents.**
  - a touchscreen user;
  - an attentive driver that shares gaze with SuRT and takes over from
    automation. It runs on a fast straight road or on the MyoSuite full
    body through myosuite-carla.
- **Simulated studies.** Simulated participants, parameter inference by
  approximate Bayesian computation, bootstrap intervals, and traces of
  every run.
- **Benchmark.** 25 benchmarks score the model against 42 published human
  references. Any module can be swapped from the command line. Reports
  and the fitting scripts' outputs are in [evidence/](evidence/).
- **Tutorial.** [examples/tutorial_modules.ipynb](examples/tutorial_modules.ipynb)
  shows how to change parameters, swap implementations, and write,
  register, test and benchmark a new module.

The package contains only published aggregates of human data, and no
code from the projects listed under [Relation to other work](#relation-to-other-work).

## What was replicated

Here, *replicated* means rebuilt from a publication and checked against
it. There are two kinds.

### Published models, reimplemented from their equations

| Model | Module | Checked against the publication |
| --- | --- | --- |
| Saccade main sequence and log-normal latency (Carpenter, 1988; Baloh et al., 1975) | `eyes` | duration slope and intercept, median latency |
| Foveated resolution (Geisler & Perry, 1998) | `vision` | resolution halves at E2 = 2.3° |
| ACT-R base-level memory (Anderson et al., 2004) with a focus of four (Cowan, 2001) | `memory` | activation, retrieval and latency equations; span of 4 |
| Fitts' law with minimum-jerk paths (Bi, Li & Zhai, 2013; Flash & Hogan, 1985) | `motor` | throughput and slope in ISO tapping through physical contact |
| Uncertainty sampling (Senders et al., 1967) with a Kalman belief; SEEV (Wickens et al., 2003); AttenD (Kircher & Ahlström, 2009) | `attention`, `estimator` | thresholds, covariance growth, SEEV probabilities, AttenD timing |
| Muscle spindle Ia rates (Prochazka & Gorassini, 1998) | `proprioception` | rates on a MuJoCo muscle |
| Thresholds in quiet and masking (ISO 226; Zwicker & Terhardt, 1980), Piéron latency (Kohfeld et al., 1981), MoSQITo loudness | `hearing` | threshold table, A-weighting, 4 sone at 60 phon |
| Time window of integration (Diederich & Colonius, 2015) with a central stage shared as in threaded cognition (Salvucci & Taatgen, 2008) | `response` | redundant-signal facilitation; waiting time d²/2T within 2 ms |
| Six-degree-of-freedom subjective vertical conflict (Bos & Bles, 1998; Kamiji et al., 2007; Wada et al., 2018); ISO 2631-1 dose | `vestibular` | Kamiji et al.'s closed-form band-pass within 0.5%; Wf weighting and dose within 1% |
| Three-compartment fatigue (Xia & Frey-Law, 2008; Looft et al., 2018); Consumed Endurance (Hincapié-Ramos et al., 2014) | `fatigue` | endurance times per joint, median error 17% |
| Appraisal-based emotion (Zhang et al., 2024) | `affect` | the published appraisal profiles; ratings fitted at least as well as the original (RMSE 0.098 against 0.13) |
| Driver's risk field (Kolekar et al., 2020) | `risk` | closed form of lane risk |
| Extended decision field theory of trust (Gao & Lee, 2006) | `trust` | inertia and hysteresis |
| Vibration thresholds (Morioka & Griffin, 2005, 2007, 2008) | `touch` | tables, and vector data of figures; the orderings the papers report |
| Take-over process (Zeeb et al., 2015) | driver agent | the structure: hands return in parallel with the gaze, and action waits for the scene to be assessed |

### Published human results

- **Replicated:** nothing was fitted to the result, and the model falls
  inside its acceptance range.
- **Fitted:** the result set parameters, so matching it checks only
  consistency.
- **Not replicated:** the result was held out, and the model misses it.

| Study | Result | Status |
| --- | --- | --- |
| Diederich & Colonius (2007) | saccades to lights at 50° and 70° are 39 and 58 ms slower than at 10° | replicated (within 1 ms) |
| Diederich & Colonius (2004) | a tone is answered faster than a light, a light faster than a vibration | replicated (order) |
| Petermeijer et al. (2017) | sound and vibration take-over requests are answered faster than light | replicated (order) |
| Gold et al. (2013) | first saccade away from SuRT and first road fixation after a take-over request | replicated |
| Gold et al. (2013) | hands on the wheel | fitted |
| Gold et al. (2013) | first braking or steering input | **not replicated**: 1.81 s against 2.10–2.89 s |
| Zeeb et al. (2015) | first gaze at the road after a take-over request | replicated |
| Zeeb et al. (2015) | hands on the wheel with a knob task; monitoring during automation | **not replicated** |
| Ko et al. (2022) | hands on the wheel after an auditory warning | replicated (1.52 s against 1.87 ± 0.51 s) |
| Zhang et al. (2019) | take-over time across 520 conditions | replicated (2.03 s against 2.72 ± 1.45 s) |
| Kujala & Saariluoma (2011) | mean and within-person SD of in-vehicle glances | fitted |
| Kujala & Saariluoma (2011) | share of glances over 2 s | replicated |
| Kujala & Saariluoma (2011) | longest glances | **not replicated**: 4.2 s against 2.4–3.7 s |
| NHTSA (2013); in-vehicle glance studies | SuRT glances on the full body | partly: criteria and mean met; share over 2 s just below |
| Senders et al. (1967); Jokinen et al. (2021) | how glance duration changes with speed | **not replicated**: lies between the two results |
| Conti et al. (2014) | DRT costs of n-back and SuRT | fitted |
| Conti et al. (2014) | DRT time and its cost of driving | **not replicated**: 60 ms slow; cost +28 ms against +31 to +70 ms |
| Cooper et al. (2020) | DRT cost of a 2-back on the road; slowing of older drivers | replicated |
| Cooper et al. (2020) | on-road DRT time; DRT cost of SuRT | **not replicated** |
| O'Hanlon & McCauley (1974) | motion sickness after 2 h of vertical oscillation | fitted (one parameter) |
| McCauley et al. (1976) | motion sickness over time at nine other conditions | **not replicated**: error 14.8 against ≤ 10 |
| Frey-Law & Avin (2010) | endurance times per joint | fitted (by Frey-Law et al., 2012) |
| Liu et al. (2016) | preferred speed by lane width | fitted on two widths; replicated on three held out |
| Kraus et al. (2020) | trust over a drive | fitted on Study 1; replicated on Study 2 |
| Kraus et al. (2020) | transient drop in trust after a malfunction | **not replicated** |
| Zhang et al. (2024) | emotion ratings in three tasks | replicated (RMSE 0.098; the original model 0.13) |
| Kim et al. (2014) | key size beyond which larger touch keys stop helping | replicated (12.5 mm, at the edge of 12.5–22.5 mm) |

The [benchmark table](#benchmark) gives every number and acceptance range,
and [VALIDATION.md](VALIDATION.md) records each run.

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
implementation. The tutorial notebook
[examples/tutorial_modules.ipynb](examples/tutorial_modules.ipynb) walks
through changing, swapping, writing, testing and benchmarking modules:

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
| `touch` | `vibrotactile` | Vibration thresholds by frequency for the hand on a handle, the seat and the foot in three axes, for the fingertip, palm and grip, and for a steering wheel; felt only through contact | Morioka & Griffin (2005, 2007, 2008) |
| `vestibular` | `subjective_vertical` | 6-DOF subjective vertical conflict model: otoliths, canals and their internal model; conflict to motion sickness incidence. Alternative: `iso_2631` (Wf-weighted motion sickness dose) | Bos & Bles (1998); Kamiji et al. (2007); Wada, Fujisawa & Doi (2018); ISO 2631-1 |
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
    the driver is looking; the response module combines them, once the
    task frees the shared central stage. The driver then drops the task,
    returns gaze and hand, and assesses the scene; the request also lowers
    trust. The take-over time emerges from these steps.
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
  splits take-overs into response, gaze, hand, eyes and situation
  assessment, and exports its full trace.
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
python -m embodied_human.bench --import my_modules --replace attention=my_sampler
```

Results of the baseline (seed 0; [report](evidence/benchmark-baseline.json)).
Sixteen of twenty-five benchmarks pass:

| Benchmark | Measured | Human reference (acceptance range) | Kind of check |
| --- | --- | --- | --- |
| `eyes.main_sequence` | 2.2 ms/deg, 21 ms | 2.2–2.7 ms/deg (2.0–3.7); 21 ms (13–37) | consistency |
| `eyes.latency` | median 201 ms | 150–250 ms (Saslow, 1967) | consistency |
| `objects.peripheral_acuity` | E2 = 1.98° | 1.0–2.6° (letters, Landolt C) | consistency |
| `vision.foveation` | E2 = 2.3° | 1.0–3.0° | consistency |
| `memory.span` | 4 items | 3–5 chunks (Cowan, 2001) | consistency |
| `motor.touch_tapping` | 6.28 bits/s; 104 ms/bit | 6.95 bits/s (5.5–8.5; MacKenzie, 2015); 105 ms/bit (75–135; Bi et al., 2013) | **through physical contact** |
| `driving.surt_glances` | Full body at 72 km/h: mean off-road glance 0.93 s; 2.2% over 2 s | NHTSA (2013) criteria: pass. Human in-vehicle glances, mean 0.8–1.2 s: pass; 3–13% over 2 s: just below (score 0.98) | **held out** (fitted on a straight road at 50 km/h) |
| `driving.takeover` | chime and icon: first input after 2.03 s, SD 0.69 s (8 runs); median response 0.51 s, gaze leaves SuRT 0.61 s, eyes on the road 0.75 s, hand on the wheel 1.58 s, driving 1.71 s | 2.72 ± 1.45 s across 520 conditions (Zhang et al., 2019): pass | **held out** |
| `driving.takeover_components` | at 120 km/h, means of 16 runs: gaze leaves SuRT 0.48 s, first road fixation 0.73 s, hands on 1.63 s, first input 1.81 s | Gold et al. (2013), between their 5 and 7 s budgets: 0.41–0.54 s and 0.69–0.94 s: pass; hands 1.45–1.79 s: fitted to the midpoint; first input 2.10–2.89 s: **fail** | **held out**; hands fitted |
| `driving.takeover_motor_readiness` | task worked through a knob, icon in its display: first road fixation 0.74 s, hands on 0.97 s | Zeeb et al. (2015): 0.69 s (0.64–0.74): pass; 1.14 s (1.05–1.23): **fail** | **held out** |
| `driving.automation_monitoring` | a visual task during automation at 120 km/h: longest look away from the road 1.8 s per minute; eyes on the road 36% of the time | Zeeb et al. (2015), between their low- and high-risk groups: 8.6–44.4 s and 2.2–12.5%: **fail** | **held out** |
| `driving.takeover_modality` | median response: sound 0.35 s, vibration 0.50 s, light 0.52 s; hand on the wheel after chime and icon 1.52 s | sound and touch faster than light (Petermeijer et al., 2017): pass. Grabbing the wheel 1.87 s (1.36–2.38; Ko et al., 2022): pass | **held out** |
| `response.drt_load` | DRT cost of driving +28 ms; of an auditory 2-back +284 ms; of SuRT +382 ms | +51 ms (31–70; Conti et al., 2014), predicted from threaded cognition: just below. Cooper et al. (2020) on the road: 2-back +234 ms (152–316): pass; SuRT +177 ms (104–250): **fail** | driving and Cooper held out; chunks fitted to Conti |
| `response.modality_rt` | tone 252 ms < light 319 ms < vibration 335 ms | same order (Diederich & Colonius, 2004) | order only |
| `response.eccentricity` | a light at 50° and 70° answered 40 and 59 ms later than at 10° | saccades 39 and 58 ms later (Diederich & Colonius, 2007), RMS error ≤ 10 ms: pass | **held out**; 70° extrapolated |
| `response.drt` | while driving: head-mounted light 385 ms (353 ms without driving); vibration and windshield light pooled 368 ms; older drivers 1.32× slower | head-mounted light 326–362 ms (Conti et al., 2014; 293 ms without driving): **fail**; pooled on the road 526 ms (482–570; Cooper et al., 2020): **fail**; ageing 1.07–1.36×: pass | **emergent** |
| `vestibular.vertical_sickness` | RMS error 7.5 % points, r = 0.93 (19.7 with the published conflict scale) | 2 h vertical oscillation (O'Hanlon & McCauley, 1974): ≤ 10 | fitted (one parameter) |
| `vestibular.sickness_time_course` | RMS error 14.8 % points, r = 0.76 over 43 points | sickness over time at nine other conditions (McCauley et al., 1976), ≤ 10: **fail**; their descriptive model: 6.3. `--replace vestibular=iso_2631`: 12.4 | **held out** |
| `fatigue.endurance` | median error 17% | endurance times per joint (Frey-Law & Avin, 2010) | consistency |
| `risk.lane_width` | 56.2, 68.7 and 85.1 km/h on 2.85, 3.25 and 3.75 m lanes; mean error 2.1 km/h | 56.2, 65.5 and 81.9 km/h (Liu et al., 2016), error ≤ 9.4: pass | threshold and width growth fitted on 3.0 and 3.5 m; three held out |
| `trust.kraus_dynamics` | trust over a drive: RMS error 0.29 points; malfunction dip +0.20 | Kraus et al. (2020), Study 2 low transparency: error ≤ 0.3: pass; dip −0.47 (−0.94 to −0.10): **fail** | fitted on Study 1, Study 2 held out |
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
- **Take-over.** The take-over follows Zeeb et al.'s (2015) model. After
  noticing the request, the driver drops the task: the eyes go to the road
  and, in parallel, the hand returns to the wheel. It acts only after the
  hand is back and the road scene has been assessed.
  - **Reaction.** The request finds the central stage busy with SuRT,
    whose chunk was fitted to Conti et al.'s DRT data, so the reaction
    waits 0.36 s on average. Without being fitted to Gold et al. (2013),
    this puts the first saccade away from SuRT and the first road fixation
    between their means for 5 and 7 s budgets. Without the wait, both came
    too early (0.39 and 0.43 s).
  - **Hands.** The time to let go of the task grows with its manual load.
    It is fitted to Gold et al.'s touchscreen SuRT. That predicts Ko et
    al.'s laptop game (1.52 s against 1.87 ± 0.51 s). Assuming no release
    time for Zeeb et al.'s knob, the first road fixation matches theirs,
    but the hands arrive 0.17 s early.
  - **Acting.** The first input comes 1.08 s after the eyes reach the road,
    against 1.4–1.95 s in Gold et al.: three scene fixations are too few.
    Zeeb et al. found that drivers who had watched the road less before the
    request braked later. Each second of their longest look away added
    about 19 ms (derived from the three group means). The model does not
    link assessment to monitoring yet.
  - **Full body:** the first input comes after 2.03 s, against the
    meta-analytic 2.72 s.
- **Monitoring during automation.** Zeeb et al.'s drivers looked away from
  the road for 8.6–44 s at a time and watched it 2–12% of the time. The
  model glances as if it were steering: it looks away for at most 1.8 s
  and watches the road 36% of the time. Its attention returns to the road
  when its belief about the lane grows uncertain, and nothing tells it
  that automation is driving. Trust (Gao & Lee's decision field) would be
  the natural input.
- **Senses and responses.** Older drivers slow down as much as in Cooper
  et al. (2020), but absolute DRT times miss in both directions. The
  head-mounted light is answered about 60 ms later than in Conti et al.,
  with or without driving: TWIN's parameters come from another lab's
  lights and task. Cooper et al.'s on-road DRT is 158 ms slower than the
  model: their light changed colour as a reflection in the windshield, and
  their drivers were in traffic. Latency
  grows with eccentricity as in Osaka's (1976) manual reactions, and
  Diederich and Colonius's (2007) saccades grow by the same amounts at 50°
  and 70° (within 1 ms), though 70° lies beyond Osaka's data. Their palm
  vibrations are specified in volts, so they cannot calibrate tactile
  latency: their 50 Hz vibration matches the model about 18 dB above
  threshold, but no felt vibration in the model is as slow as their 5 Hz
  one (95 ms behind the light, against at most 66 ms).
- **Load.** Concurrent tasks share one central stage (threaded cognition).
  Driving's own cost, taken from threaded cognition's driver without
  fitting, comes out at 28 ms, just below Conti et al.'s 51 ms. With the
  n-back and SuRT chunks fitted to Conti et al.'s visual DRT, the model
  predicts Cooper et al.'s 2-back cost on the road, but doubles their SuRT
  cost: their touch-based SuRT paused at intersections, and their reaction
  times pooled two stimulus types and counted only detected stimuli.
- **Motion sickness.** One parameter (the conflict scale) refitted to the
  2-hour data cuts the error from 19.7 to 7.5. On the held-out time courses
  the error halves (28.6 to 14.8) but stays above the descriptive model of
  McCauley et al. (6.3): the conflict model's onset is too slow. The ISO
  dose was partly derived from these data.
- **Perceived risk.** With its width growth refitted (0.005 instead of
  0.001 per metre, from one driver in the original), the risk field
  predicts the three held-out lane widths within 2.1 km/h.
- **Trust.** Gao and Lee's model follows the slow growth of trust over a
  drive but cannot produce the transient dip after a malfunction, which
  people recover from by the next measurement: its belief and trust
  updates both have inertia. A fast situational component on top of
  learned trust is the missing piece.
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
- **Senses:** touch has thresholds for vibration only, at twelve sites; full
  skin with receptor types (TouchSim, once licensed) or sensor density per
  body part (as in MIMo) would replace it. Hearing handles tones in
  broadband noise, not speech or spatial sound.
- **Load:** chunks of concurrent tasks are fitted per task; deriving them
  from the task models (rules, retrievals, fixations) would make them
  predictions too. Hit rates under load are not modelled.
- **Take-over:** the model has no time budget or hazard to act on, so it
  cannot show that drivers react sooner with less time (Gold et al., 2013).
  Its assessment of the scene does not depend on how well it watched the
  road before the request (Zeeb et al., 2015).
- **Attention under automation:** monitoring ignores automation and trust
  (see above).
- **Next benchmarks:**
  - Occlusion distance on highways: mean 15.5 m (Kujala et al., 2016).

## Development

```bash
python -m pytest tests -q
python -m ruff check src tests examples
python -m mypy src/embodied_human --ignore-missing-imports
python examples/calibrate_glances.py --out evidence/glance-calibration.json
python examples/fit_motion_sickness.py --out evidence/motion-sickness-fit.json
python examples/fit_drt_load.py --out evidence/drt-load-fit.json
python examples/fit_takeover.py --out evidence/takeover-fit.json
python examples/parameter_table.py                      # regenerates docs/parameters.md
```

`pip install -e ".[test]"` also installs what the tests need to execute
the tutorial notebook. Continuous integration runs the tests, including
the notebook but without MyoSuite, then ruff, mypy and the build on
Python 3.10, 3.12 and 3.14.

Apache-2.0.
