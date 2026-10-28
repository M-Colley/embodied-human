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

**Modules.** Sixteen categories cover the senses (`eyes`, `vision`, `objects`, `hearing`, `touch`, `vestibular`, `proprioception`), cognition (`attention`, `memory`, `estimator`, `response`), action (`motor`, `fatigue`) and experience (`affect`, `risk`, `trust`). Each has an interface and a cited baseline. Alternatives include SEEV attention and the ISO motion sickness dose, native [PyACTUp memory and PyDDM choices](docs/model-integrations.md), the [TouchSim afferent pathway](docs/touchsim.md), and [contact maps on the existing MuJoCo body](docs/contact-sensing.md).

**Parameter sources.** The default human and driver have 113 parameters: 12 measured, 5 set by standards, 52 fitted, 4 derived and 40 assumed. [The generated table](docs/parameters.md) records their sources. Optional adapters expose their own provenance. Their additional task and numerical settings are labelled assumed.

**Tasks and agents.** SuRT, the detection response task, delayed digit recall and ISO tapping support a touchscreen user and an attentive driver that shares gaze with SuRT and takes over from automation. The driver runs on a fast road surrogate or the MyoSuite body through myosuite-carla. Native cue based serial recall and drift diffusion choice tasks are additional unscored model demonstrations.

**Experiments.** Simulated participants, approximate Bayesian inference, paired comparisons, parameter sweeps, SciPy bootstrap intervals and optional [SALib Morris and Sobol sensitivity analyses](docs/sensitivity.md) make assumptions inspectable. [Run bundles and replay](docs/experiments.md) retain resolved configurations, declared seeds, references, validation roles, dependency versions, source hashes and available traces. Simulation intervals and sensitivity indices are not independent human validation.

**Benchmarks.** The original 28 benchmarks compare the model with 49 published aggregate references. Each result distinguishes consistency, calibration, held out evaluation and exploratory comparisons. Unresolved independence is identified explicitly. Module choices can be swapped from the command line. Reports and fitting outputs are in [evidence/](evidence/). The [failure audit](docs/benchmark-failure-audit.md) records verified corrections and remaining model limitations. The [proprioception data guide](docs/proprioception-data.md) identifies open human datasets for a dedicated task benchmark.

**Local visual builder.** Drag models into categories, edit parameters and inspect sources, then run seeded comparisons, sweeps or sensitivity experiments in a separate worker. The interface provides cancellation, execution limits, tables, plots and downloadable run bundles, alongside JSON or Python configuration export and a seeded memory preview. Launch `python -m embodied_human.builder`; [the guide](docs/builder.md) describes the supported workflows.

**Tutorial.** [The module notebook](examples/tutorial_modules.ipynb) shows how to change parameters, swap implementations, and write, register, test and benchmark a new module.

**Contributing.** [The contributor guide](CONTRIBUTING.md) covers module interfaces, parameter sources, tests, scientific evaluation roles, optional backends and automated benchmark comparisons.

Human reference data consist of published aggregates. The package includes an attributed MIT subset of MIMo for contact sampling and force allocation; optional PyACTUp, PyDDM, TouchSim and SALib computations use their installed upstream packages. [The licensing description](#relation-to-other-work) identifies that included code and its retained notice.

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
| Six-degree-of-freedom subjective vertical conflict (Bos & Bles, 1998; Kamiji et al., 2007; Wada et al., 2018); ISO 2631-1 dose | `vestibular` | Kamiji et al.'s closed-form band-pass within 0.5%; Bos & Bles's closed-form sickness for vertical sinusoids within 1%; Wf weighting and dose within 1% |
| Three-compartment fatigue (Xia & Frey-Law, 2008; Looft et al., 2018); Consumed Endurance (Hincapié-Ramos et al., 2014) | `fatigue` | endurance times per joint, median error 16% |
| Appraisal-based emotion (Zhang et al., 2024) | `affect` | the published appraisal profiles; ratings fitted at least as well as the original (RMSE 0.098 against 0.13) |
| Driver's risk field (Kolekar et al., 2020) | `risk` | closed form of lane risk |
| Extended decision field theory of trust (Gao & Lee, 2006) | `trust` | Eqs. 6-9 and Table 1 against the paper; the closed form of their Eq. 11; inertia and hysteresis |
| Vibration thresholds and the growth of perceived intensity (Morioka & Griffin, 2005, 2008, 2009) | `touch` | tables, and vector data of figures; the orderings and the steering-wheel crossover (16 Hz stronger than 125 Hz at 1.0 m/s², weaker at 0.2 m/s²) the papers report |
| Take-over process (Zeeb et al., 2015) | driver agent | the structure: hands return in parallel with the gaze, and action waits for the scene to be assessed |
| Trust and monitoring of automation (Hergeth et al., 2016) | driver agent | checks of the road fall with trust: rate = 0.62 × (1 − trust), fitted to their two measurements |

### Published human results

- **Replicated:** nothing was fitted to the result, and the model falls
  inside its acceptance range.
- **Fitted:** the result set parameters, so matching it checks only
  consistency.
- **Not replicated:** the result was held out, and the model misses it.

| Study | Result | Status |
| --- | --- | --- |
| Diederich & Colonius (2007) | saccades to lights at 50° and 70° are 39 and 58 ms slower than at 10° | replicated (within 1 ms) |
| Diederich & Colonius (2004) | a tone is answered faster than a light, a light faster than a vibration | consistency (the tactile latency uses this study) |
| Petermeijer et al. (2017) | sound and vibration take-over requests are answered faster than light | replicated (order) |
| Petermeijer et al. (2017) | eyes on the road 0.37 and 0.50 s later after a light than after a sound or a vibration | **not replicated**: 0.07 and −0.01 s |
| Gold et al. (2013) | first road fixation after a take-over request | replicated |
| Gold et al. (2013) | first saccade away from SuRT | **not replicated**: 0.81 s against 0.41–0.54 s |
| Gold et al. (2013) | hands on the wheel | fitted |
| Gold et al. (2013) | first braking or steering input | **not replicated**: 1.97 s against 2.10–2.89 s |
| Zeeb et al. (2015) | first gaze at the road and hands on the wheel with a knob task | **not replicated**: 0.90 s against 0.64–0.74 s; 1.03 s against 1.05–1.23 s |
| Hergeth et al. (2016) | monitoring frequency at two levels of trust during automation | fitted |
| Zeeb et al. (2015) | monitoring during automation: longest look away and share of time on the road | replicated (15.9 s against 8.6–44.4 s; 11.6% against 2.2–12.5%) |
| Ko et al. (2022) | hands on the wheel after an auditory warning | replicated (1.62 s against 1.87 ± 0.51 s) |
| Zhang et al. (2019) | take-over time across 520 conditions | replicated (2.43 s against 2.72 ± 1.45 s) |
| Zhang et al. (2019) | effects of the request's modality and of a visual task on take-over time | **not replicated**: modality effects a quarter or less of the human ones; the task's effect four times larger |
| Kujala & Saariluoma (2011) | mean and within-person SD of in-vehicle glances | fitted |
| Kujala & Saariluoma (2011) | share of glances over 2 s | replicated |
| Kujala & Saariluoma (2011) | longest glances | **not replicated**: 4.2 s against 2.4–3.7 s |
| NHTSA (2013); in-vehicle glance studies | SuRT glances on the full body | partly: criteria and mean met; share over 2 s just below |
| Senders et al. (1967); Jokinen et al. (2021) | how glance duration changes with speed | **not replicated**: lies between the two results |
| Conti et al. (2014) | DRT costs of n-back and SuRT | fitted |
| Conti et al. (2014) | DRT time and its cost of driving | **not replicated**: 60 ms slow; cost +28 ms against +31 to +70 ms |
| Cooper et al. (2020), tactile responses from their raw data | DRT cost of a 2-back on the road; slowing of older drivers | replicated |
| Cooper et al. (2020), tactile responses from their raw data | on-road DRT time; DRT cost of SuRT | **not replicated** |
| O'Hanlon & McCauley (1974) | motion sickness after 2 h of vertical oscillation | fitted (one parameter) |
| McCauley et al. (1976) | motion sickness over time at nine other conditions | **not replicated**: error 14.8 against ≤ 10 |
| Donohew & Griffin (2009) | how sickness from roll-compensated lateral oscillation changes with frequency | replicated (all five significant differences) |
| Donohew & Griffin (2009) | more sickness with roll compensation than without, at 0.315 Hz | **not replicated**: the model predicts half as much |
| Frey-Law & Avin (2010) | endurance times per joint | fitted (by Frey-Law et al., 2012) |
| Liu et al. (2016) | preferred speed by lane width | fitted on two widths; replicated on three held out |
| Kraus et al. (2020) | trust over a drive | fitted on Study 1; replicated on Study 2 |
| Kraus et al. (2020) | transient drop in trust after a malfunction | **not replicated** |
| Zhang et al. (2024) | emotion ratings in three tasks | unclassified data independence (RMSE 0.098; the original model 0.13) |
| Kim et al. (2014) | key size beyond which larger touch keys stop helping | replicated (12.5 mm, at the edge of 12.5–22.5 mm) |

The [benchmark table](#benchmark) gives every number and acceptance range,
and [VALIDATION.md](VALIDATION.md) records each run.

## Install

```bash
python -m pip install -e .                         # NumPy and SciPy baseline
python -m pip install -e ".[experience]"          # emotion recognition and loudness
python -m pip install -e ".[memory,decision]"     # native PyACTUp and PyDDM
python -m pip install -e ".[analysis,contact]"    # SALib and MuJoCo contact maps
python -m pip install -e ../embodied-ui             # physical touch and button tasks
python -m pip install -e ../myosuite-carla           # driving with the MyoSuite body
python -m pip install -r requirements-touchsim.txt  # pinned MIT TouchSim revision
```

Optional libraries load only when their native computation is requested. The package remains importable without embodied-ui, myosuite-carla, MuJoCo, PyACTUp, PyDDM, SALib or TouchSim. TouchSim installation uses Git from a source checkout or unpacked source distribution because that upstream package is not published on PyPI; its pinned requirements file is kept outside the package dependency metadata. [TouchSim instructions](docs/touchsim.md) explain the units, deterministic protocol and installation requirements.

## Use

```python
import embodied_human as eh

human = eh.build_human(seed=0)                          # the cited baseline
human = eh.build_human(eh.HumanConfig().replace("attention", "seev"))
for row in human.provenance():                          # every parameter's source
    print(row["parameter"], row["value"], row["kind"], row["citation"])
```

The memory baseline implements selected ACT-R equations. The optional `memory=pyactup` backend adds native feature cues, partial matching and numeric blending through PyACTUp, while the baseline focus policy stays explicit. [Native memory and choice examples](docs/model-integrations.md) describe the interfaces and seeded task demonstrations. [Memory models and newer alternatives](docs/memory-models.md) reviews the maintained ACT-R architecture, the 2024 Oberauer and Lin interference model and adaptive forgetting work published in 2026. PyACTUp is not an implementation of those distinct newer theories, and none of these added demonstrations is claimed as a fitted human replication.

The optional `response=pyddm` backend adds an explicit choice and response time API with supplied drift and deadlines. Its inherited TWIN detection path keeps its existing parameters. Choosing it does not infer evidence from sensory signals or automatically change DRT and takeover predictions. The `touch=contact_map` backend reads body contact forces, while `touch=touchsim` accepts explicit indentation traces; a contact mechanics model and a validated perceptual decoder are still needed to connect those pathways scientifically.

Use [the visual experiment interface](docs/builder.md) or [the experiment CLI](docs/experiments.md) to run matching seeds across alternatives, compare with the baseline, and retain reports for replay. Matching a model to the reference used to set its parameters remains a consistency check. Optional sensitivity estimates depend on the declared parameter bounds and seed protocol.

The [26 October repository review](docs/repository-review.md) records bug fixes,
integration limits and priorities for further work. Benchmark JSON uses `null`
for undefined or nonfinite measurements, so strict JSON readers can load it.

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
| `touch` | `vibrotactile` | Vibration thresholds by frequency for the hand on a handle, the seat and the foot in three axes, for the fingertip, palm and grip, and for a steering wheel at two hand positions; felt only through contact; perceived intensity of steering-wheel vibration by Stevens' power law | Morioka & Griffin (2005, 2008, 2009) |
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
    only if memory retrieves it. While automation drives, the driver
    instead checks the road at a rate that falls with its trust.
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
Seventeen of twenty-eight benchmarks meet every scored acceptance criterion. The report counts consistency, calibration and held out metric evaluations separately:

| Benchmark | Measured | Human reference (acceptance range) | Kind of check |
| --- | --- | --- | --- |
| `eyes.main_sequence` | 2.2 ms/deg, 21 ms | 2.2–2.7 ms/deg (2.0–3.7); 21 ms (13–37) | consistency |
| `eyes.latency` | median 203 ms | 150–250 ms (Saslow, 1967) | consistency |
| `objects.peripheral_acuity` | E2 = 1.98° | 1.0–2.6° (letters, Landolt C) | consistency |
| `vision.foveation` | E2 = 2.3° | 1.0–3.0° | consistency |
| `memory.span` | 4 items | 3–5 chunks (Cowan, 2001) | consistency |
| `motor.touch_tapping` | 6.28 bits/s; 104 ms/bit | 6.95 bits/s (5.5–8.5; MacKenzie, 2015); 105 ms/bit (75–135; Bi et al., 2013) | **through physical contact** |
| `driving.surt_glances` | Full body at 72 km/h: mean off-road glance 0.93 s; 2.2% over 2 s | NHTSA (2013) criteria: pass. Human in-vehicle glances, mean 0.8–1.2 s: pass; 3–13% over 2 s: just below (score 0.98) | **held out** (fitted on a straight road at 50 km/h) |
| `driving.takeover` | chime and icon: first input after 2.43 s, SD 0.83 s (8 runs); median response 0.57 s, gaze leaves SuRT 0.92 s, eyes on the road 0.84 s, hand on the wheel 1.58 s, driving 2.34 s | 2.72 ± 1.45 s across 520 conditions (Zhang et al., 2019): pass | **held out** |
| `driving.takeover_components` | at 120 km/h, means of 64 runs: gaze leaves SuRT 0.81 s, first road fixation 0.90 s, hands on 1.63 s, first input 1.97 s | Gold et al. (2013), between their 5 and 7 s budgets: gaze 0.41–0.54 s: **fail**; road 0.69–0.94 s: pass; hands 1.45–1.79 s: fitted to the midpoint; first input 2.10–2.89 s: **fail** | **held out**; hands fitted |
| `driving.takeover_motor_readiness` | task worked through a knob, icon in its display: first road fixation 0.90 s, hands on 1.03 s | Zeeb et al. (2015): 0.69 s (0.64–0.74): **fail**; 1.14 s (1.05–1.23): **fail** | **held out** |
| `driving.takeover_determinants` | log ratio of first-input times: icon alone against tone and icon +0.05; vibration added −0.01; SuRT against no task +0.50 | Zhang et al. (2019), mixed-effects model of 129 studies: +0.21 (0.13–0.29), −0.18 (−0.27 to −0.09), +0.10 (0.04–0.16): **fail all** | **held out** |
| `driving.automation_monitoring` | a visual task during automation at 120 km/h, trust 0.74: longest look away from the road 15.9 s in a minute; eyes on the road 11.6% of the time; 8.4 road glances a minute | Zeeb et al. (2015), between their low- and high-risk groups: 8.6–44.4 s and 2.2–12.5%: pass | **held out** (checking rate fitted to Hergeth et al., 2016) |
| `driving.takeover_eyes_by_modality` | single-modality requests, LEDs beside the display: eyes on the road 0.07 s later after a light than a sound, 0.01 s earlier than after a vibration | Petermeijer et al. (2017): 0.37 s (0.10–0.64) and 0.50 s (0.18–0.82): **fail** | **held out** |
| `driving.takeover_modality` | median response: sound 0.36 s, vibration 0.44 s, light 0.58 s; hand on the wheel after chime and icon, mean 1.62 s | sound and touch faster than light (Petermeijer et al., 2017): pass. Grabbing the wheel 1.87 s (1.36–2.38; Ko et al., 2022): pass | **held out** |
| `response.drt_load` | DRT cost of driving +28 ms; tactile DRT cost of an auditory 2-back +297 ms, of SuRT +367 ms | +51 ms (31–70; Conti et al., 2014), predicted from threaded cognition: just below. Cooper et al. (2020) on the road, tactile responses from their raw data: 2-back +294 ms (261–328): pass; SuRT +193 ms (162–223): **fail** | driving and Cooper held out; chunks fitted to Conti |
| `response.modality_rt` | tone < light < vibration | same order (Diederich & Colonius, 2004) | consistency; tactile latency uses this study |
| `response.eccentricity` | a light at 50° and 70° answered 40 and 59 ms later than at 10° | saccades 39 and 58 ms later (Diederich & Colonius, 2007), RMS error ≤ 10 ms: pass | **held out**; 70° extrapolated |
| `response.drt` | while driving: head-mounted light 383 ms (358 ms without driving); vibration 380 ms; older drivers 1.30× slower | head-mounted light 326–362 ms (Conti et al., 2014; 293 ms without driving): **fail**; vibration on the road 523 ms (490–556; Cooper et al., 2020, from their raw data): **fail**; ageing 1.20–1.41×: pass | **emergent** |
| `vestibular.vertical_sickness` | RMS error 7.5 % points, r = 0.93 (19.7 with the published conflict scale) | 2 h vertical oscillation (O'Hanlon & McCauley, 1974): ≤ 10 | fitted (one parameter) |
| `vestibular.sickness_time_course` | RMS error 14.8 % points, r = 0.76 over 43 points | sickness over time at nine other conditions (McCauley et al., 1976), ≤ 10: **fail**; their descriptive model: 6.3. `--replace vestibular=iso_2631`: 12.4 | **held out** |
| `vestibular.roll_compensation` | 30 min of lateral oscillation with roll that cancels the lateral force at the seat: MSI rises from 0.05 to 0.2 Hz and falls from 0.315 to 0.8 Hz; at 0.315 Hz compensation halves MSI | Donohew & Griffin (2009): all five significant differences between frequencies: pass; compensation made people sicker at 0.315 Hz: **fail** | **held out** |
| `fatigue.endurance` | median error 16% | endurance times per joint (Frey-Law & Avin, 2010) | consistency |
| `risk.lane_width` | 56.2, 68.7 and 85.1 km/h on 2.85, 3.25 and 3.75 m lanes; mean error 2.1 km/h | 56.2, 65.5 and 81.9 km/h (Liu et al., 2016), error ≤ 9.4: pass | threshold and width growth fitted on 3.0 and 3.5 m; three held out |
| `trust.kraus_dynamics` | trust over a drive: RMS error 0.29 points; malfunction dip +0.20 | Kraus et al. (2020), Study 2 low transparency: error ≤ 0.3: pass; dip −0.47 (−0.94 to −0.10): **fail** | fitted on Study 1, Study 2 held out |
| `affect.emotion_study` | RMSE 0.098, R² 0.84 | normalised emotion ratings in three tasks (Zhang et al., 2024); the original model: 0.13, 0.78 | unclassified data independence |
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
  - **Reaction.** SuRT occupies the central stage when the request arrives. Its chunk duration is fitted to Conti et al.'s DRT data. The reaction waits approximately 0.36 s. First road fixation falls between Gold et al.'s means for 5 and 7 s budgets. Hand release uses their hands outcome for fitting, so these outcomes share participants with calibration. The first saccade away from SuRT occurs too late (0.81 s against 0.41–0.54 s). The combined response and saccade delays exceed the observed gaze reaction time.
  - **Hands.** The time to let go of the task grows with its manual load.
    It is fitted to Gold et al.'s touchscreen SuRT. That predicts Ko et
    al.'s laptop game (a mean of 1.62 s against 1.87 ± 0.51 s). Assuming no
    release time for Zeeb et al.'s knob, the hands arrive 0.11 s early and
    the eyes reach the road 0.22 s late.
  - **Acting.** The first input comes 1.07 s after the eyes reach the road,
    against 1.4–1.95 s in Gold et al.: three scene fixations are too few.
    Zeeb et al. found that drivers who had watched the road less before the
    request braked later. Each second of their longest look away added
    about 19 ms (derived from the three group means). The model does not
    link assessment to monitoring yet.
  - **What changes take-over time.** In Zhang et al.'s (2019)
    meta-analysis, a request without sound takes 24% longer and one with a
    vibration 16% shorter. In the model these effects are 5% and 1%,
    because a request's modality only changes its first few hundred
    milliseconds of detection; in people it seems to change how urgently
    they act. Petermeijer et al.'s (2017) drivers looked at the road 0.37
    and 0.50 s later after a light than after a sound or a vibration; the
    model's driver 0.07 s later and 0.01 s earlier. A visual task,
    conversely, costs people 10% and the model 66%. Without a task the
    model's driver already looks at the road and has nothing to let go of,
    and takes over in 1.19 s; Zhang et al.'s model puts drivers without a
    visual task at 1.95 s.
  - **Full body:** the first input comes after 2.44 s, against the
    meta-analytic 2.72 s.
- **Monitoring during automation.** While automation drives, the driver
  no longer steers from its lane belief. It checks the road at a rate of
  0.62 × (1 − trust) per second. Hergeth et al.'s (2016) two measurements
  (0.13 checks/s at 74% trust and 0.097 at 82%) fix the gain and lie on
  that line within 4%, and their first rating sets the driver's initial
  trust. Without any fitting to Zeeb et al. (2015), this predicts their
  drivers' longest look away (15.9 s against 8.6–44 s) and share of time
  on the road (11.6% against 2–12%). Before, the model glanced as if it
  were steering (1.8 s and 36%).
- **Senses and responses.** Older drivers slow down as much as in Cooper
  et al. (2020), but absolute DRT times miss in both directions. The
  head-mounted light is answered about 60 ms later than in Conti et al.,
  with or without driving: TWIN's parameters come from another lab's
  lights and task. On the road, Cooper et al.'s drivers answered a
  vibration 146 ms and a light reflected in the windshield 197 ms more
  slowly than the model. Their light changed colour rather than switching
  on, and their drivers were in traffic. Latency
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
  predicts Cooper et al.'s 2-back cost on the road for the vibration
  (+290 against +294 ms; tactile times recomputed from their raw data).
  It nearly doubles their SuRT cost (+371 against +193 ms): their
  touch-based SuRT paused at intersections and hazards.
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

The baseline senses, responses, fatigue, emotion, risk and trust modules reimplement published equations. Code released without a licence, including Zhang et al.'s emotion model and Jang et al.'s fatigue model, was not used. PyACTUp and PyDDM are optional native MIT backends, and SALib supplies optional sensitivity methods under its own MIT licence. Their task and protocol assumptions are documented separately from published human reference values.

TouchSim's verified MIT revision is supported through an [optional adapter](docs/touchsim.md), with no vendored TouchSim code. The [contact map implementation](docs/contact-sensing.md) includes an adapted NumPy subset of MIMo's geometry sampling and linear force scaling from commit `040b0ae4914cbfb26afdf830aa81775b90922f3f`. The full [MIMo MIT notice](src/embodied_human/_vendor/MIMO_LICENSE.txt) is retained in source and wheel distributions. Original package code remains under Apache 2.0, the adapted subset remains under MIT, and distribution metadata records `Apache-2.0 AND MIT`.

The complete MIMo package pins MuJoCo 3.3.0 and SciPy 1.13.1, which conflict with this package's MuJoCo extras and SciPy requirement. Its NumPy 2.0.2 pin alone does not conflict with the core requirement. The adapted contact subset avoids those pins and binds the existing body rather than replacing it. Its force conservation and coordinate checks verify the implementation; they do not establish a human skin mechanics or receptor model.

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

These comparisons provide research context. The included third party code is the attributed MIT subset of MIMo described above; the other native backends remain separate installed dependencies.

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
- **Senses:** touch has vibration thresholds at twelve sites, optional TouchSim indentation responses and configurable contact force maps on existing MuJoCo bodies. Contact sensor placement is geometric and assumed rather than an anatomical receptor distribution. Force to indentation conversion and a validated spike to perception decoder remain future work. Hearing handles tones in broadband noise, not speech or spatial sound.
- **Load:** chunks of concurrent tasks are fitted per task; deriving them
  from the task models (rules, retrievals, fixations) would make them
  predictions too. Hit rates under load are not modelled.
- **Take-over:** the model has no time budget or hazard to act on, so it
  cannot show that drivers react sooner with less time (Gold et al., 2013).
  Its assessment of the scene does not depend on how well it watched the
  road before the request (Zeeb et al., 2015), and the request's modality
  changes only detection, not urgency (Zhang et al., 2019).
- **Monitoring:** the checking rate depends on trust alone; the urgency of
  the traffic, the task's demand and fatigue during long drives are not
  modelled.
- **Vestibular:** roll that compensates lateral motion halves the model's
  sickness at 0.315 Hz, where people got sicker (Donohew & Griffin, 2009).
- **Next benchmarks:**
  - Occlusion distance on highways: mean 15.5 m (Kujala et al., 2016).

## Empirical driving data

The empirical campaign imports six driving datasets and records their measurement coverage. Participant partitions precede fitting and aggregation. Calibration uses training participants. Validation participants select the candidate. Fresh simulation seeds evaluate the selected candidate on the locked test participants. Reports retain source checksums, configurations, missing observations, and paired source bootstrap intervals.

| Dataset | Scientific role |
| --- | --- |
| [Lehtonen 2018](https://doi.org/10.6084/m9.figshare.5572636) | Fixed target gaze comparison with the actual attentive driver. The in vehicle conditions support a contextual attention fit. |
| [RightGazeUrban](https://doi.org/10.5281/zenodo.20717040) | Coded gaze allocation during test track turns. The existing SEEV model supplies contextual area probabilities. |
| [ExpertDrivingDataset](https://doi.org/10.6084/m9.figshare.29664056) | Urban telemetry and gaze references. Clock alignment, actuator calibration, and scene geometry remain required for a matched control comparison. |
| [NGSIM](https://catalog.data.gov/dataset/next-generation-simulation-ngsim-vehicle-trajectories-and-supporting-data) | IDM acceleration comparison against recorded states, followed by an independent closed loop replay through the MyoSuite body. Groups identify vehicle tracks. |
| [DriveDNA Controlled](https://huggingface.co/datasets/HenryYHW/DriveDNA-Controlled) | Participant grouped vehicle telemetry. Access requires the provider's approval and applicable research licence. |
| [MPDB](https://doi.org/10.6084/m9.figshare.22192831.v3) | Event relative simulator EMG references. Units, anatomical correspondence, and amplitude normalization require verification before muscle force comparison. |

Install the optional dependencies and enter local dataset paths in a copy of [the input registry](examples/driving_data_inputs.json). Paths are resolved relative to the registry file. The physical following evaluation requires MyoSuite from its development branch and the current source checkout of `myosuite-carla`. Install both before running the campaign. The bridge must provide `myosuite_carla.empirical_following`.

```shell
python -m pip install -e ".[empirical,driving]"
python -m pip install -e ../myosuite-dev -e ../myosuite-carla
python examples/evaluate_driving_data.py --inputs examples/driving_data_inputs.json --out artifacts/empirical-driving --fit --samples 64 --seconds 30 --max-following-groups 96 --closed-loop-trials 6
```

The command attempts every dataset and saves a separate report for each source. Missing access leaves the campaign partial and returns exit code 2. Processing failures return exit code 1. An evaluated report records computational completion. Its errors and coverage determine empirical adequacy. Candidate profiles retain their test regressions and remain separate from the package defaults.

Profiles preserve the human, attentive driver, and reflex settings together. `driving_profiles.resolve_profile(report, context=...)` restores the selected parameters for an explicitly matching task context. Its resolved object provides `build_human`, `make_driver_parameters`, and `make_reflex_config`. The profile retains its fitted provenance alongside the original baseline source labels. [VALIDATION.md](VALIDATION.md) records the measured campaign outcomes.

### Physical closed loop calibration

The closed loop calibration scores trajectories generated by the attentive driver, MyoSuite muscles, and vehicle surrogate for every candidate. Recorded follower speed and gap initialize each episode. The model generates the subsequent follower motion. Recorded leader trajectories supply the traffic input.

```shell
python examples/calibrate_closed_loop_following.py --source ../myosuite-carla/artifacts/ngsim/ngsim-i80-cars.csv --exclude-report ../myosuite-carla/evidence/empirical-driving-2026-10-28/ngsim.json --out artifacts/closed-loop-calibration
```

The default pilot uses 16 Sobol candidates, six training tracks, three validation tracks, and six test tracks. Earlier inspected source tracks are excluded before selection. The command writes the fixed design before simulation and records objective calls as they finish. Each candidate uses complete 20 second episodes. The normalized objective includes speed, bumper gap, and acceleration errors with fixed scales of 1 m/s, 5 m, and 1 m/s². Collisions, road departures, incomplete windows, and numerical failures make the candidate infeasible. A source paired bootstrap evaluates the selected candidate against the baseline. Contextual acceptance requires safe and complete test pairs with a 95% upper error change below zero. Global defaults retain their original values.

The fitted parameters describe the complete model and the Nissan Patrol surrogate under these conditions. Vehicle differences in NGSIM can affect those estimates. Sixteen draws and one evaluation seed provide a pilot. Larger source samples and repeated simulation seeds are needed for population parameter inference.

## Development

```bash
python -m pytest tests -q
python -m ruff check src tests examples
python -m mypy src/embodied_human --ignore-missing-imports
python examples/calibrate_glances.py --out evidence/glance-calibration.json
python examples/fit_motion_sickness.py --out evidence/motion-sickness-fit.json
python examples/fit_drt_load.py --out evidence/drt-load-fit.json
python examples/fit_takeover.py --out evidence/takeover-fit.json
python examples/cooper_drt_aggregates.py drt.csv --out evidence/cooper-drt-aggregates.json  # raw data from the authors
python examples/parameter_table.py                      # regenerates docs/parameters.md
```

`python -m pip install -e ".[test]"` also installs the dependencies needed to execute the tutorial notebook. Core continuous integration runs the offline tests and notebook, Ruff, mypy and the build on Python 3.10, 3.12 and 3.14. An additional job on Python 3.10 and 3.12 installs PyACTUp, PyDDM, SALib, TouchSim and MuJoCo and runs their native integration tests, then checks the built distributions for the retained MIMo licence and source attribution. The regular test matrix skips MyoSuite body tests when its model and assets are absent.

The [benchmark workflow](.github/workflows/benchmarks.yml) compares 21 core protocols on pull requests and main pushes. Base and proposed revisions run with identical dependencies and seeds. Existing model discrepancies remain visible. New regressions, lost coverage and incomparable reference declarations fail the comparison. Its manual `full` profile runs all 28 protocols with pinned physical dependencies and the offline vehicle surrogate. [The contributor guide](CONTRIBUTING.md#run-automated-benchmark-comparisons) describes local commands, GitHub execution and retained evidence.

Original code: [Apache 2.0](LICENSE). Adapted MIMo subset: [MIT](src/embodied_human/_vendor/MIMO_LICENSE.txt). Distribution expression: `Apache-2.0 AND MIT`.
