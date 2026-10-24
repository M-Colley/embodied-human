# embodied-human

A modular simulated human for interaction and driving research. It covers
eyes, vision, attention, memory, proprioception, motor control and state
estimation. Each is a replaceable module with cited parameters, and a
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
| `attention` | `uncertainty_sampler` | Look back at the road when the believed lateral SD exceeds 0.4 m; leave below 0.15 m; at most 2 s away. Alternatives: `seev` (Wickens et al., 2003) and the AttenD buffer as a metric (Kircher & Ahlström, 2009) | Senders et al. (1967); NHTSA (2013) |
| `memory` | `act_r` | Focus of 4 chunks over ACT-R base-level memory (d = 0.5, τ = 0, s = 0.25, F = 1) | Cowan (2001); Anderson et al. (2004) |
| `proprioception` | `spindle` | Ia rate 4.3 v^0.6 + 2 d + r0 (r0 = 10 imp/s); 3° joint position error; 25 ms delay | Prochazka & Gorassini (1998); Malik et al. (2016); Talis & Levik (2016); Pruszynski et al. (2009) |
| `motor` | `ffitts_min_jerk` | MT = 147.7 + 105.0 log2(A/W + 1) ms; endpoint SD² = (W/4.133)² + σa², σa = 1.5 mm; minimum-jerk path | Bi, Li & Zhai (2013); Soukoreff & MacKenzie (2004); Flash & Hogan (1985) |
| `estimator` | `lane_kalman` | Kalman filter on lateral offset and heading with known speed and yaw rate; heading process noise sets how fast uncertainty grows | Senders et al. (1967) account; parameters assumed |

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
  - After a take-over request, the driver drops the task, returns gaze and
    hand, and assesses the scene. The take-over time emerges from these
    steps.

## Benchmark

```bash
python -m embodied_human.bench --out results.json
python -m embodied_human.bench --replace attention=seev --out seev.json
```

Results of the baseline (seed 0; [report](evidence/benchmark-baseline.json)):

| Benchmark | Measured | Human reference (acceptance range) | Kind of check |
| --- | --- | --- | --- |
| `eyes.main_sequence` | 2.2 ms/deg, 21 ms | 2.2–2.7 ms/deg (2.0–3.7); 21 ms (13–37) | consistency |
| `eyes.latency` | median 201 ms | 150–250 ms (Saslow, 1967) | consistency |
| `objects.peripheral_acuity` | E2 = 1.98° | 1.0–2.6° (letters, Landolt C) | consistency |
| `vision.foveation` | E2 = 2.3° | 1.0–3.0° | consistency |
| `memory.span` | 4 items | 3–5 chunks (Cowan, 2001) | consistency |
| `motor.touch_tapping` | 6.28 bits/s; 104 ms/bit | 6.95 bits/s (5.5–8.5; MacKenzie, 2015); 105 ms/bit (75–135; Bi et al., 2013) | **through physical contact** |
| `driving.surt_glances` | mean off-road glance 1.45 s; 0% over 2 s | NHTSA (2013) criteria: mean ≤ 2 s, ≤ 15% over 2 s | **emergent** |
| `driving.takeover` | mean 1.66 s (8 runs) | 2.72 ± 1.45 s across 520 conditions (Zhang et al., 2019) | **emergent** |

**How to read these results.** A *consistency* check confirms that a module
behaves as its cited parameters say. It becomes informative when a module is
replaced. *Emergent* results come from the interplay of modules and are the
real test:

- **Glances.** The baseline's off-road glances meet the NHTSA criteria.
  Those are design criteria, not a fit to human glance distributions.
- **Take-over.** The take-over time lies within one SD of the meta-analytic
  mean, but it is shorter than that mean.

Many of the integrated agent's parameters are assumed (see docs/parameters.md)
and are the first candidates for fitting.

## Relation to other work

- **User-in-the-Box** (Ikkala et al., 2022) couples MuJoCo biomechanics,
  perception and RL for interaction tasks.
- **Computational rationality** (Oulasvirta et al.) and Jokinen, Kujala &
  Oulasvirta (2021) model in-car glances as optimal adaptation under
  uncertainty. Their RL policy is a natural replacement for the
  `uncertainty_sampler`.
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
- **Reference data:** human glance-duration distributions, DRT response
  times and take-over decompositions are the next benchmarks. Their
  references are being verified before inclusion.

## Development

```bash
python -m pytest tests -q
python -m ruff check src tests
python -m mypy src/embodied_human --ignore-missing-imports
```

Apache-2.0.
