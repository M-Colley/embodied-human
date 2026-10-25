# Requirements for computational user models, and where this package stands

Dagstuhl Seminar 26432, *Computational Theories of Interactive Behavior*
(October 2026), discussed what a shared library and benchmark of
computational user models need. This page restates those requirements in
our own words and maps each to embodied-human. It is not an official
seminar output.

## Benchmarks

A benchmark needs tasks, human data, metrics of human-likeness and a
baseline model. The seminar distinguished three kinds of task:

| Kind | Here | Status |
| --- | --- | --- |
| A battery of lab tasks covered by one model | Saccades, peripheral acuity, foveation, memory span, ISO tapping, all run with one simulated human | Done. The DRT runs but is not yet scored |
| A naturalistic task | Driving the MyoSuite full body with SuRT and take-over requests (`driving.surt_glances`, `driving.takeover`); glance studies on a straight road (`driving.glance_population`) | Partial: straight roads, myosuite-carla's vehicle surrogate offline; CARLA live through myosuite-carla |
| An application: rank design alternatives as people do | `design.touch_key_size` repeats the published analysis of Kim et al. (2014) with 30 simulated participants and predicts where larger touch keys stop helping | New; stationary use only |

## Properties of a good model and benchmark

| Requirement | What the package provides | Evidence |
| --- | --- | --- |
| **Differences between people** | `population.Population` draws simulated participants by varying parameters (`Variation`) and shifting groups (`Profile`, e.g. `OLDER_ADULTS`). Each spread is labelled measured or assumed | `driving.glance_population` reports per-participant results |
| **Variability within a person** | The return threshold of the attention module is drawn anew for every glance (`threshold_spread`) | Within-person SD of glance durations 0.58 s (95% CI 0.54–0.64) vs 0.48–0.66 s in Kujala & Saariluoma (2011) |
| **Uncertainty quantification** | Bootstrap intervals over simulated participants (`bench.metrics`); posterior intervals and posterior predictive intervals from inference | `evidence/glance-calibration.json`, benchmark details |
| **Inversion (parameter inference)** | `inference.abc_rejection`: approximate Bayesian computation with Sobol prior draws (SciPy) for any simulator | `examples/calibrate_glances.py` |
| **Predictive vs explanatory accuracy** | Fitted summaries are reported apart from held-out ones (long-glance share, longest glance, speed effect, full-body driving) | See *Fitted and held-out* below |
| **Explanation by mechanisms** | Every return of gaze to the road records its cause (`uncertainty`, `maximum_glance`, `task_done`); every take-over is split into response, hand, eyes and situation assessment | `returns_by_reason`, `takeover_components()` |
| **Generative traces, not only summaries** | `DrivingLog.trace()` and `write_csv()`: gaze, belief, belief SD and true state per step; embodied-ui's `TraceRecorder` for device channels and engine outputs | Tests |
| **Counterfactual prediction** | `driving.glance_speed` predicts how glances change with speed | Shows a conflict in the human data (below) |
| **Standard formulations and libraries of forward models** | One interface per category; implementations registered by name and swapped with `HumanConfig.replace` or `--replace`; every parameter carries its source | `docs/parameters.md` |
| **Embodiment** | MuJoCo and MyoSuite bodies; physical contact with embodied-ui devices; proprioception from muscle spindles; touch felt only through contact; muscle fatigue | `motor.touch_tapping` through contact |
| **Multisensory perception** | Hearing over cabin noise, vibrotactile touch, visual eccentricity, and the TWIN model that integrates them; embodied-ui speakers and vibrotactors deliver the signals | `response.*`, `driving.takeover_modality` |
| **Closed loop with real-time systems** | embodied-ui's lockstep bridge to Unity and CARLA | Unity 6.6 and CARLA live tests in embodied-ui |
| **Easy application** | `pip install`, a benchmark CLI, examples | README |
| **Experience and motivation** | Emotion by appraisal of learned values; perceived risk that sets speed; trust that drives reliance; fatigue; motion sickness | `affect.emotion_study`, `risk.lane_width`, `fatigue.endurance`, `vestibular.vertical_sickness`; trust has no human benchmark yet |

## Architecture

The seminar described user models in terms of perception, belief, memory,
control, action, goals, individual traits and a mental model of the world.
The modules map onto these as follows:

| Concept | Module or component |
| --- | --- |
| Perception | `eyes`, `vision`, `objects`, `hearing`, `touch`, `vestibular`, `proprioception`; `response` integrates the senses |
| Belief | `estimator` (lane Kalman filter) |
| Mental model | The estimator's process model of the vehicle |
| Memory | `memory` (a focus of four chunks over ACT-R memory) |
| Meta-control (what to attend to) | `attention` (uncertainty sampler, SEEV) |
| Control and action | `motor`, the agents in `agents`, and the MyoSuite full body |
| Goals | The tasks (`tasks`): SuRT, DRT, n-back, tapping |
| Traits and biases | `population`: variations and profiles |
| Experience | `affect` (emotion), `fatigue` (effort), `vestibular` (sickness) |
| Motivation | `risk` (perceived risk), `trust` (reliance) |

## Fitted and held-out results of the glance model

The attention module's return threshold and its glance-to-glance spread
were fitted by ABC to two summaries of Kujala and Saariluoma (2011): the
mean in-vehicle glance and the mean within-person SD. Everything else was
not fitted.

| Quantity | Model | Human | Fitted? |
| --- | --- | --- | --- |
| Mean glance (20 participants, 50 km/h) | 1.13 s (95% CI 1.08–1.19) | 1.06–1.20 s | fitted (consistency check) |
| Within-person SD | 0.58 s (0.54–0.64) | 0.48–0.66 s | fitted (consistency check) |
| Glances over 2 s | 7.9% | 5.7–10% (Kujala & Saariluoma, 2011); naturalistic median 3.4%, mean 12.8% (Tivesten & Dozza, 2014) | held out: within range |
| Longest glance | 4.2 s | 2.42–3.70 s per interface | held out: **too long**. The fitted log-normal threshold has too heavy a tail, and the 4 s safety bound truncates it |
| Speed effect (elasticity of glance duration, 40–120 km/h) | −0.45 | voluntary occlusion −0.72 to −0.53 (Senders et al., 1967); in-car search 0 (Jokinen et al., 2021) | held out: **between the two**, matching neither |
| Touch-key size where gains stop | 12.5 mm | 17.5 mm (Kim et al., 2014; range 12.5–22.5) | held out: within range |

## What the conflicts mean

**Speed.** In the model, the eyes return to the road when the belief
becomes too uncertain, and uncertainty grows faster at speed. That is why
drivers in occlusion studies look away for roughly a constant distance.
Glances to an in-car search task did not depend on speed in Jokinen et
al.'s data. A model that times glances by road uncertainty alone cannot
reproduce both. The in-car task's own structure probably sets when people
look back, for example finishing a search chunk. This is a concrete target
for an alternative attention module, such as a learned supervisor like
Jokinen et al.'s.

**Tail.** The model's longest glances are too long. A threshold
distribution with a lighter tail, or a return rule that also depends on
time away, would be the next candidates. Either can be fitted and compared
with the same tools.

## Open questions this package can help answer

- **When can a simulated user replace participants in an optimization
  loop?** `design.touch_key_size` re-runs a published study's analysis on
  simulated participants. Doing the same for more design studies would
  show when the conclusions agree.
- **Which published results would replicate?** The same mechanism, with the
  original sample size and statistics, gives a model-based prediction of
  whether an effect is detectable.
- **Individual differences.** Profiles need measured spreads. Only joint
  position sense is measured so far; the others are labelled assumptions.
