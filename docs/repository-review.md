# Repository review, 26 October 2026

The review covered the source modules, module registry, configuration,
population and inference utilities, tasks, agents, benchmarks, examples,
packaging, CI and existing tests. It combines code inspection, targeted
regression cases, the full test suite and offline benchmark execution.
This is a review of the implementation and its reproducibility; it does not
establish independent human validity for every model.

## Bugs corrected

| Area | Observed problem | Resulting behaviour |
| --- | --- | --- |
| Public registry | `create("memory", "act_r")` failed in a fresh interpreter until another call imported built-ins. | Direct creation loads built-ins; unknown categories have a useful error. |
| Configurations | Default module choices shared mutable parameter dictionaries; modifying one configuration could change another. | Each default choice has its own dictionary; supplied dictionaries are copied. Nested values remain ordinary Python objects. |
| Memory | Small noise, extreme trace strengths and latency scales could overflow; invalid parameters and future-value retrieval were accepted. | Stable log-space activation, logistic probability and latency, parameter validation and finite chronological encoding checks. |
| Memory provenance | The illustrative `noise=0.25` setting was called fitted. | It is assumed, with its verified 2024 source. Numeric defaults are unchanged. |
| ABC inference | Invalid scales, duplicate priors and failed/nonfinite simulations could produce misleading posteriors. | Inputs are validated before simulation; acceptance requires enough finite simulations; mapper result counts are checked. |
| Detection-response task | Skipping frames could leave a stale lamp on and lose scheduled stimuli; a first-frame response could be ignored. | Every due stimulus is processed, misses counted, the lamp expires, and responses advance the schedule before scoring. |
| Fatigue | `endurance_time(..., dt=0)` could loop forever. | Nonadvancing/nonfinite steps and invalid load/duration inputs are rejected. |
| Takeover | Undetected warnings produced an invented response after seven seconds. | An unsensed request remains unanswered; a regression test checks no reaction or engagement. |
| Affect evaluation | The emotion benchmark ignored the selected affect implementation and parameter overrides. | Registered replacements are used; the report identifies its learner and skips incompatible emotion labels. |
| Affect control | Forced actions such as `0`, `False` and `""` were ignored. | Presence in the forced-action map determines the action. |
| Tapping | Perfect identical endpoints divided by zero; malformed arrays produced misleading results. | Input validation and documented undefined effective difficulty/throughput for zero endpoint spread. |
| Reports | Unscored results passed through `all([])`; NaN/Infinity made reports invalid strict JSON; an explicit empty selection ran all benchmarks. | Unscored results do not pass, undefined values serialize as `null`, and an empty selection runs nothing. |
| Source distribution | The included tutorial test referenced a notebook omitted from the source archive. | Examples, notebooks, documentation and validation evidence are explicitly included. |
| Touch | Invalid frequencies/NaNs and large sensation levels could fail or return plausible-looking latency. | Explicit domain checks and overflow-safe latency arithmetic. |

## Added integrations and interface

[TouchSim](touchsim.md) is a lazy optional `touch=touchsim` implementation.
The exact upstream MIT revision is pinned in `requirements-touchsim.txt`;
the core package's distribution metadata has no direct Git dependency.
The adapter exercises upstream skin mechanics and SA1/RA/PC spikes with
explicit indentation and geometry in millimetres. Its trials use fixed neuron
indices and no native membrane noise, so repeatability does not rely on falsely
claiming the human seed controls upstream Numba randomness. Tests execute the
real upstream model as well as checking absent-dependency behaviour.

The existing psychophysical threshold, latency and intensity pathway remains
available. TouchSim spikes do not yet drive task perception: that requires a
calibrated decoder, appropriate contact mechanics and independent human data.
Upstream whole-hand populations, surface geodesics and stimulus generators
are useful through its native API; this adapter deliberately exposes a small
explicit population on a flat patch.

The [local no-code builder](builder.md) derives choices, defaults and sources
from the registry. It supports pointer dragging, keyboard/menu alternatives,
typed parameter editing, JSON import/export, runnable Python export and a
seeded delayed-digit-recall preview. Configuration validation constructs modules;
the preview exercises memory only. It is a local configuration editor for the
package's fixed category interfaces, not a replacement scientific model or a
general signal-wiring engine. No hosting account or browser JavaScript build
dependencies are required.

## Next improvements, in priority order

1. Add cue/context-based memory and intrusion-scoring tasks before implementing
   the 2024 Oberauer–Lin interference model. For repeated long-term learning,
   investigate trace-specific adaptive decay instead. The [memory review](memory-models.md)
   links primary sources through September 2026 and describes calibration gaps.
2. Improve the existing held-out failures before expanding the number of models:
   takeover modality effects, the first gaze away from SuRT, glance extremes,
   DRT task costs and compensated-roll sickness. Preserve fitting and held-out
   results separately. Matching a source used to set parameters is a consistency
   check, not independent validation.
3. Build a tactile contact-to-indentation model and fit a perception decoder
   before using TouchSim afferent responses to claim improved driving warnings.
   Whole-hand and texture experiments need task-specific human references.
4. Extend the builder with task-specific previews and a benchmark runner, after
   making cancellation and runtime limits explicit. A general graph editor needs
   typed signals, connection validation and execution scheduling; dragging modules
   between existing categories already uses the current executable architecture.
5. Add broader scientific-domain validation to older modules and exercise extreme
   overrides. For example, the central-stage load simulator uses a finite horizon
   of ten periods per task; saturated periodic schedules need an explicit overload
   policy. Fatigue integration also assumes steps well below its recruitment time
   constant, and ISO dose uses a whole-record FFT. These are modelling/numerical
   limits to address with separate validation, rather than silently changing the
   calibrated baseline during a repository review.

Validation commands, versions, actual outcomes and remaining execution limits
are recorded in [VALIDATION.md](../VALIDATION.md). No GPL implementation was
copied and no new fitted scientific constants were invented.
