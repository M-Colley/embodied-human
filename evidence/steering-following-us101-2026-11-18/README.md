# Steering diagnostic on an inspected US 101 source

The baseline road departure reproduces exactly for source `ngsim-us101:1761:1`, base seed 702 and simulation seed 760246850. The reference matches every archived prediction and trajectory metric. Departure occurs after 974 steps (19.48 s). This source was inspected during the completed US 101 comparison. The present experiment provides a development diagnostic.

The intervention bypasses the muscle steering mechanism. The vehicle receives the reflex controller's requested normalized steer. Its fitted steering rate limit and dynamics remain active. Both interventions replay every applied longitudinal command from the reference. They share the complete initial MuJoCo integration state, human random states, vehicle state, source trajectory and seed. Command values and application times match exactly. The muscle replay matches every reference physical state and telemetry row. The comparison uses the reference's 19.48 s horizon.

| Outcome | Muscle replay | Ideal steering command |
| --- | ---: | ---: |
| Maximum absolute lane offset (m) | 3.54164 | 0.07531 |
| Lane belief error RMS (m) | 0.02614 | 0.02590 |
| Road departure | Yes | No |

The muscle steering error grows late in the trial. Its normalized wheel tracking RMSE is 0.30989 after 15 s. At departure, requested steer is +0.42942 and measured wheel position is -0.06677. Lane belief remains close to the true lane state. These observations support further work on the steering actuator and control loop. Controller parameters remain unchanged.

The signed error analysis covers the 23 source tracks that completed both seeds in both archived arms. Mean predicted minus observed gap is -0.38335 m for baseline and +1.43982 m for inverse actuation. End gap bias is -1.10959 m and +2.46450 m respectively. Mean speed bias is +0.05380 m/s and -0.12416 m/s. Sources have equal weight after averaging their seeds. These summaries retain the original source identities.

The native regression tests passed with MyoSuite 3.1.0 and development revision `c93329ec64bc8b544d0b3427bba6766377c3528e`. Both environments passed three tests. The tests cover command isolation, complete physical resets, exact muscle replay and equal source weighting. Three existing MuJoCo prefix warnings occurred in each run. Ruff and mypy passed. The diagnostic used bridge PR5 source `9bd4d73c1fbfed3bd3ba54900965a9b017a1c653`. No CARLA server was used.

The source fixture, full intervention traces, signed errors and derived metrics are retained beside this file. `summary.json` records runtime identity and reproduction checks. `diagnostic-source.zip` preserves the exact diagnostic source used for the run. Subsequent code changes add type annotations, preserve typed access to estimator state and require every parity check for a successful command line exit. The remaining human source is based on PR3 revision `43e74929a9cec2b4b380f05763c7f2a7e8dcfe12`.

The source trajectory derives from the U.S. Department of Transportation Federal Highway Administration NGSIM US 101 records (https://doi.org/10.21949/1504477, CC BY-SA 3.0). The original 24 track fixture and paired episode reports were published in the US 101 comparison evidence. The new source fixture and summaries retain this attribution and licence.

```shell
python examples/diagnose_following_steering.py --study path/to/us101-study --out artifacts/steering-diagnostic
python -m pytest tests/test_steering_diagnostics.py -q
```
