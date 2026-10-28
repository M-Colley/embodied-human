# Validation

## Actuator correction and independent evaluation on 2026-10-28

The corrected actuator interface reduced mean test normalized trajectory MSE from 0.7686406859 to 0.6987135337 (9.10%). The paired change was −0.0699271523 (95% source bootstrap interval [−0.1654942475, 0.02120664355]). The interval includes zero. The result is `inconclusive_contextual_candidate`. Global defaults remain unchanged.

| Mean source test metric | Legacy actuator interface | Corrected actuator interface |
| --- | --- | --- |
| Speed RMSE | 0.88229290 m/s | 0.83071484 m/s |
| Bumper gap RMSE | 3.41394387 m | 3.59434404 m |
| Acceleration RMSE | 0.85608742 m/s² | 0.75578731 m/s² |

The experiment compared three fixed configurations on 32 development tracks sampled from the 112 previously inspected tracks. Arm A used legacy actuation and the baseline IDM. Arm B used the vehicle inverse and measured pedal feedback with the same IDM parameters. Arm C combined corrected actuation with the earlier component IDM fit. Development objective values were 1.03503493, 1.00556169, and 1.10160000, respectively. Development selection therefore chose B. The selected configuration was stored before test jobs began.

The independent test used 24 fresh tracks outside all 112 inspected identities. Each track had one complete 20 second window and two common simulation seeds (117 and 118). The test contained 48 matched seed pairs and 96 physical episodes. Development contributed 96 episodes. All 192 episodes completed without collision, road departure, numerical failure, or incomplete coverage. Each worker reused one 114 muscle MyoSuite body. All worker protocols and surrogate vehicles were closed after execution.

The primary score first averages metric RMSE across seeds within each source. It then squares normalized errors across speed, bumper gap, and acceleration. The fixed normalization scales remain 1 m/s, 5 m, and 1 m/s². Sources receive equal weights. The paired source bootstrap uses 5,000 resamples. Complete safe pairs and an upper 95% error change below zero are required for contextual acceptance. An independent audit reproduced all development distances, the selected arm, every episode RMSE, and the test interval.

The engineering reference grid contained nine conditions per interface (18 trials). Initial speeds were 5, 10, and 15 m/s. Acceleration demands were −1, 0, and +1 m/s². Both interfaces were scored at common times after two seconds. Feasible reference acceleration becomes zero after the reference vehicle stops. Mean reference grid tracking RMSE decreased from 0.54323634 to 0.08542668 m/s². This metric was computed from the 18 saved traces after physical execution. The traces remained unchanged. This consistency check uses the same fitted vehicle model for inverse control and simulation.

The comparison design was fixed before its simulation calls. Development selection was written at 16:17:36 UTC. Test workers started at approximately 16:17:37 UTC. Human source revision `e938e90b7d38638fe2c588e5e7f7a6621e41eec2` and bridge source revision `d545099dd1766c0f044cab74810620ff39e0381c` were committed at 16:17:46 UTC. The first completed test result was written at 16:17:47 UTC. The source commits occurred during the transition to testing. Runtime model source hashes remained unchanged throughout execution.

The [fixed design](evidence/actuation-improvement-2026-10-28/design.json) and [study evidence archive](evidence/actuation-improvement-2026-10-28/study-evidence.zip) retain source assignments, configurations, all episodes, selection records, commands, and provenance. The local montage is `../myosuite-carla/artifacts/actuation-improvement-videos/held-out-trials.mp4`. Its two illustrative sources, `ngsim:1136:2` and `ngsim:1193:2`, were chosen by ascending test source identifier. These videos show verified offline replays.

The completed full suites passed 513 bridge tests with four live tests skipped and 561 human tests with 24 optional tests skipped. Ten targeted tracking and video checks also passed. Ruff and mypy passed for both packages. Formatting checks passed for the changed Python files. Wheel and source builds passed. The environment used Python 3.12.13, MuJoCo 3.17.0, NumPy 2.5.4, and SciPy 1.17.1. MyoSuite used development revision `37c8b863fe448f8fff2b29eb62a53a269b79d16f`.

The longitudinal controller receives exact vehicle telemetry in this experiment. The human estimator covers lateral state. Recorded vehicles differ from the fitted Nissan Patrol surrogate. Perception error, vehicle heterogeneity, and broader driving conditions require further evaluation. Live CARLA validation of this correction remains pending.

The configured reflex delay was 0.02 s at a 0.02 s control timestep. The history stores `max(1, round(perception_delay / dt))` observations and appends the current observation before reading its oldest entry. This setting adds zero history lag. The preserved timing is relevant when interpreting human response times.

Hosted CI passed for the final evidence commits. [Bridge CI at `fedb997`](https://github.com/M-Colley/myosuite-carla/actions/runs/40729055945) passed its offline and integration jobs. [Human tests at `db886b3`](https://github.com/M-Colley/embodied-human/actions/runs/40729056025) and the [matched core benchmarks](https://github.com/M-Colley/embodied-human/actions/runs/40729056192) passed. The bridge live job was skipped.

Commands ran from the standalone `myosuite-carla` repository:

```powershell
$env:PYTHONPATH = (Resolve-Path '../embodied-human/src').Path + [IO.Path]::PathSeparator + (Resolve-Path '../myosuite-dev-review').Path
.venv/Scripts/python.exe examples/check_acceleration_tracking.py --out artifacts/actuation-improvement-2026-10-28/acceleration-tracking.json
.venv/Scripts/python.exe ../embodied-human/examples/evaluate_actuation_improvement.py --source artifacts/ngsim/ngsim-i80-cars.csv --component-report evidence/empirical-driving-2026-10-28/ngsim.json --development-report evidence/empirical-driving-2026-10-28/ngsim.json --development-report artifacts/closed-loop-calibration-2026-10-28/run/fullreport.json --development-report artifacts/closed-loop-benchmark.json --out artifacts/actuation-improvement-2026-10-28/run --development-sources 32 --test-sources 24 --seed 20261029 --development-seed 17 --test-seeds 117 118 --seconds 20 --desired-speed 16 --bootstrap 5000 --workers 3
.venv/Scripts/python.exe examples/record_empirical_trials.py --source artifacts/ngsim/ngsim-i80-cars.csv --report artifacts/actuation-improvement-2026-10-28/run/fullreport.json --out artifacts/actuation-improvement-videos --groups ngsim:1136:2 ngsim:1193:2 --base-seed 117
.venv/Scripts/python.exe -m pytest tests -q
.venv/Scripts/python.exe -m pytest ../embodied-human/tests -q
```


## Physical closed loop calibration on 2026-10-28

The physical calibration completed 141 full episodes without a collision, road departure, numerical failure, or incomplete window. Validation selected candidate 3. Its validation objective decreased from 1.23025 to 1.12046. On the six fresh test tracks, normalized trajectory MSE increased from 1.03338 to 1.16906 (13.13%). The paired change was +0.13568 (95% bootstrap interval [−0.02073, 0.30352]). The interval includes zero. The predeclared acceptance rule rejected the candidate. Model defaults remain unchanged.

| Test metric | Baseline | Fitted candidate |
| --- | --- | --- |
| Mean source speed RMSE | 0.98997 m/s | 1.03742 m/s |
| Mean source bumper gap RMSE | 4.44644 m | 4.95188 m |
| Mean source acceleration RMSE | 0.99246 m/s² | 1.00335 m/s² |

The [summary](evidence/closed-loop-calibration-2026-10-28/summary.json) retains the measured outcome and file hashes. The [report archive](evidence/closed-loop-calibration-2026-10-28/reports.zip) contains all 141 episodes, candidate draws, objective calls, validation scores, source exclusions, exact code, and provenance. The [comparison figure](evidence/closed-loop-calibration-2026-10-28/closed-loop-comparison.png) shows all six test pairs. The design and source code were [published](https://github.com/M-Colley/embodied-human/commit/cbb05bf934005e25dff01af9023bcfa5b9446cb2) before test evaluation. The [bridge source](https://github.com/M-Colley/myosuite-carla/commit/c09b2cf0882e27cb3e9141cff33afb1655b32213) was also published before that stage. Runtime source hashes remained identical throughout execution. An independent calculation reproduced all training and validation distances, the selected candidate, and the paired test interval.

The calibration objective now runs the complete attentive driver, MyoSuite body, and fitted vehicle surrogate for every parameter candidate. Recorded leader motion drives the scene. Recorded follower speed and gap initialize the episode. Subsequent follower motion comes from the model. The physical protocol reuses one body and caches exact parameter, source, trial, and simulation seed combinations.

The experiment excludes all 96 tracks inspected in the previous empirical campaign and one additional track used for a runtime measurement. A seeded selection supplies 15 fresh tracks. Six train the candidate, three select it, and six evaluate it. These assignments, priors, normalization constants, and acceptance criteria are written before physical execution. Each source contributes one complete 20 second window. The fit seed is 17. The evaluation seed is 117. Source specific seeds derive from the source identity and trial index.

Sixteen Sobol draws explore four IDM parameters. The assumed search ranges are 0.6 to 2.6 seconds for time headway, 0.5 to 4 m for minimum gap, 0.5 to 2 m/s² for maximum acceleration, and 0.8 to 3 m/s² for comfortable deceleration. Desired speed remains 16 m/s. The baseline competes with the four accepted draws on validation data. This finite search supplies a contextual calibration pilot. Accepted draw intervals describe the search. Human parameter uncertainty remains unevaluated.

The primary objective averages squared normalized RMSE across speed, bumper gap, and acceleration. The fixed scales are 1 m/s, 5 m, and 1 m/s². These scales are assumed normalization constants. Trial RMSE is averaged within each source before normalization and squaring. Sources receive equal weights. A collision, road departure, numerical failure, or incomplete episode makes a candidate infeasible. Contextual acceptance requires complete safe test pairs and a paired source bootstrap upper 95% error change below zero. The bootstrap uses 2,000 resamples. All component errors remain visible. Global defaults retain their original values and citations.

The physical experiment uses the fitted CARLA 0.10 Nissan Patrol surrogate. Recorded NGSIM vehicles have heterogeneous dynamics. Fitted controller values can absorb vehicle and actuator mismatch. The experiment uses one training seed and one distinct evaluation seed. Additional independent tracks and repeated simulation seeds are required to assess generalization. Live CARLA validation remains a separate requirement.

The complete human suite passed 541 tests and skipped 24 optional tests in 87.13 seconds. The complete bridge suite passed 454 tests and skipped four live tests in 118.06 seconds. Ruff and mypy passed for both packages. Mypy checked 67 human source files and 24 bridge source files. Focused tests cover source exclusions, seed and cache identities, infeasible episodes, matched scoring, fixed partitions, single source acceptance guards, and preservation of existing evidence.

Wheel and source builds passed for both packages. Hosted human tests and the matched core benchmark comparison passed for source revision `cbb05bf`. Hosted bridge integration and all Linux and Windows jobs passed for source revision `c09b2cf`. The opt in live job remained skipped. The source distributions retain the new examples and evidence. NGSIM attribution and its recorded CC BY-SA 3.0 licence are retained in the report archive and summary.

Commands ran from `myosuite-carla`:

```powershell
$env:PYTHONPATH = (Resolve-Path '../embodied-human/src').Path + [IO.Path]::PathSeparator + (Resolve-Path '../myosuite-dev-review').Path
.venv/Scripts/python.exe ../embodied-human/examples/calibrate_closed_loop_following.py --source artifacts/ngsim/ngsim-i80-cars.csv --out artifacts/closed-loop-calibration-2026-10-28/run --exclude-report evidence/empirical-driving-2026-10-28/ngsim.json --exclude-report artifacts/closed-loop-benchmark.json --samples 16 --train-sources 6 --validation-sources 3 --test-sources 6 --seconds 20 --trials-per-source 1 --seed 20261028 --fit-seed 17 --evaluation-seed 117 --accept 0.25 --bootstrap 2000 --desired-speed 16
.venv/Scripts/python.exe -m pytest ../embodied-human/tests -q
.venv/Scripts/python.exe -m pytest tests -q
```

## Empirical driving campaign on 2026-10-28

Five public datasets completed processing. DriveDNA Controlled remains access pending after the provider returned HTTP 403 with `GatedRepo`. The owner confirmed that access is pending. All six adapters are implemented. The campaign exits with code 2 while access is pending. No dataset processing failure remains in the retained campaign.

All three contextual fits were rejected after evaluation. Package defaults and their cited source labels remain unchanged. The rejection rule is a conservative post hoc release review. Test observations were never used to retune a candidate. A subsequent fit requires a new evaluation design because these test results have now been inspected.

| Source | Coverage | Outcome |
| --- | --- | --- |
| Lehtonen 2018 | 14 participants after the authors' exclusions; 12 eligible for the in vehicle fixation protocol; partitions 7, 2, and 3 | Test mean glance duration MAE increased from 0.498 to 0.966 seconds. All three test metrics regressed. |
| RightGazeUrban | 34 participants, 539 trials, and 1,109,163 aligned samples; partitions 20, 7, and 7 | Road allocation MAE decreased from 0.0603 to 0.0200. Rear mirror and right shoulder errors increased. The SEEV candidate was rejected. |
| ExpertDrivingDataset | 20 participants, 280 CAN trials, and 609,597 vehicle samples; eye records from 8 participants | Telemetry and gaze references were computed. Clock alignment, actuator calibration, and matched scene geometry remain required for control comparison. |
| NGSIM | 4,346 source tracks and 8,878 following windows; 96 seeded tracks selected before partitions of 58, 19, and 19 | Mean trial acceleration RMSE decreased from 2.823 to 1.348 m/s² against recorded states. Closed loop speed and gap errors increased. |
| DriveDNA Controlled | Provider approval pending | Evaluation remains pending. |
| MPDB | All 35 participants, 6,953 processed epochs, four channels, and 27,812 channel epochs | There were 10,457 detected, 649 left censored, and 16,706 right censored onsets. Anatomical channel correspondence and source units remain unverified. |

The NGSIM physical comparison ran six paired 20 second held out leader replays through `AttentiveDriver`, one cached MyoSuite body with 114 muscles, and the fitted CARLA 0.10 Nissan Patrol surrogate. All 12 episodes completed without a collision or road departure. Speed RMSE increased from 0.8803 to 1.0177 m/s. Gap RMSE increased from 3.6497 to 4.5717 m. The paired source mean increase in speed RMSE was 0.1374 m/s (95% bootstrap interval [0.0133, 0.2789]). These runs used the offline surrogate. UE5.8 live validation remains pending.

The fixation fit uses a fixed visual target task in the actual attentive driver. Target geometry is estimated from training participants. Its two degree area extent is an assumption. RightGaze fits the existing SEEV allocation component. Its coded areas retain the source ambiguity around zero event identifiers. MPDB reports event relative EMG onset with a trailing 50 ms RMS window, a baseline from −0.40 to −0.05 seconds, a median plus three scaled MAD threshold, and 50 ms sustained detection. Competing events and censoring are retained. Activation onset has no established mapping to reaction time or absolute muscle force.

Original source groups remain intact through splitting, aggregation, paired comparison, and bootstrap resampling. Seeds are averaged within trials. Trials are averaged within source groups. Sources receive equal weights. The source groups in NGSIM identify vehicle tracks. Participant identifiers are unavailable there. Reports preserve the original observations, selected configurations, assumptions, checksums, and provenance. Profiles require an explicitly matching context. The original SuRT default trace remains identical over 3,000 seeded steps.

The [evidence summary](evidence/empirical-driving-2026-10-28/summary.json) lists outcomes and retained file hashes. The [report archive](evidence/empirical-driving-2026-10-28/reports.zip) contains all six reports, the acquisition manifest, and the campaign manifest. It contains derived observations and configuration records. Bulk dataset files remain in the local artifact cache. The [paired replay figure](evidence/empirical-driving-2026-10-28/closed-loop-comparison.png) shows every physical trial. Dataset attribution and licences remain in the source reports.

The environment used Python 3.12.13, NumPy 2.5.4, SciPy 1.17.1, pandas 3.0.6, and MuJoCo 3.17.0. MyoSuite used development base `230f198458b501c423a0892b78da60419ffc49a3` with fatigue correction `37c8b863fe448f8fff2b29eb62a53a269b79d16f`. The campaign records actual import paths and source hashes separately from installed distribution metadata. This distinguishes the `PYTHONPATH` development checkout from the installed MyoSuite checkout.

The complete human suite passed 512 tests and skipped 24 optional tests in 116.49 seconds. The subsequent campaign provenance checks passed all 16 focused tests. The bridge suite passed 434 tests and skipped four live tests. Ruff, mypy, wheel builds, and source builds passed for both packages. The source distributions include the input registry, campaign examples, and retained evidence. The new task and empirical support preserve lazy optional imports.

Commands ran from `myosuite-carla` with the local human and MyoSuite development sources:

```powershell
$env:PYTHONPATH = (Resolve-Path '../embodied-human/src').Path + [IO.Path]::PathSeparator + (Resolve-Path '../myosuite-dev-review').Path
.venv/Scripts/python.exe ../embodied-human/examples/evaluate_driving_data.py --inputs artifacts/empirical-driving-2026-10-28/inputs.json --out artifacts/empirical-driving-2026-10-28/final --fit --samples 64 --seconds 30 --max-following-groups 96 --closed-loop-trials 6
.venv/Scripts/python.exe -m pytest ../embodied-human/tests -q
.venv/Scripts/python.exe -m pytest tests -q
.venv/Scripts/python.exe ../embodied-human/examples/plot_driving_comparison.py artifacts/empirical-driving-2026-10-28/release/campaign-datasets/ngsim.json --out artifacts/empirical-driving-2026-10-28/closed-loop-comparison.png
```

The release evidence adds actual runtime import provenance. It reuses the completed fixation evaluation from the fresh campaign after checking its observations, contexts, source fingerprint, numerical dependencies, partitions, settings, and sample count. It reruns every other source and all 12 physical episodes. The evidence archive retains both the fresh fixation evaluation and the exact release command. No fitting data or parameter changed between these runs.

## 2026-10-28, local rotating frame head sensing snapshot

The local driving source composes head acceleration with vehicle rotation at the head centre of mass. Six analytic tests cover centripetal acceleration, the seat offset, Coriolis acceleration, relative acceleration, a rotating head centre of mass, and head orientation. The driving dependency requires myosuite-carla 0.1.1. This source was used for the live CARLA 0.10.0 recordings documented in [myosuite-carla](https://github.com/M-Colley/myosuite-carla/blob/70457a8b83041fd1d48ba04a14a00582d5179d3f/VALIDATION.md).

The complete suite passed 452 tests and skipped 24 optional tests in 88.85 seconds ([output](evidence/rotating-head-local-2026-10-28-suite.txt), [JUnit](evidence/rotating-head-local-2026-10-28-suite.xml)). Tests loaded the source from the original local embodied-human checkout. The environment used Python 3.12.13, MuJoCo 3.17.0, NumPy 2.5.4, and MyoSuite development base `230f198458b501c423a0892b78da60419ffc49a3` with the fatigue correction in `37c8b863fe448f8fff2b29eb62a53a269b79d16f`. Ruff passed. Mypy passed all 60 source files. Wheel and source builds passed. The existing [head sensing PR](https://github.com/M-Colley/embodied-human/pull/1) contains the same model correction and its seeded benchmark comparison.

Commands ran from the embodied-human repository:

```powershell
$env:PYTHONPATH = (Resolve-Path 'src').Path + [IO.Path]::PathSeparator + (Resolve-Path '../myosuite-dev-review').Path
../myosuite-carla/.venv/Scripts/python.exe -m pytest tests -q --junitxml=../myosuite-carla/artifacts/embodied-human-live/human-final-snapshot-tests.xml
../myosuite-carla/.venv/Scripts/python.exe -m ruff check src tests
../myosuite-carla/.venv/Scripts/python.exe -m mypy src/embodied_human --ignore-missing-imports
../myosuite-carla/.venv/Scripts/python.exe -m build
```

## 2026-10-27, Linux contact geometry CI correction

The hosted test workflow for `d4eaf21` failed in all five jobs on the capsule and cylinder contact map cases. MuJoCo 3.17.0 enum equality is asymmetric with NumPy integer scalars on Linux. `scalar == enum` returns true, while `enum == scalar` returns false. Tuple membership uses the latter comparison and therefore skipped both supported geometry types. Windows bindings accept both operand orders. Geometry dispatch now converts the model type and enum constants to Python integer codes. Two regression cases reproduce the Linux equality behavior on either platform. No module defaults or human reference ranges changed.

The original failure was reproduced on Ubuntu 24.04 with Python 3.12.3, NumPy 2.5.4 and MuJoCo 3.17.0. The original primitive subset passed three cases and failed two. The corrected complete contact test file passes 20 tests and skips the optional MyoSuite body test on Linux. Windows Python 3.12.13 passes all 21 contact tests, including that body test. Ruff, mypy for all 60 source files, and wheel and source builds pass in a separate checkout containing only this correction. A complete local run passed 469 tests and correctly rejected one experiment replay after a concurrent source change in the editable `myosuite-carla` dependency. Its targeted rerun passes with a stable dependency snapshot. The Windows environment includes the vehicle package's currently declared Shapely dependency (2.2.0). Concurrent workspace edits are excluded from this commit.

The hosted benchmark workflow failed because its comparison against `9d38317` retained the seven core metric regressions and two declaration changes already documented in the failure audit. Its original [reports, summary, comparison context and dependency records](evidence/ci-audit-core/) are preserved before the hosted artifact expires. The new correction uses `d4eaf21` as the push baseline. All 41 core metric values exactly match the corrected baseline locally. The strict regression, missing coverage, skip and declaration gates remain active. A manual comparison against the older scientific model revision still reports the documented differences.

The failed hosted runs are [tests](https://github.com/M-Colley/embodied-human/actions/runs/40518585560) and [benchmarks](https://github.com/M-Colley/embodied-human/actions/runs/40518585599). The new hosted run must verify the correction across Linux Python 3.10, 3.12 and 3.14, both optional integration jobs, and the matched core benchmark job.

## 2026-10-27, benchmark failure audit

The audit corrected study specific DRT response windows, first road fixation timing, gaze reaction eligibility, planned hand completion censoring, final glance exposure accounting and the vestibular sample timestamp. McCauley datum 17 now uses the observed value 28 from the scanned table. The fast takeover simulation allows ten seconds after its default request and reports component and condition completion counts. Response inputs reject nonfinite and impossible values. A continuously occupied central task returns an infinite wait. Proprioceptive noise and delay mappings now carry assumed provenance. Their numerical values remain unchanged. The experimental parallel gaze coupling is optional, labelled assumed and disabled by default.

The [source audit](docs/benchmark-failure-audit.md) records all 11 failing protocols and the remaining scientific requirements. The full seed zero run completed 28 protocols in 178.398 seconds with zero skips. Seventeen protocols pass and 11 fail. Thirty five of 53 scored metric evaluations lie within their ranges, compared with 34 before the audit. Roles separate nine consistency checks, five calibration evaluations, 36 held out evaluations, two exploratory task mismatch comparisons and one unresolved evaluation. Their within range counts are nine, five, 20, zero and one. Gold's intervention time improves from 1.972 to 2.254 seconds and enters its range. Nine metrics move farther outside their ranges after the corrected measurements and required refits. Two declaration changes concern the unchanged glance speed values. The [comparison report](evidence/benchmark-failure-audit.md) preserves all regression flags. The [archived baseline](evidence/benchmark-before-failure-audit.json) is byte identical to the baseline at commit `9d38317ec9f0e07c5a393212ab5d8fae210f5ac8` (Git blob `f0b095dfc902a2377828675854cf1d9675e09fb7`). Numerical acceptance ranges are unchanged.

Both required calibration scripts were rerun after response and vestibular changes. DRT fitting uses Conti's 100 to 2000 millisecond hit window. Its fitted chunks are 0.757666 seconds for zero back, 1.130280 for two back and 2.484397 for SuRT. Defaults use 0.758, 1.130 and 2.484 seconds. These contrasts share the observed Conti driving baseline. Cooper's observations remain external to fitting. The hand release refit uses Gold's existing hands target and selects 0.20 seconds. Its mean is 1.620628 seconds against the 1.62 second target. The local grid accounts for discrete simulation steps. Motion sickness fitting uses only the 14 vertical endpoint conditions and selects a conflict scale of 0.196237117. The default is 0.196237. Fitted endpoint RMSE is 7.462753 percentage points. Reserved time course RMSE is 14.838772. The corrected descriptive McCauley predictions have RMSE 6.335577 over those same observations. [DRT fit](evidence/drt-load-fit.json), [hand fit](evidence/takeover-fit.json), [motion fit](evidence/motion-sickness-fit.json).

The complete Python 3.12.13 suite passed 468 tests in 100.94 seconds. The retained [JUnit report](evidence/tests-failure-audit.xml) records the result. Nine upstream MuJoCo and Jupyter warnings remain. The following provenance label edits passed 43 module, study and reporting tests. Python 3.10.20 passed 98 focused timing, censoring, response validation, source integration and comparison tests. Ruff passed for source, tests and examples. Mypy passed all 60 source files. The parameter table was regenerated. Both wheel and source distribution builds passed. The complete numerical run follows the corrected source and provenance labels. Hosted CI execution has not been verified. A comparison against the preaudit revision is expected to report the documented numerical regressions and declaration changes.

The numerical environment uses NumPy 2.5.4 and SciPy 1.18.1. Test tools are pytest 9.1.1, Ruff 0.17.2, mypy 2.4.0, build 1.6.1 and setuptools 84.2.1. The Python 3.10 checks use NumPy 2.2.6 and SciPy 1.15.3. Optional native package versions remain those recorded in the preceding native model validation. The built wheel was reinstalled with `uv pip` into the isolated core environment. It imports and executes response calculations without MuJoCo or the other optional native packages. Its corrected provenance labels also pass the smoke check. The [proprioception data guide](docs/proprioception-data.md) records primary sources, inspected files, checksum provenance, licences and protocol limitations. It adds no participant records or fitted human proprioception benchmark.

```powershell
.venv/Scripts/python.exe examples/fit_drt_load.py --out evidence/drt-load-fit.json
.venv/Scripts/python.exe examples/fit_motion_sickness.py --out evidence/motion-sickness-fit.json
.venv/Scripts/python.exe examples/fit_takeover.py --out evidence/takeover-fit.json
.venv/Scripts/python.exe examples/parameter_table.py
.venv/Scripts/python.exe -m embodied_human.bench --out evidence/benchmark-baseline.json --compare evidence/benchmark-before-failure-audit.json --summary evidence/benchmark-failure-audit.md
.venv/Scripts/python.exe -m pytest tests -q --junitxml=evidence/tests-failure-audit.xml
.venv/Scripts/python.exe -m ruff check src tests examples
.venv/Scripts/python.exe -m mypy src/embodied_human --ignore-missing-imports
.venv/Scripts/python.exe -m build
```

## 2026-10-27, contribution guide and automated benchmark comparisons

The contribution guide includes an executable module, tests, provenance requirements, calibration triggers, catalogue declarations and local or hosted benchmark commands. The new benchmark workflow runs the 21 protocol core profile on pull requests and main pushes. Its manual full profile runs all 28 protocols with pinned physical dependencies. Both revisions use one numerical environment and seed zero. Reports, summaries, dependency records and logs are retained for 14 days. The workflow uses read access to repository contents and the ordinary pull request event.

The automated full evaluation ran all 28 protocols in 185.861 s with no skips. All 53 scored metric evaluations match the committed baseline exactly. The regression gate returned exit code 0. Existing acceptance results remain 17 passing protocols and 11 failing protocols. The [retained report](evidence/benchmark-automation.json), [summary](evidence/benchmark-automation.md) and [analysis environment](evidence/benchmark-automation-environment.json) record the results. The final comparison presentation was regenerated from the retained numerical report after the output formatting changes.

```powershell
.venv/Scripts/python.exe -m embodied_human.bench --profile full --compare evidence/benchmark-baseline.json --fail-on-regression --fail-on-skip --out artifacts/automated-full/report.json --summary artifacts/automated-full/summary.md --bundle artifacts/automated-full
```

A negative control set `memory=act_r:capacity=2` and selected `memory.span`. Its span changed from four to two. The gate reported one regression and returned exit code 1. JSON, Markdown and the bundle were preserved. Comparator tests also verify increases beyond the zero floor of existing scores, lost metrics, undefined scores, changed units, changed reference conditions, changed validation declarations and numerical tolerances. Invalid comparisons retain diagnostics and return exit code 2.

The complete Python 3.12.13 suite passed 399 tests in 103.01 s. Nine upstream MuJoCo and Jupyter warnings remain. After the experiment runner adopted the validated catalogue, its focused tests and the new comparison and CLI tests passed 113 cases. Python 3.10.20 passed 92 focused automation, experiment and reporting tests in 13.38 s. Ruff passed for source, tests and examples. Mypy passed for all 60 source files. The guide's extracted module example passed four tests, Ruff and mypy. Its core profile examples executed 21 protocols twice with no skips or regressions.

The numerical environment remains NumPy 2.5.4, SciPy 1.18.1 and the optional package versions recorded in the preceding native model validation. Test and build tools remain pytest 9.1.1, Ruff 0.17.2, mypy 2.4.0 and build 1.6.1. The Python 3.10 checks use NumPy 2.2.6 and SciPy 1.15.3. No model equations or numerical defaults changed. The source distribution includes the guide, package instructions, workflow and pinned benchmark requirements. Workflow YAML, embedded Python and all nine Bash steps were syntax checked. Linux Python 3.12 dependency resolution succeeded for the full pinned requirements. The local physical preflight constructed the offline MyoSuite surrogate successfully. Hosted workflow execution remains a separate verification step after publication.

An isolated environment installed only the core CI numerical requirements and this package. MuJoCo, MyoSuite, myosuite-carla and embodied-ui were absent. The 21 protocol core profile ran in 205.015 s with no skips. All 41 of its scored metric evaluations match the corresponding committed baseline values exactly. The final catalogue guard also loads builtin registrations and requires metadata for packaged protocols outside the bench namespace. Its focused CLI, comparison, experiment and builder checks passed 109 tests. Core and full workflow executions use separate concurrency groups.

## 2026-10-27, native models and reproducible experiments (Windows, CPU)

The package includes native PyACTUp memory, native PyDDM choice distributions, an attributed MIT contact sensing subset from MIMo, SALib sensitivity analysis and a local experiment runner. The interface provides metric cards, confidence intervals, paired comparisons, sensitivity graphics and downloadable bundles. Memory and choice protocols use assumed task settings. Independent human evaluation remains pending.

The final Python 3.12 environment used Python 3.12.13, NumPy 2.5.4, SciPy 1.18.1, PyACTUp 2.2.5, PyDDM 0.9.0, SALib 1.6.0, MuJoCo 3.17.0 and TouchSim 0.1.2. Pytest 9.1.1, Ruff 0.17.2, mypy 2.4.0 and build 1.6.1 provided the checks. MyoSuite 3.0.0, myosuite-carla 0.1.0, embodied-ui 0.1.0 and myo-sim 0.2.3 supplied the optional physical integration. [The environment record](evidence/experiment-environment.json) contains exact versions, package source hashes, editable dependency identities and the pinned TouchSim Git commit.

The complete offline baseline meets every acceptance criterion in 17 of 28 protocols. Eleven protocols fail and none are skipped. The full run took 514.40 s under concurrent validation load. Driving uses the seeded myosuite-carla vehicle surrogate and the existing MyoSuite body. A targeted verification refresh checks the corrected vision, trust and risk implementation paths. Default trust and risk predictions match the former implementation exactly. Vision now measures the cutoff field returned by `perceive`.

| Declared role | Scored metric evaluations | Within acceptance range |
| --- | ---: | ---: |
| Consistency | 9 | 9 |
| Calibration | 5 | 5 |
| Held out | 38 | 19 |
| Unclassified | 1 | 1 |

These counts contain repeated reference metrics across protocols. The baseline uses 49 unique published aggregate references. Held out statistics can share participants with calibration. The modality ordering result is a consistency check because tactile latency uses the same study. Throughput uses MacKenzie's separate smartphone experiment. Emotion data independence remains unclassified. [The report](evidence/benchmark-baseline.json) includes data identifiers, conditions and the rationale for each role.

The DRT fitter returned central task chunks of 0.757758 s for 0 back, 1.127573 s for 2 back and 1.762292 s for SuRT. The committed constants use 0.758 s, 1.128 s and 1.762 s. The motion sickness refit returned a conflict scale of 0.196237 m/s², which retains the rounded default of 0.196. Its fitted endpoint RMSE is 7.463 percentage points. Held out time course RMSE is 14.758 percentage points and still fails its range. The generated parameter table retains 112 human and driver entries. Glance sampler coefficients retain the earlier fit.

```powershell
.venv/Scripts/python.exe examples/parameter_table.py
.venv/Scripts/python.exe examples/fit_drt_load.py --out evidence/drt-load-fit.json
.venv/Scripts/python.exe examples/fit_motion_sickness.py --out evidence/motion-sickness-fit.json
.venv/Scripts/python.exe -m embodied_human.bench --out evidence/benchmark-baseline.json --bundle artifacts/baseline-run
.venv/Scripts/python.exe -m pytest tests -x -q --junitxml=evidence/tests.xml
.venv/Scripts/python.exe -m ruff check src tests examples
.venv/Scripts/python.exe -m mypy src/embodied_human --ignore-missing-imports
node --check src/embodied_human/builder/builder.js
.venv/Scripts/python.exe -m build
```

The final complete suite passed 344 tests in 272.07 s. All optional native pathways ran. Nine warnings concern upstream MuJoCo model prefixes and Windows Jupyter event loop integration. Ruff, mypy and JavaScript syntax checks passed. The Windows progress file race has two actual file locking regressions. Atomic JSON replacement retries transient reader locks and preserves the previous file when the bounded wait is exhausted. An additional reporting check verifies that installing a previously absent optional dependency requires an explicit replay comparison override.

Python 3.10.20 compatibility used NumPy 2.2.6 and SciPy 1.15.3. The complete compatibility run passed 290 tests and skipped 20 unavailable optional integrations. Final focused checks passed 92 tests covering reporting, trust, risk, vision, experiments and builder jobs. Native PyACTUp, PyDDM and SALib execute in that environment. The optional CI job explicitly installs the native integrations on Python 3.10 and 3.12. The existing core matrix remains Python 3.10, 3.12 and 3.14.

Four [request files](examples/experiments/) produced the committed [experiment evidence](evidence/experiments/). Serial recall compares two configurations across five seeds (10 protocol runs). The choice example sweeps three bounds across five seeds (15 runs). Morris evaluates two DRT factors using eight trajectories and three seeds (72 runs). Sobol evaluates two eye parameters with 64 base samples (256 runs). The evidence retains all rows, design order, seed outputs and base parameters. Per case overrides reconstruct each configuration without repeating its provenance. Full bundles also retain per case resolved configurations and source archives.

The DRT Morris absolute effects are 150.0 ms for the central stage range and 69.448 ms for the tactile latency range. These effects depend on the declared bounds. The eye Sobol estimates are S1 = 1.0183 and ST = 1.0169 for duration slope. The intercept effect is numerically zero. Values above one remain visible as finite sample estimation error. The eye example measures consistency with its own coefficients. Human parameter distributions require separate empirical estimation.

Native serial recall replay used the following command without comparison overrides. Its request, every result row, summaries and paired differences matched exactly.

```powershell
.venv/Scripts/python.exe -m embodied_human.experiments --replay artifacts/serial-recall/manifest.json --out artifacts/serial-recall-replay
```

The wheel and source distribution include the MIMo licence and exact source attribution. A core environment installed the wheel with NumPy and SciPy. It constructed all 16 categories and every lazy adapter, listed 30 protocols, served the builder, ran the memory preview and wrote a checked bundle. PyACTUp, PyDDM, SALib, MuJoCo, MyoSuite, embodied-ui, TouchSim and scikit-learn were absent. Browser verification covered native recall and choice, Morris indices with recorded units, cancellation, retained jobs after reload and actual bundle downloads. Responsive summaries retain exact values in expandable tables. External dependency checkouts and assets require separate retention, as documented in [the experiment guide](docs/experiments.md).

## 2026-10-26, repository review and optional TouchSim (Windows, CPU)

All commands below used the isolated `.venv` interpreter, Python 3.12.13,
unless a different environment is named. Versions: NumPy 2.5.4, SciPy 1.18.1,
pytest 9.1.1, Ruff 0.17.2, mypy 2.4.0, build 1.6.1, setuptools 84.2.1,
MuJoCo 3.17.0, scikit-learn 1.9.1, MoSQITo 1.2.1, nbformat 5.11.1,
nbclient 0.11.0 and ipykernel 7.4.0. Optional dependencies:

- embodied-ui 0.1.0, local checkout `605862b`;
- myosuite-carla 0.1.0, local checkout `85983b1`;
- MyoSuite 3.0.0, local checkout `c0af9c6`, and myo-sim 0.2.3;
- TouchSim 0.1.2 at verified MIT commit
  `c7ece893c1ec75f2de10ade34422e0090a3093b9`, Numba 0.68.0 and
  scikit-image 0.26.0.

The [review](docs/repository-review.md) records the changes and remaining
model limitations. Numeric baseline defaults are unchanged. Memory noise
`0.25` is now honestly classified as assumed, rather than fitted, with a verified
source. `python examples/parameter_table.py` regenerated the parameter table:
112 entries including the driver, with 14 measured, 5 standard, 52 fitted,
4 derived and 37 assumed. The default human alone has 85 entries.

### Benchmark and interpretation

```powershell
.venv/Scripts/python.exe examples/parameter_table.py
.venv/Scripts/python.exe -m embodied_human.bench --out evidence/benchmark-baseline.json
```

The full seeded baseline run completed in 495.26 s: **17 of 28 pass, 11 fail,
none skipped**. Integrated driving ran offline on myosuite-carla's fitted
vehicle surrogate with MyoSuite's full body; no CARLA server was used. The
overall pass count matches the prior baseline. Key outcomes remain takeover
2.44 s, emotion RMSE 0.098, fitted vertical sickness RMSE 7.47 and held-out
sickness time-course RMSE 14.8. The latter still fails its acceptance range.
Existing takeover modality, gaze, glance-speed, DRT-load and compensated-roll
misses remain visible in the report.

Fitted/source-parameter matches remain consistency checks. In particular,
the memory span checks its own capacity source; vertical sickness fit-set
performance is separate from held-out time-course performance. No new fitted
scientific constants were introduced. Attention, estimator, eye, vestibular
and response module source files were not changed, so the module-specific
refitting triggers did not apply. A no-detected-warning takeover now stays
unanswered rather than receiving an invented seven-second response.

### Additional verification

- Final full suite:
  `.venv/Scripts/python.exe -m pytest tests -q --junitxml=evidence/tests.xml`
  passed **185 tests, none skipped**, in 126.68 s, including actual optional
  TouchSim, offline full-body driving and tutorial execution. Seven upstream
  warnings concerned MuJoCo spec prefixes and Windows/Jupyter event-loop
  integration; no test failed. [JUnit report](evidence/tests.xml).
- Python 3.10.20 compatibility used a separate `.venv-py310` with NumPy 2.2.6,
  SciPy 1.15.3 and pytest 9.1.1, without optional physics/TouchSim dependencies.
  Running `tests/test_modules.py`, `test_memory_robustness.py`,
  `test_regressions.py`, `test_report_json.py`, `test_builder.py`,
  `test_tapping_robustness.py`, `test_affect_regressions.py` and `test_touchsim.py`
  passed **122 tests, 3 skipped**, in 2.66 s. This caught and corrected a Python
  3.10 bare-`ClassVar` annotation issue in the builder. Builder tests separately
  passed all 19 cases on both Python 3.10 and Python 3.12 after that correction.
- TouchSim tests execute the actual upstream package, including repeatability,
  optional import behaviour, units/shapes and invalid-input handling. A final
  independent review reproduced upstream NaN derivatives on two-sample traces;
  the adapter now requires three original and internal-rate samples. The final
  TouchSim-focused run passed **26 tests**.
- The affect study now uses the configured implementation and overrides.
  A before/after seeded comparison (`seed=7`, two classifiers, one run,
  50 training episodes) preserved all nine baseline predictions exactly.
  Replacement classifiers/parameters and falsy forced actions have regression
  coverage. A separate default study rerun checks the selected-module path.
  `python -m embodied_human.bench --only affect.emotion_study --out artifacts/affect-review.json`
  returned the same default metrics as the complete baseline; a copy of that
  report is [evidence/affect-module-review.json](evidence/affect-module-review.json).
- `python -m ruff check src tests examples` and
  `python -m mypy src/embodied_human --ignore-missing-imports` passed.
  JavaScript syntax passed with Node 24.16.0 using
  `node --check src/embodied_human/builder/builder.js`.
- `python -m build` produced a wheel and source distribution. Wheel assets
  include the builder HTML, CSS and JavaScript; the source distribution includes
  the separately pinned TouchSim requirement, documentation, evidence and the
  tutorial notebook needed by its included tests. No direct Git dependency is
  included in package metadata, preserving PyPI compatibility.
- A clean `.venv-core` installed the wheel with only NumPy and SciPy.
  Importing/building all 16 modules, selecting lazy TouchSim, constructing the
  builder server, listing its registry and running the memory preview succeeded
  with TouchSim, MuJoCo, MyoSuite, embodied-ui and scikit-learn all absent.
- Browser QA used the local server: dragging SEEV onto attention, click/menu
  selection, editing memory capacity, source/override display, module validation,
  seeded preview and actual JSON download succeeded. The pointer-drag fallback
  was added after native HTML5 dragging failed in the first browser check.
  JSON/Python round trips, structured dataclass restoration, invalid imports and
  HTTP origin/path handling are also covered by automated tests. The memory
  preview is a demonstration; only memory affects its results.

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

## 2026-10-26: Cooper et al.'s raw DRT data

Same PC and versions. Cooper et al. (2020) pooled responses to a vibration
under the collarbone with responses to a light reflected in the
windshield. Their raw data are in `data.zip` at
https://github.com/utahcdst/Aging-Report-Frontiers.

- **Download.** Downloaded on request (5.1 MB). The repository has no
  licence, so the raw data stay outside this package. Only aggregates
  were written ([evidence/cooper-drt-aggregates.json](evidence/cooper-drt-aggregates.json)),
  by `examples/cooper_drt_aggregates.py`.
- **Procedure.** The authors' `rt.R` keeps only hits, and SuRT stimuli
  only while SuRT was in use. Averaging per session, as `rt.R` does,
  reproduces their Table 4 exactly: 526, 639, 760, 944, 703 and 881 ms.
- **Stimulus codes.** Code 2 is the vibration: it is hit in 97–98% of
  single-task trials. Code 1 is the light, the one `hr.R` scores hits for.
- **Intervals.** Participants drove up to six vehicles, so intervals use
  participant means: 65 younger and 63 older drivers.

Younger drivers, participant means with 95% intervals:

| Stimulus | Driving | 2-back cost | SuRT cost | Older / younger |
| --- | --- | --- | --- | --- |
| Vibration | 523 ms [490, 556] | +294 ms [261, 328] | +193 ms [162, 223] | 1.30 [1.20, 1.41] |
| Light | 557 ms [531, 583] | +205 ms [170, 239] | +181 ms [151, 211] | 1.19 [1.11, 1.29] |
| Pooled | 536 ms [511, 562] | +248 ms [221, 275] | +182 ms [160, 205] | 1.24 [1.16, 1.33] |

The model's DRT is tactile, so `response.drt` and `response.drt_load` now
score the vibration alone. The intervals from 65 participants are also
narrower than the earlier ones, which used n = 24.

```bash
python examples/cooper_drt_aggregates.py drt.csv --out evidence/cooper-drt-aggregates.json
python -m embodied_human.bench --out evidence/benchmark-baseline.json
```

`python -m pytest tests -q` passes 68 tests and skips 1 (the notebook
needs the `test` extra). The benchmark took 145 s; 16 of 25 pass, as
before.

| Benchmark measure | Model | Cooper et al., vibration | Result |
| --- | --- | --- | --- |
| DRT while driving | 380 ms | 523 ms (490–556) | **fail** (was a fail against the pooled 526 ms) |
| Older / younger | 1.30 | 1.30 (1.20–1.41) | pass |
| 2-back cost | +289 ms | +294 ms (261–328) | pass (held out) |
| SuRT cost | +365 ms | +193 ms (162–223) | **fail** |

The light, reported in the benchmark details, is answered 197 ms more
slowly by Cooper et al.'s drivers than by the model (557 against 360 ms).

## 2026-10-26 (later): Zhang et al., Bos & Bles, Gao & Lee, Diederich & Colonius

Same PC and versions. Four more papers were read:

- **Zhang et al. (2019):** the take-over meta-analysis.
- **Bos & Bles (1998):** the one-dimensional subjective vertical model.
- **Gao & Lee (2006):** the trust model.
- **Diederich & Colonius (2007, Exp. Brain Res.):** auditory and tactile
  non-targets with visual saccade targets.

### Changes

- **Common random numbers in TWIN.** Every response now draws one
  peripheral time per known modality, used or not.
  - **Why:** adding a vibration used to shift every later random draw. In
    100 runs that made the model's take-over 0.07 s *slower* with an extra
    signal. The race itself was correct; the comparison was not.
  - **Now:** a counterfactual (a request with and without a vibration)
    compares the same random draws.
  - **Side effects:** all response-based results shift by sampling noise.
    The take-over benchmarks now use 64 runs (32 for the modality
    benchmark), because 16 left a standard error of about 0.1 s.
- **Refit.** `hand_release` is refitted with 64 runs, in one pass: the
  hands-on time grows one for one with it. It is now 0.59 s (was 0.64 s),
  giving 1.63 s against the 1.62 s target.
- **Ko et al.** They report means, so `driving.takeover_modality` now
  scores the model's mean time to the wheel instead of its median.
- **`driving.takeover_determinants` (new).** The model's log effects of
  the request's modality and of a visual task are scored against Zhang
  et al.'s mixed-effects coefficients (Table 5). The ranges are 95% CIs
  from each estimate and its t value.
- **New tests.**
  - The one-dimensional reduction of the vestibular model reproduces Bos
    & Bles's closed-form sickness (Eqs. 5–6, Table 1) within 1%.
  - The trust model reduces to Gao & Lee's Eq. 11.
- **Checked, with no change needed.** Gao & Lee's Eq. 6 (beliefs), Eqs.
  7–9 and Table 1 match `DecisionFieldTrust` term for term. Their stated
  θ = 1.9 √(2σ²) = 0.27 implies σ = 0.1, not the σ² = 0.1 of their Table
  1, as the parameter's note already said.

### Tests

`python -m pytest tests -q` passes 70 tests and skips 1 (the notebook
needs the `test` extra). The tutorial notebook was re-executed.

### Benchmark

```bash
python examples/fit_takeover.py --out evidence/takeover-fit.json
python -m embodied_human.bench --out evidence/benchmark-baseline.json
python -m embodied_human.bench --replace vestibular=subjective_vertical:accel_gain=0,accel_feedback=0,conflict_scale=0.7 --only vestibular.vertical_sickness vestibular.sickness_time_course --out evidence/benchmark-vestibular-bos-bles.json
```

The run took 148 s; **16 of 26 benchmarks pass**.

| Benchmark | Before | Now | Human reference |
| --- | --- | --- | --- |
| driving.takeover_determinants (new) | — | icon alone +0.06, vibration −0.01, SuRT +0.42: **fail** | +0.21 (0.13–0.29), −0.18 (−0.27 to −0.09), +0.10 (0.04–0.16) |
| driving.takeover_motor_readiness | road 0.74 s: pass (16 runs) | road 0.81 s, hands 1.03 s: **fail** (64 runs) | 0.69 (0.64–0.74); 1.14 (1.05–1.23) |
| driving.takeover_components | gaze 0.48, road 0.73, hands 1.63, input 1.81 s | gaze 0.42, road 0.78, hands 1.63 (fitted), input 1.82 s | 0.41–0.54, 0.69–0.94, 1.45–1.79, 2.10–2.89 s |
| driving.takeover_modality | hands 1.52 s (median) | hands 1.62 s (mean): pass | 1.87 ± 0.51 s |
| driving.takeover (MyoSuite) | 2.03 s | 2.20 s, SD 0.80: pass | 2.72 ± 1.45 s |

The Zeeb et al. road fixation that passed before was sampling noise
from 16 runs.

### Interpretation

- **What changes take-over time.**
  - **Modality:** the request's modality changes only how fast it is
    detected, a few hundred milliseconds. People's take-over times change
    four to eighteen times more with it, which suggests the modality also
    changes how urgently they act.
  - **Task:** the model's visual task, conversely, costs too much. Without
    one, its driver is already looking at the road with nothing to let go
    of.
- **Bos & Bles's own model.** With their one-dimensional parameters
  (no internal acceleration model, b = 0.7 m/s²), the 2-hour data are
  missed by 17.6 % points (r = 0.65) and the time courses by 20.3
  (r = 0.88). The refitted six-degree-of-freedom model scores 7.5 and
  14.8.
  - **Caveat:** they describe McCauley's accelerations as RMS values,
    while the benchmark converts mean absolute accelerations to
    amplitudes. Part of the gap may be that conversion.
  - **Conflict size:** at the frequency where sickness peaks (0.167 Hz),
    the six-degree-of-freedom model's conflict is only 17% of the 1-D
    model's, because its internal acceleration model cancels most of it.
    This is why its conflict scale had to be refitted far below Bos &
    Bles's.
- **Diederich & Colonius (Exp. Brain Res.)** report crossmodal
  interaction ratios, not peripheral times per modality. Their palm
  vibration (50 Hz, 1–2 mm) was far above threshold, so it cannot
  calibrate the touch latency. Their finding that a second non-target
  adds facilitation is consistent with the race, but their
  focused-attention design (only the light triggers the saccade) is not
  what the take-over model simulates, so it is not scored.

## 2026-10-26 (evening): trust-driven monitoring, Petermeijer et al., Donohew & Griffin, steering wheel

Same PC and versions. Five more papers were read:

- **Hergeth et al. (2016)** and **Petermeijer et al. (2017)**: used.
- **Donohew & Griffin (2009)**: used.
- **Morioka & Griffin (2009, Applied Ergonomics)**: used.
- **Morioka & Griffin (2008, Acoustics'08)**: reports the seated (no
  backrest) thresholds already taken from the JSV paper, so it adds
  nothing new.

### Changes

- **Monitoring during automation.** While automation drives and a task
  is shown, the driver leaves the task to check the road at a rate of
  `monitoring_gain` × (1 − trust).
  - **Before:** the attention module sampled the road by lane
    uncertainty, as if the driver were steering.
  - **Fit:** `examples/fit_monitoring.py` fits the gain to Hergeth et
    al.'s two measurements (median 0.13 checks/s at a mean trust of 74%,
    0.097/s at 82%). It gives 0.62/s; the model's medians are 0.125 and
    0.100.
  - **Road glances:** a check on the road lasts 0.6 s on average (Zeeb et
    al., measured).
- **Trust prior.** The trust module's initial belief in automation is now
  0.744, Hergeth et al.'s first rating (it was an assumed 1.0). The
  capability a take-over request reveals is re-derived from Kraus et
  al.'s first request (0.752 to 0.702): 0.55, was 0.8.
- **`driving.automation_monitoring` measurement fix.** It now measures from
  the per-sample gaze record. The previous version dropped looks away still
  under way at the end of the window, which hid long looks at the task.
- **`driving.takeover_eyes_by_modality` (new).** It tests Petermeijer et
  al.'s eyes-on-road differences between a light and a sound or a
  vibration. The 95% CIs are derived from their paired p values.
- **`vestibular.roll_compensation` (new).** It simulates Donohew &
  Griffin's 30 min lateral oscillations, with and without the roll that
  cancels the lateral force at the seat (Table I), with the head 0.795 m
  above the roll centre. Two tests check the motion: zero lateral force at
  the seat, and 7.30° of peak roll at 0.2 Hz.
- **Steering-wheel touch.**
  - **Thresholds:** from Morioka & Griffin's (2009) Table 2, now
    tabulated, for both hand positions. The 2007 figure reading agreed
    within 2%.
  - **Perceived intensity:** `perceived_intensity` gives it by Stevens'
    power law with their exponents and constants. A test reproduces
    their Fig. 4 example and their crossover: 16 Hz feels stronger than
    125 Hz at 1.0 m/s², weaker at 0.2 m/s².
- **Hand release.** `hand_release` was refitted to the new monitoring; it
  stays at 0.59 s.

### Tests

`python -m pytest tests -q` passes 73 tests and skips 1 (the notebook
needs the `test` extra). The tutorial notebook was re-executed: its last
section now compares two trust levels, since attention modules no longer
decide when to look up during automation.

### Benchmark

```bash
python examples/fit_monitoring.py --out evidence/monitoring-fit.json
python examples/fit_takeover.py --out evidence/takeover-fit.json
python -m embodied_human.bench --out evidence/benchmark-baseline.json
```

The run took 602 s (148 s before, with other load on the machine);
**17 of 28 benchmarks pass**.

| Benchmark | Before | Now | Human reference |
| --- | --- | --- | --- |
| driving.automation_monitoring | longest look away 1.8 s, road 36%: fail | 15.9 s, 11.6%, 8.4 road glances/min: **pass** | 8.6–44.4 s, 2.2–12.5% (Zeeb et al.) |
| driving.takeover_eyes_by_modality (new) | — | light − sound +0.07 s, light − vibration −0.01 s: **fail** | +0.37 (0.10–0.64), +0.50 (0.18–0.82) |
| vestibular.roll_compensation (new) | — | 5 of 5 frequency differences: pass; compensated/uncompensated at 0.315 Hz 0.49: **fail** | 5 of 5; sicker with compensation |
| driving.takeover_components | gaze 0.42, road 0.78, input 1.82 s | gaze 0.81 (**fail**), road 0.91, input 1.98 s | 0.41–0.54, 0.69–0.94, 2.10–2.89 s |
| driving.takeover_motor_readiness | road 0.81, hands 1.03 s | road 0.91, hands 1.03 s: **fail** | 0.64–0.74; 1.05–1.23 |
| driving.takeover_determinants | +0.06, −0.01, +0.42 | +0.05, −0.01, +0.50: **fail** | +0.21, −0.18, +0.10 |
| driving.takeover (MyoSuite) | 2.20 s | 2.44 s, SD 0.84: pass | 2.72 ± 1.45 s |

### Interpretation

- **Monitoring.** One fitted gain, with the trust prior measured in the
  same study, predicts Zeeb et al.'s held-out monitoring. The model
  closes the gap from the overview.
- **Gaze reaction.** Gold et al.'s gaze reaction now fails. Earlier, the
  driver's frequent, uncertainty-driven glances often left SuRT just after
  the request by chance. Absorbed in the task, it looks up only after
  reacting, which takes the reaction (0.56 s with the central wait) plus a
  saccade latency.
- **Modality.** Petermeijer et al.'s data confirm Zhang et al.'s
  meta-analysis: in the model, a request's modality changes only its
  detection.
- **Roll compensation.** The six-degree-of-freedom model orders the
  frequencies of roll-compensated motion correctly. It predicts less
  sickness with compensation at 0.315 Hz, where people got more. Kamiji
  et al.'s own validation of the model used uncompensated lateral motion.
