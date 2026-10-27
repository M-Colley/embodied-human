# Benchmark failure audit

The audit corrected response censoring, event timing, exposure accounting and a source transcription error. The 28 protocols retain their original numerical acceptance ranges. The [full comparison report](../evidence/benchmark-failure-audit.md) records all metric changes against the [archived baseline](../evidence/benchmark-before-failure-audit.json). Several failures require additional models and task replications. Their source values remain visible.

The corrected run passes 17 protocols and fails 11. Eighteen of 53 scored metric evaluations remain outside their ranges, compared with 19 before the audit. Gold's intervention time improves from 1.972 to 2.254 seconds and enters its published range. Nine other metric evaluations move farther outside their ranges after the corrected measurement and required refits. The comparison report retains those regression flags. Two glance speed evaluations are incomparable with their old declarations because their role now explicitly records the task mismatch. Their values and numerical ranges are unchanged.

| Failing protocol | Current discrepancy | Remaining scientific requirement |
| --- | --- | --- |
| `driving.surt_glances` | Long glance share 0.0222, range 0.03 to 0.13 | Repeated runs and a matched human secondary task |
| `response.drt_load` | Driving cost 28.3 ms and tactile SuRT cost 574.7 ms | Task specific demand and stimulus models; the two back cost passes |
| `trust.kraus_dynamics` | Malfunction response positive 0.197, observed negative 0.47 | A mechanism for transient negative evidence and transparency |
| `response.drt` | Visual 382.9 ms and tactile 380.3 ms | Matched sensory conditions and task response preparation |
| `vestibular.sickness_time_course` | RMSE 14.839 percentage points, limit 10 | A justified temporal or frequency response extension |
| `vestibular.roll_compensation` | Predicted compensated ratio 0.492, opposite observed direction | Head dynamics and evaluation with compatible sickness outcomes |
| `driving.glance_speed` | One SuRT slope negative 0.453 against two different tasks | Separate road occlusion and letter search protocols |
| `driving.takeover_components` | Gaze reaction 1.025 s, range 0.41 to 0.54 | Independently calibrated gaze and manual effector coupling |
| `driving.takeover_motor_readiness` | Road 1.086 s and hands 1.399 s | Rotary task demand, geometry and readiness conditions |
| `driving.takeover_determinants` | Three effects disagree with the meta analysis | Urgency, task interference and comparable factor contrasts |
| `driving.takeover_eyes_by_modality` | Visual differences 0.074 s and 0.003 s | Warning detectability, timing and the original secondary tasks |

All 64 fast takeover runs complete in each scored determinant condition. Gold's road, hand and intervention outcomes also have 64 observed completions. Its gaze outcome has 52 eligible observed reactions. The remaining runs began outside a task display fixation. The full body takeover mean is 2.865 seconds and remains within its broad meta analytic range. These integrated means do not establish agreement for the component failures.

The baseline comparison gate still reports the nine worsened evaluations and two declaration changes. A comparison against the preaudit revision therefore requires review. The committed corrected baseline preserves the measured results for subsequent software regression checks. The audit supplies no override that suppresses changed human reference errors.

## Takeover measurements and effector coupling

The first road fixation was gated on completion of the manual response stage. This delayed the recorded eye event when gaze reached the road earlier. The log now records the first observed road fixation after the request. A gaze reaction is eligible only when gaze was on the task display at the request. This prevents a later display departure from being counted as the initial gaze reaction for someone already watching the road. [Gold et al. (2013)](https://doi.org/10.1177/1541931213571433), [Zeeb et al. (2015)](https://doi.org/10.1016/j.aap.2015.02.023).

The hand log also recorded a planned movement completion after an episode had ended. It now records completion only when the episode observes that time. The planned completion remains a separate diagnostic. The fast takeover simulation allows ten seconds after the default request and stops once engagement occurs. Component counts and completion counts identify missing outcomes. The Petermeijer comparison now applies the published exclusion of responses below 0.3 seconds directly and reports retained and discarded counts. Its SuRT task still differs from the source's reading and video tasks. Its stimulus amplitudes and visual geometry also contain assumptions. [Petermeijer et al. (2017), Methods and Results](https://repository.tudelft.nl/file/File_d111583a-c624-49c6-94b7-aba58ccf1051).

The default serial coupling still gates eye commands on the complete response latency. `parallel_gaze_preparation=True` provides an experimental coupling through peripheral detection and concurrent task waiting. Its source label is assumed. TWIN's second stage combines integration and response preparation. Saslow's saccade latency also includes several processing stages. Neither source identifies an isolated manual preparation delay that can simply be removed. The optional coupling therefore needs calibration and independent evaluation before a validity claim. Custom response modules without stage decomposition retain their total latency, with the applied fallback recorded explicitly. [Diederich and Colonius (2015), p. 2](https://uol.de/f/6/dept/psycho/ag/kogn/colonius/Diederich_colonius2015PsychReview_TWINrelatingRTtoTOJ.pdf), [Saslow (1967)](https://doi.org/10.1364/JOSA.57.001024).

The required DRT refit changes the SuRT chunk from 1.762 to 2.484 seconds. The two back chunk changes from 1.128 to 1.130 seconds. Both remain fitted only to Conti's secondary task contrasts. Those contrasts subtract the observed driving baseline, so the driving observations are shared with calibration. The hand release fit then changes from 0.59 to 0.20 seconds and reproduces Gold's calibration target of 1.62 seconds with a mean of 1.620628 seconds. Its fitter checks a local 10 millisecond grid to account for discrete simulation steps. None of these fits targets the reserved Zeeb or Petermeijer outcomes. Transfer from Conti's keypad task to other SuRT implementations remains an assumption.

## Glance exposure and long glance share

An unfinished final glance previously disappeared from all exposure calculations. Completed glance duration statistics still exclude this censored duration. The observed portion now contributes to eyes off road time for any completed task screen that overlaps it. Reports include the number of censored glances and their observed exposure. The full body SuRT protocol still uses a single 60 second run at 72 km/h. Its human long glance reference comes from a different task context. A small run supplies a noisy estimate of a rare long glance proportion. Repeated seeds, longer exposure and a matched human task are required before interpreting this failure as a stable population effect. The protocol and its acceptance range remain unchanged.

## Response input validation

Response parameters now reject nonfinite values and impossible durations. Signals preserve positive infinite latency as the nondetection sentinel. A central task with zero period remains inactive. A task occupying its entire positive period now returns an infinite central wait. The previous ten period construction falsely released such a continuously occupied stage. Seeded tests confirm that the valid default response draws remain unchanged.

## Trust

The `trust.kraus_dynamics` failure remains an empirical limitation of the current model and event representation. The audit found no measurement indexing error. Kraus et al. report Study 1 in the order `tpre, t0, t1, t2, m1, t3, t4, t5, t6, t7, m2, tpost`, with the malfunction immediately before `t4`. This matches `kraus_events(1, group)` and places the malfunction response at array index 6. Study 2 removes the manual takeover measurements, so its order is `tpre, t0, t1, t2, t3, t4, t5, t6, t7, t8`; the scored dip correctly subtracts index 4 from index 5. Tables 2 and 7 agree with the checked in aggregate values. [Kraus et al. primary paper, Figure 2, Tables 2 and 7, pp. 724 to 730](https://www.uni-ulm.de/fileadmin/website_uni_ulm/iui.f3/kraus-et-al-2019-the-more-you-know-trust-dynamics-and-calibration-in-highly-automated-driving-and-the-effects-of-take.pdf)

Practice differs between the studies. In Study 1, participants experienced a takeover request and malfunction during practice. In Study 2, participants in the low transparency groups did not experience a practice malfunction, while those in the high transparency groups did. Representing every practice interval by one `ok` observation is a simplifying protocol assumption, rather than a literal reconstruction of these experiences. A rise in average trust over practice does not establish that all events in practice were positive.

The baseline calibrates `ok`, `tor` and `malfunction` capability observations and `opacity` on Study 1 only, with inertia fixed at 0.5 and protocol noise disabled. It minimizes the unweighted sum of squared residuals over both groups at all 11 measurements after `tpre`. Initial trust is conditioned on each group's observed `tpre` mean. Study 2 scores use the two low transparency groups; high transparency predictions are descriptive because the model has no transparency mechanism. The apparently positive fitted malfunction capability is the optimizer's compromise between a brief dip and rapid subsequent recovery under the model's two smoothing stages. It is not evidence that a malfunction improved human trust.

| Diagnostic | Study 1 RMSE | Study 2 RMSE | Predicted malfunction dip |
| --- | ---: | ---: | ---: |
| Current baseline | 0.207750 | 0.290007 | +0.197390 |
| Also fit inertia on Study 1 | 0.207309 | 0.290089 | +0.208535 |
| Fit only the main experimental interval | 0.179130 | 0.180600 | +0.131716 |

All values use the study's 1 to 7 trust scale. The second diagnostic fits inertia within 0 to 1 and converges to 0.446151, while opacity becomes 2.241258. The last diagnostic conditions on observed `t0` in each study and fits Study 1 responses through `t7`, excluding practice and the second takeover sequence. Its Study 2 RMSE excludes practice too, so it is not directly comparable to the full baseline RMSE. Every diagnostic still predicts a positive malfunction response. The observed Study 2 low transparency response is `5.13 minus 5.60`, or negative 0.47. Acceptance ranges remain unchanged.

These diagnostics inspected Study 2 and cannot independently validate a model choice selected from their results. No source equations, fitted defaults, event indexing or acceptance criteria were changed by this audit. A future extension that represents faster event responses or transparency needs a justified mechanism and independent validation. Selecting a slower or faster update setting solely because its Study 2 dip has the desired sign would reuse the reserved observations for model selection.

The diagnostic was run on 27 October 2026 at source commit `9d38317ec9f0e07c5a393212ab5d8fae210f5ac8`, using Python 3.12.13, NumPy 2.5.4 and SciPy 1.18.1. The following script reproduces the three rows with the existing benchmark helpers. The parameter vector is ordered as `ok, tor, malfunction, opacity`, followed by inertia when fitted.

```python
import numpy as np
from scipy.optimize import least_squares
from embodied_human.bench.motivation import (
    KRAUS_STUDY1, KRAUS_STUDY2, kraus_events, trust_course,
)


def diagnostic(free_inertia=False, main_interval=False):
    def courses(x, study, data):
        capability = dict(zip(("ok", "tor", "malfunction"), x[:3]))
        options = {"opacity": x[3], "inertia": x[4] if free_inertia else 0.5}
        return {
            group: trust_course(
                observed[1] if main_interval else observed[0],
                kraus_events(study, group)[1:9]
                if main_interval else kraus_events(study, group),
                capability, **options,
            )
            for group, observed in data.items()
        }

    def residuals(x):
        return np.array([
            model - observed
            for group, course in courses(x, 1, KRAUS_STUDY1).items()
            for model, observed in zip(
                course[1:], KRAUS_STUDY1[group][2:10]
                if main_interval else KRAUS_STUDY1[group][1:],
            )
        ])

    solution = least_squares(
        residuals, [0.85, 0.6, 0.3, 3.0] + ([0.5] if free_inertia else []),
        bounds=([0, 0, 0, 1] + ([0] if free_inertia else []),
                [1, 1, 1, 50] + ([1] if free_inertia else [])),
    )
    predicted = courses(solution.x, 2, KRAUS_STUDY2)
    errors = [
        model - observed
        for group in ("MF-/LT", "MF+/LT")
        for model, observed in zip(
            predicted[group][1:], KRAUS_STUDY2[group][2:10]
            if main_interval else KRAUS_STUDY2[group][1:],
        )
    ]
    i = 4 if main_interval else 5
    print(solution.x, np.sqrt(np.mean(solution.fun**2)),
          np.sqrt(np.mean(np.square(errors))),
          predicted["MF+/LT"][i] - predicted["MF+/LT"][i - 1])


diagnostic()
diagnostic(free_inertia=True)
diagnostic(main_interval=True)
```

## Vestibular time course

The source implements the original three integrator SVC model, called In3 in Wada and Bos (2025). Their paper also introduces an In1 variant. Its derivative agrees term for term with Appendix A, equations A1 through A8, including the acceleration and angular velocity feedback integrators. Replacing these integrators would change the scientific model. The newer paper identifies stability limitations of In3, but does not demonstrate that changing to In1 resolves these benchmark observations. No such replacement or parameter adjustment was made. [Primary equations](https://link.springer.com/article/10.1007/s00422-025-01018-0#App1)

There was a real timestamp error. `SubjectiveVertical._run` previously stored the state after the first integration interval at output index zero. The time course benchmark reads index `minutes * 60 / dt`, so predictions were one interval ahead of the requested time. The corrected implementation stores the rest state at sample zero and integrates each subsequent interval once. The vertical endpoint benchmark, its fitting helper and the roll motion grid now include the final exposure sample so that their declared durations remain exact. Regression tests compare the output with an independent analytic conflict solution and numerical quadrature of its exact sickness convolution.

There was also a transcription error. McCauley et al. (1976), Appendix B, printed page 57, datum 17, shows observed MSI 28 at 0.180 Hz, 0.170 rms g and 15 minutes. The repository used 23. The scan is imperfect, but inspection at 220 dpi reads 28; its separately printed difference of positive 6 is inconsistent with the predicted value 21 and must not be used to silently reconstruct another observation. The observed column now supplies 28, with that inconsistency recorded in the source. The remaining selected entries match the printed observed column. [Primary scanned report, Appendix B](https://www.vliz.be/imisdocs/publications/243026.pdf)

After these corrections, using the unchanged conflict scale 0.196 and step 0.1 seconds, time course RMSE is 14.833032 percentage points, correlation is 0.757947 and the descriptive McCauley model has RMSE 6.335577 over the same 43 points. The acceptance limit stays 10. These corrections make the benchmark more accurate but do not resolve the empirical failure. The scale remains calibrated only on the 14 vertical endpoint conditions; the time course observations have not been used to refit it.

The mismatch also has an analytic explanation. For translation without rotation, let `d` be signed vertical conflict, `Ka` the acceleration gain, `Kac` the acceleration feedback, `Kvc` the vertical feedback and `tau` the vertical lag. Reducing the existing equations gives the transfer function `d(s)/a(s) = (1 minus Ka) s / (tau s squared + (1 + Kac tau) s + Kvc)`. Its sinusoidal amplitude and average Hill drive can be calculated independently of the simulation. With the current parameters, 0.6 Hz at 0.444 rms g gives conflict amplitude 0.299310 m/s² and asymptotic average MSI drive 38.434348 percent. At 0.7 Hz and 0.555 rms g they are 0.319340 m/s² and 40.536839 percent. Those large predicted drives explain the excessive long exposure predictions at high frequencies; reducing the integration step or correcting a sample index cannot remove them. A new regression test checks the full default transfer function at five frequencies, including the high frequency conditions.

## Vestibular roll compensation

The motion reconstruction follows the published seat compensation protocol: eight compensated frequencies, peak lateral velocity 1 m/s through 0.2 Hz, peak jerk 1.96 m/s³ at higher frequencies, and a roll centre at the seat surface. The sign convention cancels lateral specific force there and obeys the gravity transport relation `dg/dt = minus angular_velocity cross g`. The analytical head acceleration includes both tangential and centripetal terms for the stated rigid head assumption. A source inspection and the existing seat force cancellation test found no reversed sign or missing rotation term. [Donohew and Griffin primary paper, Methods and Table I](https://resource.isvr.soton.ac.uk/staff/pubs/PubPDFs/Pub11351.pdf)

After the timestamp and exposure endpoint corrections, with step 0.05 seconds and unchanged parameters, compensated MSI at 0.315 Hz is 2.914980 percent and uncompensated MSI is 5.925083 percent, giving ratio 0.491973. All five frequency orderings pass; the compensation direction still fails. A coarser 0.1 second diagnostic also gives ratio 0.491788, so the direction is robust to this numerical resolution check.

This comparison has limits. The paper measured illness ratings and time to nausea, while the model predicts vomiting incidence. Its Cox interaction coefficient 3.707 is not a measured vomiting incidence ratio. The existing broad ratio acceptance therefore tests direction across different sickness outcomes, not quantitative agreement with that coefficient. The paper also warns that subjects do not move as rigid bodies at high frequencies. The benchmark currently assumes a rigid head 0.795 m above the seat and has no measured head motion or postural controller. Reversing the tilt sign, substituting seat acceleration for head acceleration or fitting to the observed ratio would alter the physical protocol or reuse evaluation data. The acceptance range remains unchanged.

The corrections and diagnostics used Python 3.12.13, NumPy 2.5.4 and SciPy 1.18.1 on 27 October 2026. The following scoped commands passed five new tests, Ruff and mypy for the affected sources, and reproduced the two remaining empirical failures. Full fitting and baseline regeneration are recorded separately in `VALIDATION.md` when complete.

```powershell
.venv/Scripts/python.exe -m pytest tests/test_vestibular_timestamps.py -q
.venv/Scripts/python.exe -m ruff check src/embodied_human/vestibular.py src/embodied_human/bench/senses.py tests/test_vestibular_timestamps.py
.venv/Scripts/python.exe -m mypy src/embodied_human/vestibular.py src/embodied_human/bench/senses.py --ignore-missing-imports
.venv/Scripts/python.exe -m embodied_human.bench --only vestibular.sickness_time_course vestibular.roll_compensation --out evidence/benchmark-vestibular-audit.json
```

## DRT measurement

The DRT benchmarks averaged simulated responses outside the published hit windows. Conti et al. retained responses from 100 to 2000 ms. Cooper et al. retained responses from 100 to 2500 ms. The corrected measurements apply each study's window. Hit rates, premature response rates, late response rates and uncensored means remain in the report. [Conti et al., Methods](https://www.hfes-europe.org/wp-content/uploads/2014/06/Conti.pdf), [Cooper et al., Dependent Measures](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2020.01154/full)

The measurement correction preserves every latency draw. It pools valid responses with their stimulus and gaze probabilities. Components with fewer hits contribute less to the conditional response mean. Averaging conditional component means without accounting for their hit probabilities would introduce another measurement error. Tests cover both response window boundaries, missing responses and mixtures with different hit rates.

The following diagnostic uses seed zero and the existing chunk constants before refitting. The visual DRT mean remains 382.925 ms. The tactile mean remains 380.297 ms. Their samples contained no responses outside the respective windows. The driving cost remains 28.312 ms, and the tactile 2 back cost remains 297.026 ms. The tactile SuRT cost changes from 367.348 to 366.102 ms. The visual SuRT mean changes from 780.509 to 712.241 ms because only 95.326% of responses fall within Conti's window. Therefore, the measurement correction changes the calibration objective for the SuRT chunk. The fitted constants require regeneration through `examples/fit_drt_load.py` before interpreting a complete revised baseline. Acceptance ranges remain unchanged.

The remaining tactile latency difference has no established correction from this audit. The model assumes a vibration 20 dB above threshold and transfers tactile latency estimates from another stimulus configuration. The experiment uses a body worn vibrator with an unmodelled site and intensity mapping. Adding a constant delay to match its observed mean would fit the evaluation observations. The direct touchscreen SuRT used by Cooper also differs from Conti's keypad task. A common fitted chunk and screen period remain assumptions across these protocols.

TWIN's central stage combines integration and response preparation. Its published parameter does not isolate perceptual detection from manual preparation. `Reaction` now exposes peripheral, central and concurrent task times while preserving the sampled total. Unspecified stages remain `None` for existing response implementations. This decomposition supports explicit effector mappings in the driver. It does not change the TWIN equations or fitted defaults. [Diederich and Colonius, model definition, p. 2](https://uol.de/f/6/dept/psycho/ag/kogn/colonius/Diederich_colonius2015PsychReview_TWINrelatingRTtoTOJ.pdf)

## Glance speed protocol mismatch

The current benchmark compares one SuRT speed slope with two different human tasks. Senders et al. measured voluntary road occlusion without a secondary search task. Jokinen et al. used letter triplets with six or nine items and target present or absent trials. Their task requires serial search and an explicit completion response. The simulated pop out SuRT does not reproduce these conditions. Both scores remain visible, and the report labels the comparisons as exploratory diagnostics. [Senders et al., experiment description](https://onlinepubs.trb.org/Onlinepubs/hrr/1967/195/195-002.pdf), [Jokinen et al., Method and Appendix B](https://journals.sagepub.com/doi/full/10.1177/0018720820927687)

The search paper also contains a speed inconsistency. Its Methods specifies 60 and 120 km/h. Appendix B describes the speed predictor as 80 versus 120 km/h. The study's task data must resolve this before implementing its conditions. Its speed statistic is a coefficient from a multilevel model, while the current benchmark reports a log elasticity over five speeds. The present acceptance interval is an assumed tolerance around zero and cannot remove that measurement difference.

Separate road occlusion and letter search protocols are required for task matched evaluation. They must specify road viewing opportunities, search geometry, target absence, completion rules and speed conditions. Existing calibration data can define unchanged module parameters. Selecting new mechanisms after examining the current failures requires an exploratory or calibration label and subsequent evaluation with independent observations.
