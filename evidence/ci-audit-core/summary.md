# Benchmark evaluation

Seed: 0. Protocols: 21. Pass: 13. Fail: 8. Skipped: 0. Unscored: 0.

Acceptance results describe the declared published comparisons. Regression gates compare software outcomes with the selected baseline.

| Evaluation role | Scored metric evaluations | Within range |
| --- | ---: | ---: |
| Consistency | 8 | 8 |
| Calibration | 3 | 3 |
| Held out | 27 | 14 |
| Exploratory | 2 | 0 |
| Unclassified | 1 | 1 |

Counts can contain repeated references across protocols.

## Baseline comparison

Regressions: 7. Incomparable: 2. Comparable metrics: 39. Numerically changed metrics: 14.

Reference error is measured in acceptance range widths. Regression tolerance: 1e-09. Existing discrepancies remain visible.

Changed configurations: vestibular.

* driving.takeover_components / driving.takeover_gaze: distance from acceptance range increased.
* driving.takeover_determinants / driving.zhang_no_auditory: distance from acceptance range increased.
* driving.takeover_determinants / driving.zhang_vibrotactile: distance from acceptance range increased.
* driving.takeover_determinants / driving.zhang_visual_task: distance from acceptance range increased.
* driving.takeover_motor_readiness / driving.zeeb_hands: distance from acceptance range increased.
* driving.takeover_motor_readiness / driving.zeeb_road: distance from acceptance range increased.
* response.drt_load / response.drt_surt_cost: distance from acceptance range increased.
* driving.glance_speed / driving.incar_glance_speed_elasticity: reference or validation declaration changed.
* driving.glance_speed / driving.occlusion_speed_elasticity: reference or validation declaration changed.

| Protocol / metric | Baseline | Current | Delta | Unit | Role | Change |
| --- | ---: | ---: | ---: | --- | --- | --- |
| affect.emotion_study / affect.emotion_rmse | 0.0979506 | 0.0979506 | 0 | intensity | unclassified | unchanged |
| driving.automation_monitoring / driving.automation_max_offroad | 15.8703 | 15.8703 | 0 | s | held_out | unchanged |
| driving.automation_monitoring / driving.automation_road_share | 11.5622 | 11.5622 | 0 | % | held_out | unchanged |
| driving.glance_population / driving.human_glance_sd | 0.582351 | 0.582351 | 0 | s | calibration | unchanged |
| driving.glance_population / driving.human_long_glance_share | 0.0793451 | 0.0793451 | 0 | fraction | held_out | unchanged |
| driving.glance_population / driving.human_mean_glance | 1.13194 | 1.13194 | 0 | s | calibration | unchanged |
| driving.glance_population / driving.long_glance_share | 0.0793451 | 0.0793451 | 0 | fraction | held_out | unchanged |
| driving.glance_population / driving.mean_offroad_glance | 1.13194 | 1.13194 | 0 | s | held_out | unchanged |
| driving.glance_speed / driving.incar_glance_speed_elasticity | -0.45327 | -0.45327 | undefined | 1 | exploratory | incomparable |
| driving.glance_speed / driving.occlusion_speed_elasticity | -0.45327 | -0.45327 | undefined | 1 | exploratory | incomparable |
| driving.takeover_components / driving.takeover_gaze | 0.806884 | 1.02457 | 0.217685 | s | held_out | regressed |
| driving.takeover_components / driving.takeover_hands | 1.63 | 1.62063 | -0.009375 | s | calibration | changed |
| driving.takeover_components / driving.takeover_intervention | 1.97219 | 2.25437 | 0.282187 | s | held_out | improved |
| driving.takeover_components / driving.takeover_road | 0.90125 | 0.9375 | 0.03625 | s | held_out | changed |
| driving.takeover_determinants / driving.zhang_no_auditory | 0.051721 | 0.0477731 | -0.00394792 | ln ratio | held_out | regressed |
| driving.takeover_determinants / driving.zhang_vibrotactile | -0.00747518 | -0.00486348 | 0.0026117 | ln ratio | held_out | regressed |
| driving.takeover_determinants / driving.zhang_visual_task | 0.502306 | 0.865812 | 0.363507 | ln ratio | held_out | regressed |
| driving.takeover_eyes_by_modality / driving.petermeijer_visual_vs_auditory | 0.0709091 | 0.0739623 | 0.00305317 | s | held_out | improved |
| driving.takeover_eyes_by_modality / driving.petermeijer_visual_vs_tactile | -0.00763636 | 0.00339623 | 0.0110326 | s | held_out | improved |
| driving.takeover_modality / driving.takeover_hands_auditory | 1.61969 | 1.62407 | 0.004375 | s | held_out | changed |
| driving.takeover_modality / driving.takeover_modality_order | 1 | 1 | 0 | fraction | held_out | unchanged |
| driving.takeover_motor_readiness / driving.zeeb_hands | 1.02875 | 1.39938 | 0.370625 | s | held_out | regressed |
| driving.takeover_motor_readiness / driving.zeeb_road | 0.902545 | 1.08618 | 0.183636 | s | held_out | regressed |
| eyes.latency / eyes.latency_median | 203.01 | 203.01 | 0 | ms | consistency | unchanged |
| eyes.main_sequence / eyes.duration_intercept | 21 | 21 | 0 | ms | consistency | unchanged |
| eyes.main_sequence / eyes.duration_slope | 2.2 | 2.2 | 0 | ms/deg | consistency | unchanged |
| fatigue.endurance / fatigue.endurance_error | 0.159217 | 0.159217 | 0 | \|ln ratio\| | consistency | unchanged |
| memory.span / memory.span | 4 | 4 | 0 | items | consistency | unchanged |
| objects.peripheral_acuity / objects.e2 | 1.97551 | 1.97551 | 0 | deg | consistency | unchanged |
| response.drt / response.drt_tactile | 380.297 | 380.297 | 0 | ms | held_out | unchanged |
| response.drt / response.drt_visual | 382.925 | 382.925 | 0 | ms | held_out | unchanged |
| response.drt / response.older_slowing | 1.29835 | 1.29835 | 0 | ratio | held_out | unchanged |
| response.drt_load / response.drt_2back_cost | 297.026 | 298.063 | 1.03721 | ms | held_out | changed |
| response.drt_load / response.drt_driving_cost | 28.3121 | 28.3121 | 0 | ms | held_out | unchanged |
| response.drt_load / response.drt_surt_cost | 367.348 | 574.665 | 207.318 | ms | held_out | regressed |
| response.eccentricity / response.eccentricity_rmse | 1 | 1 | 0 | ms | held_out | unchanged |
| response.modality_rt / response.modality_order | 1 | 1 | 0 | fraction | consistency | unchanged |
| risk.lane_width / risk.lane_width_speed_error | 2.13347 | 2.13347 | 0 | km/h | held_out | unchanged |
| trust.kraus_dynamics / trust.malfunction_dip | 0.19739 | 0.19739 | 0 | scale points | held_out | unchanged |
| trust.kraus_dynamics / trust.study2_rmse | 0.290007 | 0.290007 | 0 | scale points | held_out | unchanged |
| vision.foveation / vision.e2 | 2.3 | 2.3 | 0 | deg | consistency | unchanged |
