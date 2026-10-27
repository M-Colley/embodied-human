# Benchmark evaluation

Seed: 0. Protocols: 28. Pass: 17. Fail: 11. Skipped: 0. Unscored: 0.

Acceptance results describe the declared published comparisons. Regression gates compare software outcomes with the selected baseline.

| Evaluation role | Scored metric evaluations | Within range |
| --- | ---: | ---: |
| Consistency | 9 | 9 |
| Calibration | 5 | 5 |
| Held out | 38 | 19 |
| Unclassified | 1 | 1 |

Counts can contain repeated references across protocols.

## Baseline comparison

Regressions: 0. Incomparable: 0. Comparable metrics: 53. Numerically changed metrics: 0.

Reference error is measured in acceptance range widths. Regression tolerance: 1e-09. Existing discrepancies remain visible.


| Protocol / metric | Baseline | Current | Delta | Unit | Role | Change |
| --- | ---: | ---: | ---: | --- | --- | --- |
| affect.emotion_study / affect.emotion_rmse | 0.0979506 | 0.0979506 | 0 | intensity | unclassified | unchanged |
| design.touch_key_size / design.touch_key_asymptote | 12.5 | 12.5 | 0 | mm | held_out | unchanged |
| driving.automation_monitoring / driving.automation_max_offroad | 15.8703 | 15.8703 | 0 | s | held_out | unchanged |
| driving.automation_monitoring / driving.automation_road_share | 11.5622 | 11.5622 | 0 | % | held_out | unchanged |
| driving.glance_population / driving.human_glance_sd | 0.582351 | 0.582351 | 0 | s | calibration | unchanged |
| driving.glance_population / driving.human_long_glance_share | 0.0793451 | 0.0793451 | 0 | fraction | held_out | unchanged |
| driving.glance_population / driving.human_mean_glance | 1.13194 | 1.13194 | 0 | s | calibration | unchanged |
| driving.glance_population / driving.long_glance_share | 0.0793451 | 0.0793451 | 0 | fraction | held_out | unchanged |
| driving.glance_population / driving.mean_offroad_glance | 1.13194 | 1.13194 | 0 | s | held_out | unchanged |
| driving.glance_speed / driving.incar_glance_speed_elasticity | -0.45327 | -0.45327 | 0 | 1 | held_out | unchanged |
| driving.glance_speed / driving.occlusion_speed_elasticity | -0.45327 | -0.45327 | 0 | 1 | held_out | unchanged |
| driving.surt_glances / driving.human_long_glance_share | 0.0222222 | 0.0222222 | 0 | fraction | held_out | unchanged |
| driving.surt_glances / driving.human_mean_glance | 0.930222 | 0.930222 | 0 | s | calibration | unchanged |
| driving.surt_glances / driving.long_glance_share | 0.0222222 | 0.0222222 | 0 | fraction | held_out | unchanged |
| driving.surt_glances / driving.mean_offroad_glance | 0.930222 | 0.930222 | 0 | s | held_out | unchanged |
| driving.takeover / driving.takeover_time | 2.435 | 2.435 | 0 | s | held_out | unchanged |
| driving.takeover_components / driving.takeover_gaze | 0.806884 | 0.806884 | 0 | s | held_out | unchanged |
| driving.takeover_components / driving.takeover_hands | 1.63 | 1.63 | 0 | s | calibration | unchanged |
| driving.takeover_components / driving.takeover_intervention | 1.97219 | 1.97219 | 0 | s | held_out | unchanged |
| driving.takeover_components / driving.takeover_road | 0.90125 | 0.90125 | 0 | s | held_out | unchanged |
| driving.takeover_determinants / driving.zhang_no_auditory | 0.051721 | 0.051721 | 0 | ln ratio | held_out | unchanged |
| driving.takeover_determinants / driving.zhang_vibrotactile | -0.00747518 | -0.00747518 | 0 | ln ratio | held_out | unchanged |
| driving.takeover_determinants / driving.zhang_visual_task | 0.502306 | 0.502306 | 0 | ln ratio | held_out | unchanged |
| driving.takeover_eyes_by_modality / driving.petermeijer_visual_vs_auditory | 0.0709091 | 0.0709091 | 0 | s | held_out | unchanged |
| driving.takeover_eyes_by_modality / driving.petermeijer_visual_vs_tactile | -0.00763636 | -0.00763636 | 0 | s | held_out | unchanged |
| driving.takeover_modality / driving.takeover_hands_auditory | 1.61969 | 1.61969 | 0 | s | held_out | unchanged |
| driving.takeover_modality / driving.takeover_modality_order | 1 | 1 | 0 | fraction | held_out | unchanged |
| driving.takeover_motor_readiness / driving.zeeb_hands | 1.02875 | 1.02875 | 0 | s | held_out | unchanged |
| driving.takeover_motor_readiness / driving.zeeb_road | 0.902545 | 0.902545 | 0 | s | held_out | unchanged |
| eyes.latency / eyes.latency_median | 203.01 | 203.01 | 0 | ms | consistency | unchanged |
| eyes.main_sequence / eyes.duration_intercept | 21 | 21 | 0 | ms | consistency | unchanged |
| eyes.main_sequence / eyes.duration_slope | 2.2 | 2.2 | 0 | ms/deg | consistency | unchanged |
| fatigue.endurance / fatigue.endurance_error | 0.159217 | 0.159217 | 0 | \|ln ratio\| | consistency | unchanged |
| memory.span / memory.span | 4 | 4 | 0 | items | consistency | unchanged |
| motor.touch_tapping / motor.touch_slope | 104.48 | 104.48 | 0 | ms/bit | consistency | unchanged |
| motor.touch_tapping / motor.touch_throughput | 6.28071 | 6.28071 | 0 | bits/s | held_out | unchanged |
| objects.peripheral_acuity / objects.e2 | 1.97551 | 1.97551 | 0 | deg | consistency | unchanged |
| response.drt / response.drt_tactile | 380.297 | 380.297 | 0 | ms | held_out | unchanged |
| response.drt / response.drt_visual | 382.925 | 382.925 | 0 | ms | held_out | unchanged |
| response.drt / response.older_slowing | 1.29835 | 1.29835 | 0 | ratio | held_out | unchanged |
| response.drt_load / response.drt_2back_cost | 297.026 | 297.026 | 0 | ms | held_out | unchanged |
| response.drt_load / response.drt_driving_cost | 28.3121 | 28.3121 | 0 | ms | held_out | unchanged |
| response.drt_load / response.drt_surt_cost | 367.348 | 367.348 | 0 | ms | held_out | unchanged |
| response.eccentricity / response.eccentricity_rmse | 1 | 1 | 0 | ms | held_out | unchanged |
| response.modality_rt / response.modality_order | 1 | 1 | 0 | fraction | consistency | unchanged |
| risk.lane_width / risk.lane_width_speed_error | 2.13347 | 2.13347 | 0 | km/h | held_out | unchanged |
| trust.kraus_dynamics / trust.malfunction_dip | 0.19739 | 0.19739 | 0 | scale points | held_out | unchanged |
| trust.kraus_dynamics / trust.study2_rmse | 0.290007 | 0.290007 | 0 | scale points | held_out | unchanged |
| vestibular.roll_compensation / vestibular.roll_compensation_effect | 0.491973 | 0.491973 | 0 | ratio | held_out | unchanged |
| vestibular.roll_compensation / vestibular.roll_frequency_order | 1 | 1 | 0 | fraction | held_out | unchanged |
| vestibular.sickness_time_course / vestibular.time_course_msi_error | 14.7523 | 14.7523 | 0 | % points | held_out | unchanged |
| vestibular.vertical_sickness / vestibular.vertical_msi_error | 7.46582 | 7.46582 | 0 | % points | calibration | unchanged |
| vision.foveation / vision.e2 | 2.3 | 2.3 | 0 | deg | consistency | unchanged |
