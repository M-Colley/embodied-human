# External site following comparison

The comparison evaluates transfer of the acceleration inverse to a reserved road site. The earlier I 80 experiment selected this control interface on development data. The present US 101 study uses 24 original passenger vehicle tracks and two common simulation seeds per track. Each replay follows the recorded leader for 20 s. Both profiles use the same MyoSuite body, CARLA 1.0 Nissan Patrol parameters and IDM settings. The inverse profile also tracks measured pedal positions. Both profiles use a 0.02 s control step and a 0.02 s observation history lag. [Design](design.json) [Complete configuration and source hashes](frozen-inputs.json)

The FHWA portal returned 4,679,934 passenger vehicle rows from three collection periods. The loader retained one record per vehicle and timestamp, giving 3,990,400 distinct keys. None of the duplicate groups disagreed on longitudinal position, lane, preceding vehicle or vehicle length. All 3,062,862 passenger vehicle keys in the original archive CSV files were present in the portal export. Each archive CSV reached the spreadsheet row limit. The recorded fixture retains the 24 selected windows in SI units. [Source and attribution](study-source.json) [Independence audit](independence-audit.json) [Duplicate audit](duplicate-audit.json) [Trial fixture](trials.npz)

The primary score averages each trajectory RMSE across the two common seeds. It squares the normalized speed, gap and acceleration errors and gives them equal weight. Original vehicle tracks receive equal weight. The 95% interval uses a paired bootstrap across these tracks within the observed site. Acceptance requires all 96 episodes to complete safely, the interval upper bound below zero and the mean of each error metric to improve or remain equal. The data support evaluation of following kinematics at this site. The measurements do not include attention or pedal input.

The source fixture is derived from the FHWA Next Generation Simulation Vehicle Trajectories and Supporting Data under CC BY SA 3.0. The [data portal](https://data.transportation.gov/d/8ect-6jqj) and [dataset DOI](https://doi.org/10.21949/1504477) provide the raw data. The source manifest records the exact column projection, query, download pages and checksums. Before any physical test episode, commit `da9a2e8` published the frozen fixture and all profile settings.

The preparation command uses the portal CSV and its provenance manifest. A reproducible test uses a fresh output directory containing the exact `trials.npz` and `frozen-inputs.json`. The script checks the fixture, vehicle parameters and runtime before execution.

```shell
python examples/evaluate_carla1_following.py --stage prepare --design evidence/carla-1-following-us101-2026-11-18/design.json --source ngsim-us101-cars.csv --manifest study-source.json --out site-test
python examples/evaluate_carla1_following.py --stage test --design evidence/carla-1-following-us101-2026-11-18/design.json --out site-test
```

## Results

The inverse completed all 48 episodes. The baseline completed 47 of 48. It left the road at 19.48 s in source `ngsim-us101:1761:1` with base seed 702. There were no collisions. The incomplete baseline prevented acceptance under the fixed rule. The following means describe the 23 tracks that completed both common seeds in both profiles. The conditional comparison had no inferential test or role in adoption. [Decision and metrics](report.json) [All physical episodes](physical-episodes.zip) [Archive checksums](episode-archive.json)

| Error metric | Baseline | Acceleration inverse |
| --- | ---: | ---: |
| Speed RMSE (m/s) | 0.8884 | 0.8874 |
| Bumper gap RMSE (m) | 3.8408 | 4.1162 |
| Acceleration RMSE (m/s²) | 0.7884 | 0.6948 |

The inverse reduced acceleration error by 11.9% on the complete tracks. Bumper gap error increased by 7.2%. These observations pertain to the tested US 101 windows and the fixed simulated vehicle and body. The following controller receives simulated traffic state directly. The measurements establish errors in following kinematics under this assumption.
