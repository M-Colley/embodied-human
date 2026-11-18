# Desired time headway transfer

The study changes one IDM parameter. Both profiles use the same CARLA 1.0 C3 vehicle, acceleration inverse, measured pedal feedback, 114 muscle body and 0.02 s observation history lag. The development phase uses 12 previously inspected US 101 tracks and one common simulation seed. The reference uses a desired time headway of 1.6 s. The candidates use 1.2 and 1.4 s. All other controller settings remain fixed. [Design](design.json) [Development inputs](development-inputs.json)

All 36 development episodes completed safely. The rule selected 1.2 s. Gap RMSE was 4.017 m for 1.6 s, 3.764 m for 1.4 s and 3.687 m for 1.2 s. Mean signed gap error was +0.526, −0.554 and −1.640 m, respectively. The selected parameter reduced RMSE while increasing the magnitude of the directional error in development. Commit `1be1718` published the selection before external evaluation. [Selection](development-selection.json) [All development episodes](development-episodes.zip)

The intended external comparison uses 24 tracks and two common seeds. Each test track is a complete 20 s following window. The primary score is gap RMSE. Means give equal weight to trials within tracks and tracks within acquisition runs. Acquisition runs also receive equal weight and define the paired bootstrap unit. The decision reports safe completion over every attempt. Accuracy comparisons use complete pairs and require at least 22 paired tracks and representation of every selected run. The candidate must complete every episode safely. The speed margin is 0.05 m/s and the absolute mean signed gap bias must improve or remain equal. These criteria were fixed before the external results.

The I 294 L1 source cannot meet the six run requirement with the 16 m/s ceiling, even before reconstructing leaders. I 294 L2 has too few comparable samples. The external coverage plan adds independently recorded I 90/I 94 moving captures using the same eligibility criteria. The six run adoption threshold remains binding. [Coverage plan](external-coverage-plan.json)

The final cohort contains 24 newly selected tracks. It includes four tracks from I 294 L1 run 8, four from each of four I 90/I 94 moving runs and four from I 90/I 94 stationary run 2. The first stationary run yielded zero eligible tracks under the fixed rules. The second run yielded 65. The cohort commit `a33fbd7` published the exact fixture, preparation code, run assignments, data hashes and runtime before either profile ran. The runs come from three recording dates. Repeated runs on the same road and date may share traffic conditions. The measurements represent observed vehicle tracks. The data do not link tracks to persistent driver identities. [Final inputs](test-inputs.json) [External cohort manifest](tgsim/source-manifest.json) [Acquisition and preparation](tgsim/README.md)

The U.S. Department of Transportation and Federal Highway Administration provide the TGSIM source data. The published fixture uses a straight longitudinal projection and an inferred leader reconstructed from tracked lane positions. The method uses the provider lane geometry and median tracked leader length. The published provenance records the eligibility filters and source licence. [I 294 L1](https://doi.org/10.21949/1404217) [I 90/I 94 moving](https://doi.org/10.21949/1404212) [I 90/I 94 stationary](https://doi.org/10.21949/1404218)

## External comparison

The 1.2 s headway failed the external transfer rule. The experiment recorded all 96 attempts. The 1.6 s reference completed 45 of 48 episodes, and the 1.2 s candidate completed 43. Road departure accounted for all eight incomplete episodes. Twenty one tracks supplied both seeds for both profiles. Each of the six runs contributed complete pairs. The candidate failed the complete episode and minimum paired track requirements.

Mean gap RMSE increased from 3.299 to 3.754 m on complete pairs. The mean paired change was +0.455 m (95% acquisition run percentile bootstrap interval [−0.355, 1.110], 5,000 resamples). Mean signed gap error shifted from −0.932 to −2.308 m, where negative values indicate simulated gaps shorter than the recorded gaps. Mean speed RMSE increased from 0.636 to 0.657 m/s and remained inside the 0.05 m/s margin. The conditional gap errors describe the 21 complete paired tracks. Safety outcomes include every attempt. [Test report](test-report.json) [All episodes](test-episodes.zip) [Archive checksums](test-archive.json)

MyoSuite 3.1.0 and development revision `c93329ec64bc8b544d0b3427bba6766377c3528e` each passed 576 tests with 25 skips. The physical tests imported `ModelBuilder`. Both environments used MuJoCo 3.17.0 and myo-sim 0.2.5. Ruff and mypy passed. [Release tests](checks/release-tests.txt) [Development tests](checks/dev-tests.txt)

## Reproduction

Use the bridge source that supplies the reported observation history lag and the same MyoSuite runtime declared in the frozen inputs. The development executable is preserved at commit `2fd8817`. Set `PYTHONPATH` to the corresponding embodied human and bridge source folders before running the example. The work used a detached runtime source checkout for both study phases.

```shell
python examples/evaluate_headway_transfer.py --stage development --design evidence/headway-transfer-2026-11-18/design.json --prepare path/to/us101-study/trials.npz --out path/to/new-development
python examples/evaluate_headway_transfer.py --stage development --design evidence/headway-transfer-2026-11-18/design.json --out path/to/new-development
```

The external evaluation uses embodied human source commit `2fd8817` with `examples/evaluate_headway_transfer.py` from `b8a47f9` and bridge source commit `9bd4d73`. The body code and dependencies match the development run. Copy `test-inputs.json` and `test-trials.npz` to a new directory as `inputs.json` and `trials.npz`. The evaluator checks the fixture, runtime and vehicle before running.

```shell
python examples/evaluate_headway_transfer.py --stage test --design evidence/headway-transfer-2026-11-18/design.json --out path/to/new-external-test
```

The source fixture can be regenerated from the official records with the [data preparation record](tgsim/README.md). The published [amendment](external-input-amendment.json) binds the manifest and the development selection.
