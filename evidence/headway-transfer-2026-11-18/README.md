# Desired time headway transfer

The study changes one IDM parameter. Both profiles use the same CARLA 1.0 C3 vehicle, acceleration inverse, measured pedal feedback, 114 muscle body and 0.02 s observation history lag. The development phase uses 12 previously inspected US 101 tracks and one common simulation seed. The reference uses a desired time headway of 1.6 s. The candidates use 1.2 and 1.4 s. All other controller settings remain fixed. [Design](design.json) [Development inputs](development-inputs.json)

All 36 development episodes completed safely. The rule selected 1.2 s. Gap RMSE was 4.017 m for 1.6 s, 3.764 m for 1.4 s and 3.687 m for 1.2 s. Mean signed gap error was +0.526, −0.554 and −1.640 m, respectively. The selected parameter reduced RMSE while increasing the magnitude of the directional error in development. Commit `1be1718` published the selection before external evaluation. [Selection](development-selection.json) [All development episodes](development-episodes.zip)

The intended external comparison uses 24 tracks and two common seeds. Each test track is a complete 20 s following window. The primary score is gap RMSE. Means give equal weight to trials within tracks and tracks within acquisition runs. Acquisition runs also receive equal weight and define the paired bootstrap unit. The decision reports safe completion over every attempt. Accuracy comparisons use complete pairs and require at least 22 paired tracks and representation of every selected run. The candidate must complete every episode safely. The speed margin is 0.05 m/s and the absolute mean signed gap bias must improve or remain equal. These criteria were fixed before the external results.

The I 294 L1 source cannot meet the six run requirement with the 16 m/s ceiling, even before reconstructing leaders. I 294 L2 has too few comparable samples. The external coverage plan adds independently recorded I 90/I 94 moving captures using the same eligibility criteria. The six run adoption threshold remains binding. [Coverage plan](external-coverage-plan.json)

## Reproduction

Use the bridge source that supplies the reported observation history lag and the same MyoSuite runtime declared in the frozen inputs. The development executable is preserved at commit `2fd8817`. Set `PYTHONPATH` to the corresponding embodied human and bridge source folders before running the example. The work used a detached runtime source checkout for both study phases.

```shell
python examples/evaluate_headway_transfer.py --stage development --design evidence/headway-transfer-2026-11-18/design.json --prepare path/to/us101-study/trials.npz --out path/to/new-development
python examples/evaluate_headway_transfer.py --stage development --design evidence/headway-transfer-2026-11-18/design.json --out path/to/new-development
```
