# TGSIM source preparation

This directory contains the exact acquisition and preparation code, provider metadata snapshots, prior feasibility manifests and the selected cohort. The [source manifest](source-manifest.json) records source and fixture checksums, lane reconstruction, eligibility and selected track identifiers. The [execution record](reproduction-record.json) records the original commands and environment.

Use a fresh copy of this directory. Install the bridge at revision `9bd4d73` in a Python 3.12 environment. Install the recorded preparation dependencies. Run these commands from the copied directory. The acquisition scripts fetch the official raw data and author supplied geometries. The preparation script then writes `prepared-final/selected-trials.npz` and its manifest.

```shell
python -m pip install -r requirements.txt
python acquire.py
python acquire_dataset.py --dataset 6a6e-vfvi --metadata alternate-probes/i90-moving-metadata.json --counts alternate-probes/i90-moving-run_counts.json --out i90-moving
python acquire_dataset.py --dataset 9uas-hf8b --metadata alternate-probes/i90-stationary-metadata.json --counts alternate-probes/i90-stationary-next-run_counts.json --out i90-stationary-next
python prepare.py
```

Compare the selected group identifiers, run assignments and fixture checksum with the archived manifest. Source manifests include download times and paths. Those fields vary with each acquisition. Raw cloud data and geometry responses can be checked against the archived source hashes.
