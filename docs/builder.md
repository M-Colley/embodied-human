# Local module builder and experiments

Install the package and launch the builder:

```bash
pip install -e .
python -m embodied_human.builder
```

Open the printed address, normally <http://127.0.0.1:8765>. The interface uses Python's standard library server and bundled browser assets. Use `--port 8766` to select another local port. Keep the terminal open while using the interface. Press Ctrl+C to stop the server.

## Configure a human

The model library comes from the registered implementations. Drag a model onto its category, click its library button, or use the category selection menu. Each configuration contains one implementation per category.

Select a category to inspect its parameters and sources. Numbers, booleans and strings have dedicated editors. Structured parameters use JSON. Parameter changes receive a user override label. Reset individual values or restore the complete baseline configuration.

**Validate model** constructs every selected implementation with the supplied seed. Validation checks configuration structure, parameter types and model constructor constraints. Scientific validation requires the task benchmarks and appropriate human references.

**Export JSON** saves module choices, overrides, the seed and the configuration format version. **Import JSON** validates a saved file before loading it. Structured values, including fatigue joint rates, are restored to their dataclass types. Import selects implementations already registered in the running Python process. Configuration files contain data.

**Export Python** saves a script that constructs the human and prints its memory demonstration. Run the exported files with the package installed:

```bash
python human-model.py
python -m embodied_human.builder --run human-config.json
```

To load a saved configuration in an existing experiment:

```python
import json
from pathlib import Path

from embodied_human import build_human
from embodied_human.builder import config_from_dict

project = json.loads(Path("human-config.json").read_text())
human = build_human(config_from_dict(project["config"]), seed=project["seed"])
```

## Run experiments

The experiment panel lists the registered benchmarks. Select the benchmarks and the number of seed repetitions. Seeds begin at the random seed in the main toolbar. Every compared configuration receives the same seeds. Selecting baseline comparison adds the baseline configuration to the experiment.

The panel also includes two native model demonstrations. `memory.serial_recall` records serial position accuracy, intrusions and retrieval failures after an interfering list. Select `memory=pyactup` to exercise native cue retrieval. `response.choice` records choice accuracy, response time and timeouts across explicit drift and deadline conditions. Select `response=pyddm` to run it. Both demonstrations use assumed task settings and have unclassified validation roles. Independent human evaluation is pending.

Parameter sweeps replace one numeric parameter in the current configuration. Select its category and parameter, then enter up to 32 comma separated values. The worker validates every resulting configuration before running the experiment. Integer parameters require integer values.

**Start experiment** launches a separate Python worker. The interface remains available during execution. Each job shows its progress, elapsed time and current benchmark. **Cancel experiment** terminates that worker. The server permits two concurrent experiments. Each job has a configurable time limit from 1 to 3600 seconds.

Completed jobs show metric cards with units, validation roles and uncertainty intervals. Each chart compares configurations on a separate metric scale. Paired change cards report the configuration minus baseline for matching seeds. Intervals describe simulation variability across the supplied seeds. Undefined intervals retain that label. Complete estimate tables and the export plot are available in expandable sections. Matching the source used to set a parameter is a consistency check. Calibration results and held out evaluations have separate labels.

Native task accuracy and rate cards show percentages with one decimal place. Their charts span 0% to 100%. Paired changes use percentage points and retain their signed scale. Complete tables and downloads retain the underlying proportions and their precision.

Download the JSON report, CSV results, manifest, SVG plot or complete run bundle. The manifest records the software and experimental configuration required for replay. Worker files remain in a server owned temporary directory. Server shutdown removes that directory. Download the run bundle before stopping the server.

Reloading the page restores retained jobs from the running server. Completed jobs keep their downloads and result tables. Failed jobs include bounded worker diagnostics. A completed status requires all advertised output files.

The complete bundle includes `package-source.zip`, the reference data, the environment record and replay instructions. Replay checks artifact hashes, the package source hash and required dependency versions. Extract the bundle and use its recorded source and dependencies:

```bash
python -m embodied_human.experiments --replay manifest.json --out replayed
```

`--allow-source-mismatch` and `--allow-environment-mismatch` permit explicit comparison runs with changed software. The new manifest records the software used for that run.

## Analyse parameter sensitivity

Install the optional analysis dependencies specified in the package metadata. Choose Morris sensitivity screening or Sobol sensitivity indices in the experiment panel. Select a benchmark, its output metric and bounds for the continuous parameters. Each design point uses the supplied seed repetitions.

Morris design size specifies the number of trajectories. Sobol design size must be a power of two. Both methods require at least four. The interface accepts up to eight continuous parameters. Parameter bounds require a scientific justification because they determine the interpretation of the indices.

Sensitivity results use SALib's sampling and analysis implementations. The run bundle records the method, bounds, design settings and output metric. Sensitivity analysis runs as a separate experiment with the current configuration.

Sensitivity charts show Morris absolute effects or Sobol first order and total order indices. Whiskers show bootstrap confidence intervals. The expandable table reports all estimates, confidence half widths and parameter bounds. Undefined estimates retain that status and display the recorded reason. The analysis assumes independent uniform input distributions within the declared bounds.

## Memory demonstration

**Run seeded preview** executes delayed digit recall with 24 digits at each level from 0 to 5. Each level begins with fresh memory state and the same seeded streams. Accuracy cards show the six levels with memory retrieval times and response counts. Exact measurements remain available in an expandable table. Memory determines these task results. Human validation results are available through the benchmark runner.

## Local execution

The server binds to `127.0.0.1`. It serves bundled assets and explicit API routes. Requests from foreign origins and hostnames are rejected. JSON requests are limited to 256 KiB. Worker commands use fixed Python module arguments. The browser cannot supply executable code or arbitrary filesystem paths. Optional implementations require their documented dependencies.
