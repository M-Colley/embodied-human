# Experiments and reproducible evaluation

The experiment runner compares registered modules, varies parameters and estimates sensitivity. Each configuration receives the same requested seeds. Nested trials use independent SeedSequence derivation for each outer seed. The historical nested sequence is retained for outer seed zero. Confidence intervals describe simulation variation across seeds. Participant uncertainty requires an explicit participant model and human data.

Install the optional backends and start the local interface:

```sh
python -m pip install -e ".[memory,decision,analysis,contact]"
python -m embodied_human.builder
```

Select the benchmarks, seed repetitions and execution limit in the experiment panel. The comparison control adds the baseline configuration. Parameter sweeps evaluate each supplied value. Morris and Sobol analyses accept bounds for continuous parameters. Bounds are experimental assumptions and require a scientific justification. The worker runs in a separate process. Cancellation and time limits terminate that owned process. The server retains results until it closes. Download the run bundle to preserve them.

## Validation roles

Every published reference has a declared evaluation role and data identifiers in `bench/validation.py`. `consistency` identifies comparisons with sources used to set the model. `calibration` identifies outcomes used for fitting. `held_out` identifies reserved outcomes or conditions. A held out statistic can share participants with the fitting data. `unclassified` records unresolved data independence. The emotion rating comparison currently has this label. The modality ordering metric is a consistency check because its tactile latency uses the same study's modality difference.

Reports include evaluation conditions, fitting identifiers and the rationale for each role. Scored outcomes are counted separately for each role. Experiment totals count metric evaluations across cases and seeds. The unique reference metric identifiers are listed separately. Memory and choice demonstrations have unscored task outcomes and carry the unclassified label. Their task settings are documented assumptions. Independent human evaluation remains pending.

## Command line execution

Save a request as JSON. This example compares the baseline with PyACTUp on serial recall. All outcomes and traces come from the selected implementations.

```json
{
  "version": 1,
  "config": {"memory": {"name": "pyactup", "params": {"capacity": 0, "threshold": -5.0}}},
  "benchmarks": ["memory.serial_recall"],
  "seeds": [0, 1, 2, 3],
  "compare_baseline": true,
  "sweep": null,
  "sensitivity": null
}
```

```sh
python -m embodied_human.experiments --request recall.json --out results/recall
python -m embodied_human.experiments --replay results/recall/manifest.json --out results/recall-replay
python -m embodied_human.bench --only eyes.latency memory.span --bundle results/baseline
python -m embodied_human.bench --replay results/baseline/manifest.json --out results/replayed.json
```

Requests support at most 20 distinct seeds, 32 sweep values and 2048 protocol evaluations. Repeated identical seed identifiers are rejected. A sweep contains `category`, `parameter` and `values`. Sensitivity analyses contain a method, factors, bounds, design size and target metric. [Sensitivity documentation](sensitivity.md) describes the SALib analysis and its assumptions. The worker rejects nonfinite target values at any design point. Removing failed points would invalidate the design.

The [included requests](../examples/experiments/) cover native serial recall, choice bound sweeps, DRT Morris screening and eye parameter Sobol analysis. The [committed evidence](../evidence/experiments/) stores all result rows and sensitivity design points. Base parameter records and each row's overrides reconstruct the case configurations. The corresponding environment is recorded separately. Run a request with `python -m embodied_human.experiments --request examples/experiments/choice.json --out results/choice` to create its complete bundle.

## Run bundles

Each bundle contains `report.json`, `manifest.json`, `references.json`, `environment.json`, `package-source.zip` and `REPLAY.txt`. Experiment bundles also include their request, CSV results and an SVG plot. Resolved configurations contain parameter defaults and user overrides. Provenance labels describe the default source. Overrides are marked explicitly. Available protocol traces and intermediate measurements remain in each result's details.

The manifest records relevant dependency versions, Python, Git revision, dirty state, source hashes and artifact checksums. The environment also lists installed distribution versions for resolving transitive dependencies. The source archive preserves this package's files and licence notices from the recorded run. Checksums detect artifact changes. Replay checks the package source, Python runtime, every recorded relevant dependency version and the recorded source identities of known dependencies. The comparison flags `--allow-source-mismatch` and `--allow-environment-mismatch` permit deliberate comparisons with changed software. Numerical agreement across hardware still requires verification.

Editable dependency locations are discovered from their installed `direct_url.json` metadata and are not written as absolute machine paths. A Git checkout is identified by its revision, the content hash of its current diff and hashes of recognised untracked source files. Clean tracked code and assets are identified by the Git revision, so a clean MyoSim mesh tree is not read and hashed in full. A source copy without usable Git metadata is identified by hashes of recognised source and asset files, excluding caches, virtual environments, build output and generated artifact directories. Changed code, changed revisions and changed untracked source therefore require `--allow-environment-mismatch` even when the dependency version remains unchanged. Unreadable or missing editable source cannot be certified for exact replay. Older bundles without these records require the same comparison override when an editable dependency is present.

Known dependencies installed from a VCS origin also record the VCS type, installed commit identifier and requested revision from `direct_url.json`, without storing the repository URL. This distinguishes Git installations such as TouchSim that can share a package version across different commits. Changed origin metadata requires the environment comparison override. This record describes the installation origin; it is not a content snapshot of the installed distribution.

The package source archive does not contain external dependency checkouts or their assets. Their source identities support comparison, but their archives, exact revisions and required assets must be retained separately. A dependency omitted from the known scientific dependency list has its installed version listed without a source fingerprint. These records do not turn a run bundle into a complete environment archive.

The CLI `--import` option supports local benchmark extensions. Creating a bundle with this option is rejected because external extension source is absent from the package snapshot. Package an extension and document its provenance before adding it to a reproducible worker protocol.

## Interpretation

Seed comparisons measure computational variation under the chosen protocol. The reported BCa intervals use SciPy and require at least three finite observations with variation. Undefined intervals remain null in JSON and are displayed as undefined. Paired differences subtract the baseline value for the same seed. Parameter sensitivity is conditional on the supplied bounds and seeds. Sensitivity identifies influential assumptions. It establishes neither a human parameter distribution nor empirical model validity.
