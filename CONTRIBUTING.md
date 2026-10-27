# Contributing a module, task or benchmark

Start with [AGENTS.md](AGENTS.md), which defines the package rules, and the [module tutorial](examples/tutorial_modules.ipynb). A contribution should state the human behaviour it represents, the interface it implements, its units and timing, its evidence and its limitations. Preserve the current baseline unless a change to its scientific assumptions is intentional and recorded.

## Set up a development environment

Run these commands from the repository root in an activated Python virtual environment:

```sh
python -m pip install -e ".[test,lint,experience]"
python -m embodied_human.bench --list
```

Some tasks need additional packages. Follow the [installation instructions](README.md#install) for embodied-ui and myosuite-carla. Install only the native backends required by your contribution, for example `python -m pip install -e ".[memory,decision,analysis,contact]"`. TouchSim uses the separately pinned `requirements-touchsim.txt`. The package must still import and construct its default human without optional backends installed.

## Add a module with an explicit contract

A human selects one implementation per category through `HumanConfig`. Subclass the category interface, register a unique name and keep defaults in a frozen `Parameters` dataclass. Each field needs a `SOURCES` entry with a verified source and one of `measured`, `standard`, `fitted`, `derived` or `assumed`. A numerical solver setting, task choice or unverified value is assumed until evidence establishes a different status. State units, fitted conditions and parameter dependencies in each source note.

The following complete example implements the keyed `Memory` interface. Save it as `my_memory.py` in the repository root while developing. It is a finite buffer for demonstrating the extension interface. Its capacity and latency are assumed, and it has no decay, rehearsal, intrusion or human calibration.

```python
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from embodied_human.core import Parameters, Source, register
from embodied_human.memory import Memory, Retrieval


@dataclass(frozen=True)
class BufferParameters(Parameters):
    capacity: int = 4
    latency: float = 0.0

    SOURCES: ClassVar = {
        "capacity": Source(
            "Contributor interface example", None, "assumed",
            "items retained; illustrative setting without a human fit",
        ),
        "latency": Source(
            "Contributor interface example", None, "assumed",
            "seconds for either retrieval outcome; illustrative setting",
        ),
    }

    def __post_init__(self) -> None:
        if type(self.capacity) is not int or self.capacity < 0:
            raise ValueError("capacity must be a nonnegative integer")
        if not math.isfinite(self.latency) or self.latency < 0:
            raise ValueError("latency must be finite and nonnegative")


@register("memory", "example_buffer")
class ExampleBuffer(Memory):
    def __init__(self, **params: Any) -> None:
        super().__init__()
        self.params = BufferParameters(**params)
        self.reset()

    def reset(self, rng: np.random.Generator | None = None) -> None:
        super().reset(rng)
        self._items: OrderedDict[str, Any] = OrderedDict()
        self._time = -math.inf

    def _advance(self, time: float) -> None:
        if not math.isfinite(time) or time < self._time:
            raise ValueError("time must be finite and nondecreasing")
        self._time = time

    def encode(self, key: str, value: Any, time: float) -> None:
        self._advance(time)
        self._items[key] = value
        self._items.move_to_end(key)
        while len(self._items) > self.params.capacity:
            self._items.popitem(last=False)

    def retrieve(self, key: str, time: float) -> Retrieval:
        self._advance(time)
        success = key in self._items
        return Retrieval(
            self._items.get(key), success, self.params.latency,
            math.inf if success else -math.inf,
        )
```

The infinite activation values indicate availability in this example. They are not fitted activation estimates. `reset` clears state for the next run and retains the supplied NumPy Generator. Use `self.rng` for stochastic behaviour and pass local generators or explicit seeds to native libraries. Avoid reseeding process global random generators, shared mutable state and hidden randomness during construction. `build_human(config, seed=...)` assigns separate module streams; `human.reset(seed=...)` restores a reproducible initial run. A module reset without a supplied generator continues its current stream. A human reset needs an explicit seed for reproducibility because omitting it creates fresh streams.

Implement all methods required by the category and its callers. Preserve their return types, time conventions and units. Validate domains and handle empty input, invalid time, nonfinite values and boundary conditions explicitly. Each new capability, including cue retrieval or choice distributions, needs a dedicated task that calls it.

## Test the behaviour and the selected implementation

Save this as `test_my_memory.py` beside the example module:

```python
import pytest

import my_memory  # Registers example_buffer.
from embodied_human import HumanConfig, build_human


@pytest.fixture
def human():
    config = HumanConfig().replace("memory", "example_buffer", capacity=2)
    return build_human(config, seed=7)


def test_capacity_and_payload(human):
    memory = human.memory
    for index, key in enumerate(("a", "b", "c")):
        memory.encode(key, index, float(index))
    assert not memory.retrieve("a", 3.0).success
    assert memory.retrieve("b", 3.0).value == 1
    assert memory.retrieve("c", 3.0).value == 2


def test_reset_clears_state(human):
    human.memory.encode("a", 42, 0.0)
    human.reset(seed=7)
    assert not human.memory.retrieve("a", 0.0).success


def test_people_have_separate_state(human):
    other = build_human(human.config, seed=7)
    human.memory.encode("a", 42, 0.0)
    assert not other.memory.retrieve("a", 0.0).success


def test_invalid_parameters():
    with pytest.raises(ValueError):
        my_memory.ExampleBuffer(capacity=-1)
    with pytest.raises(ValueError):
        my_memory.ExampleBuffer(latency=float("nan"))
```

Run the test and the existing memory benchmark through the external module import:

```sh
python -m pytest test_my_memory.py -q
python -m embodied_human.bench --import my_memory --replace memory=example_buffer --only memory.span --seed 0 --out results/example-buffer.json
```

The default example returns a span of four because it retains four items. Human evaluation of this interface example remains pending. The tests check eviction, actual payloads, reset, state independence and domain validation. Check repetition with identical seeds and changes across distinct seeds for a stochastic implementation, including randomness in native libraries. Use independent predictions or analytical consequences as test expectations. Verify that a replacement is actually called and that parameter overrides change the relevant outcome.

For a packaged contribution, move the module into `src/embodied_human/`, use package relative imports, import it in `core._load_builtins()` and place its tests under `tests/`. Update the implementation documentation and include a selected task or benchmark command in the contribution record. `--import` loads local Python code for development. Bundling runs with `--import` is rejected because those external files are absent from the package source snapshot. The visual builder and experiment worker use packaged registrations and catalogued protocols; a local import does not install a new worker protocol.

A new category changes the public composition contract. Add its interface, registry category, default `HumanConfig` field and builtin import together. Verify configuration export and restoration, parameter provenance, population construction, human reset and builder discovery. Give the category a dedicated task and declare its benchmark coverage. Update documentation and tests that enumerate the supported categories.

## Add a task or a published benchmark

First reuse an existing benchmark if it already measures the intended behaviour. A new task belongs in `tasks/` and should expose explicit conditions, durations, units and a seed or local generator. Report failures and timeouts with clear denominators. Keep missing measurements undefined, which reports encode as JSON null, and distinguish unconditional accuracy from accuracy among completed responses. Integrated driving benchmarks must use the seeded, offline myosuite-carla vehicle surrogate.

A published benchmark is a callable taking `Human` and returning `Result`, registered with `@benchmark("category.unique_name")`. Measure the selected modules from that human or build derived cases from `human.config`. Nested trials should use `protocol_seed(index)` or `protocol_seeds(count)` from `bench.protocol`. The runner supplies the outer seed context and resets the human for each benchmark. Never silently construct a default implementation inside a replacement benchmark.

Use `result.score_against(reference_name, observed_value)` for a published reference in `bench/references.py`. That file holds aggregate published values only. A new `Reference` needs a verified primary source, exact units, task and sample conditions, central value and an acceptance range with a stated justification. Do not make up a citation, DOI or human acceptance range to obtain a passing score. Store protocol settings, their provenance, trial counts, intermediate observations and known mismatches in `result.details`. Assign `result.skipped` a useful reason if an optional dependency or required interface is unavailable. Unexpected programming errors should still fail visibly.

Declare the metric's evaluation role and data identifiers in `bench/validation.py`. A source used to set a parameter makes the corresponding comparison a consistency check. Outcomes used for fitting are calibration results. Held out results require an explicit account of which outcomes, conditions or participants were reserved and what was used for fitting. A held out statistic from the same participants must say so. Keep unresolved independence unclassified. A change to implementation or fitting data can invalidate an existing role, so review role declarations when adding an alternative model.

The global declarations describe baseline data use. An alternative implementation with a different fitting history can override a metric's `result.validation` entry using `Evaluation(...).as_dict()`. Record its evaluation and fitting data identifiers explicitly. Preserve the baseline declaration for the original implementation. The comparison report identifies declaration changes for scientific review.

Add the benchmark's categories, metric names and `ci_profile` to `bench/protocols.json`. The `core` profile contains protocols suitable for routine CI; `full` adds protocols with additional dependencies or runtime requirements. Registry and catalog tests catch omissions. Import a new benchmark module in the runner's benchmark import path, then check `python -m embodied_human.bench --list`. Add tests for the measurement, skip reason and selected implementation. Include a regression fixture when a concrete numerical defect motivated the change.

If human evaluation data are unavailable, add an unscored model demonstration to `experiment_protocols.py`. Add its callable to `PROTOCOLS` and a corresponding `CATALOG` entry with `name`, `description`, `categories` and `metrics`. The callable still returns `Result`, but places task outcomes in `metrics` without inventing reference scores. Put fixed task settings in a `Parameters` dataclass with honest `SOURCES`, and retain them in `details`. State that independent human evaluation is pending. These demonstrations remain separate from the published benchmark suite. The [native model integrations](docs/model-integrations.md) and `tests/test_experiment_protocols.py` provide worked examples.

## Run automated benchmark comparisons

Capture a baseline before changing the implementation. Use the same profile and seed for the comparison:

```sh
python -m embodied_human.bench --profile core --seed 0 --out results/before.json
python -m embodied_human.bench --profile core --seed 0 --compare results/before.json --summary results/comparison.md --out results/after.json --fail-on-regression --fail-on-skip
```

The second command writes its JSON report and Markdown summary before applying exit gates. `--fail-on-regression` requires `--compare`. It rejects new failures, worsening distance outside a reference range, missing measurements, new skips, finite measurements becoming null and incomparable references, units or evaluation roles. An existing failure that has not worsened is reported separately. New benchmark additions are reported for review and do not automatically fail the comparison. Review additions explicitly before accepting a new baseline.

Comparison requires matching seeds. Different module configurations are retained in the report so intentional replacements can be compared. `--only` selects named protocols and allows a larger baseline to be filtered to that selection. `--profile` and `--only` are mutually exclusive. The regression distance is measured outside the acceptance range in units of its width and remains informative after a displayed score has reached zero. `--regression-tolerance` controls the permitted change in that normalized distance and defaults to `1e-9`; a looser tolerance needs a numerical justification. This tolerance represents a computational regression policy, not human measurement uncertainty.

`--fail-on-skip` treats every skipped selected protocol as a failure, including an existing skip. Install the dependencies expected by the selected profile. Add tests and an explicit selected run for a new native backend or extra API. Use `--profile full` for the full published suite when its optional task and driving dependencies are available. The [benchmark workflow](.github/workflows/benchmarks.yml) runs 21 core protocols on pull requests and main pushes. It compares the base and proposed revisions using the same installed dependencies and seed. The [test workflow](.github/workflows/tests.yml) runs the unit, integration, lint, type and build checks.

For the complete offline evaluation, open the repository's Actions tab, select `benchmarks`, choose Run workflow and set `suite` to `full`. The optional `baseline_ref` input selects a comparison revision. Its default is the first parent of the selected commit. The full job installs pinned physical dependencies and runs all 28 published protocols. Reports, baseline inputs, source identities, logs and the Markdown summary remain available as workflow artifacts for 14 days. Manual workflow controls are documented in [GitHub's workflow guide](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow).

Comparison bundles archive `comparison-baseline.json` and `summary.md` with checksums. The JSON comparison also identifies the canonical baseline content hash. To reproduce the analysis after replay, provide the retained baseline with `--compare comparison-baseline.json`. The source and environment comparison rules still apply. Exit code 0 indicates that the requested gates passed. Exit code 1 indicates a regression, incomparable evaluation metadata or a selected skip rejected by a gate. Exit code 2 indicates invalid arguments or an invalid comparison, including unequal seeds. Requested outputs retain the diagnostic reason.

A regression gate compares one recorded seeded run. For scientific comparisons, repeat matching seeds with the [experiment runner](docs/experiments.md), inspect paired effects, retain outcomes for each seed and report simulation uncertainty. Independent human validation requires reserved evaluation data and an appropriate task mapping. Retain the scientific or numerical reason for each change to reference ranges or baseline evidence.

## Add a native backend responsibly

Import optional packages lazily, at the method that needs them. Keep the base package and default constructors usable without extras, and provide an actionable error when a dependency is missing. Bound versions where the adapter depends on specific native behaviour. Separate adapter assumptions, solver settings and native defaults from fitted human parameters. Test the real installed backend against published equations or verified data, record the native version, and check probability mass, timing, units and seed control where applicable. Preserve distinct interfaces when a scientific mapping between them has not been established.

Review the licence of the exact upstream revision, its bundled assets and its dependencies. Keep required copyright and licence notices with reused code and in built distributions. Record the upstream revision and any local modifications. Do not copy GPL code from User-in-the-Box, pyactr or arm_force_driving into this package. An implementation from published equations still needs a verified equation source and honest provenance. Update optional dependency declarations, integration documentation and relevant package data or source distribution entries. Build the distributions and inspect their required notices and assets.

## Required checks and calibration

Run the repository checks after a change:

```sh
python -m pytest tests -q
python -m ruff check src tests
python -m mypy src/embodied_human --ignore-missing-imports
python -m build
```

Run tests for optional integrations with their extras installed as well. Report which tests were skipped and why. After changing defaults, regenerate the parameter table and the baseline evidence:

```sh
python examples/parameter_table.py
python -m embodied_human.bench --out evidence/benchmark-baseline.json
```

After changing the attention, estimator or eye modules, refit the glance sampler and report fitted and held out results separately:

```sh
python examples/calibrate_glances.py --out evidence/glance-calibration.json
```

After changing the vestibular or response modules, rerun both fitting scripts and update the fitted constants they report:

```sh
python examples/fit_motion_sickness.py
python examples/fit_drt_load.py
```

These triggers come from [AGENTS.md](AGENTS.md). Record each triggered fit and its reported constants, including unchanged results.

## Preserve evidence for review and reuse

Record commands, Python and dependency versions, seeds, selected configurations, skip reasons and outcomes in [VALIDATION.md](VALIDATION.md). Separate fitting results from held out evaluation and distinguish simulation variation from variation between human participants. Retain the source of any human aggregate, its conditions, permitted use and the rule used to form its acceptance range. Human participant data need an appropriate consent, privacy and sharing plan; published aggregate references should not be presented as individual participant records.

Use `--bundle results/run-name` for a packaged benchmark run, or retain an experiment bundle. Bundles record configurations, parameter sources and overrides, references, evaluation roles, seeds, environment metadata, available traces and a source archive. Replay checks those records before execution. [The replay guide](docs/experiments.md#run-bundles) describes the explicit source and environment comparison overrides and the limits of dependency fingerprints. Archive external dependency sources and required assets separately. A source hash or version record does not preserve those external files.

A review should be able to answer four questions: does the implementation follow its stated interface and equations; does each scientific claim match the recorded data use and conditions; can another person rerun the selected behaviour with the recorded software and seeds; and is the contribution maintainable without weakening optional imports, tests or licence obligations? Include a small runnable example, an explicit selected test or protocol and the remaining validation limits in the change description. Keep commit authorship with the human owner and omit AI coauthor trailers.
