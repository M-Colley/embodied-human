# Contributing a module or a benchmark

The project is useful only if changes are comparable. Please follow these
rules.

## A new module

1. **Pick a category** (`eyes`, `vision`, `objects`, `attention`, `memory`,
   `proprioception`, `motor` or `estimator`). Subclass its interface, for
   example `embodied_human.memory.Memory`, and register the class with
   `@register(category, "your_name")`. A module keeps all randomness in
   `self.rng` and fully resets in `reset(rng)`.
2. **Give every parameter a source.** Put the defaults in a frozen dataclass
   deriving from `Parameters`, with a `SOURCES` entry per field. Use these
   kinds:
   - `measured` in humans, with conditions;
   - set by a `standard`;
   - `fitted` by a published model;
   - `derived` from other values;
   - `assumed`.

   Never invent a citation. An assumed value is fine if it is labelled.
3. **Run the benchmark with only your module swapped:**

   ```bash
   python -m embodied_human.bench --replace memory=your_name --out yours.json
   python -m embodied_human.bench --out baseline.json
   ```

   Report both files, the seed and the package versions. Module benchmarks
   whose reference matches your parameter source are consistency checks.
   The integrated benchmarks (glances, take-over, touch throughput) show
   whether the change helps.
4. **Keep the interface.** If your module needs more inputs, propose an
   interface change separately so every module of the category can adopt it.

## A new benchmark

1. **Store only aggregate published values** in
   `bench/references.py`: mean, SD or range, with the citation, DOI,
   conditions and an acceptance range. Never store participant-level data.
2. **Decorate a function** `(human) -> Result` with `@benchmark("name")`, and
   use `result.score_against(reference, value)` for scored metrics and
   `result.details` for everything else. Skip with `result.skipped` when an
   optional package is missing.
3. **Make it reproducible and describe its scope.** Seed every random stream
   and use the vehicle surrogate rather than a live simulator by default.
   State in the docstring what the benchmark can and cannot show.

## Code

- **Tests:** `python -m pytest tests -q`.
- **Lint and types:** `python -m ruff check src tests` and
  `python -m mypy src/embodied_human --ignore-missing-imports`.
- **Licences:** the package is Apache-2.0. Do not copy code from GPL
  projects (for example User-in-the-Box or pyactr); reimplement from the
  published equations and cite them.
