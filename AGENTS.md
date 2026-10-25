# Package instructions

This is Mark Colley's standalone `embodied-human` package: a modular
simulated human with cited baselines, tasks and a benchmark. It uses
embodied-ui (devices, engines) and myosuite-carla (the MyoSuite full-body
driver) as optional dependencies; keep it importable without them.

- **Modules.** One module per category, registered by name. A module's
  defaults live in a `Parameters` dataclass whose `SOURCES` label each value
  as measured, standard, fitted, derived or assumed. Never invent a citation
  or a DOI; label unverified values as assumed.
- **References.** `bench/references.py` holds only aggregate published values
  with their source, conditions and acceptance range.
- **Honest interpretation.** Benchmarks that compare a module with the source
  of its own parameter are consistency checks; say so when reporting.
  Integrated benchmarks must run offline on myosuite-carla's vehicle
  surrogate and be seeded.
- **Licences.** Do not copy GPL code (User-in-the-Box, pyactr, arm_force_driving);
  reimplement from published equations.
- **Checks.**
  - Test: `python -m pytest tests -q`.
  - Lint: `python -m ruff check src tests`.
  - Types: `python -m mypy src/embodied_human --ignore-missing-imports`.
  - After changing defaults, regenerate `docs/parameters.md` with
    `python examples/parameter_table.py`, and rerun
    `python -m embodied_human.bench --out evidence/benchmark-baseline.json`.
  - After changing the attention, estimator or eye modules, refit the
    glance sampler with `python examples/calibrate_glances.py --out
    evidence/glance-calibration.json`. Report fitted and held-out results
    separately.
  - After changing the vestibular or response modules, rerun
    `examples/fit_motion_sickness.py` and `examples/fit_drt_load.py` and
    update the fitted constants they report.
- **CI.** `.github/workflows/tests.yml` runs the offline suite, ruff, mypy
  and the build on Python 3.10 and 3.12; MyoSuite tests are skipped there.
- **Records.** Record what was run, with versions, in VALIDATION.md.
- Keep commit authorship with the human owner. Do not add AI co-author
  trailers.
