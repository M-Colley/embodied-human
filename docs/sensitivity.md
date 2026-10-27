# Optional sensitivity analysis

The experiment runner supports actual SALib Morris screening and Sobol variance based analysis. SALib is optional; request validation does not import it. Sampling or statistical analysis requires `python -m pip install 'SALib>=1.6,<2'`. This integration was checked with SALib 1.6.0 using its published [sampling APIs](https://salib.readthedocs.io/en/latest/api/SALib.sample.html) and [analysis APIs](https://salib.readthedocs.io/en/latest/api/SALib.analyze.html).

A sensitivity request declares `method`, `parameters`, `benchmark`, `metric` and `samples`, with optional `seed` defaulting to 0 and `num_levels` defaulting to 4. Each factor supplies a module `category`, its dataclass `parameter`, and finite strictly increasing numeric `bounds`. Validation accepts only parameters declared as continuous floats on the selected implementation. Integer capacities, booleans, categorical choices, array parameters, duplicate factors and unknown fields are rejected. At most eight factors are supported, and exactly one selected benchmark must match the requested benchmark and contain the requested metric in the benchmark protocol catalogue.

```python
from embodied_human import HumanConfig
from embodied_human.sensitivity import sample_design, validate_sensitivity

config = HumanConfig()
spec = validate_sensitivity(
    {
        "method": "morris",
        "parameters": [
            {
                "category": "eyes",
                "parameter": "duration_slope",
                "bounds": [0.0015, 0.0030],
            },
            {
                "category": "eyes",
                "parameter": "duration_intercept",
                "bounds": [0.015, 0.030],
            },
        ],
        "benchmark": "eyes.main_sequence",
        "metric": "eyes.duration_slope",
        "samples": 4,
        "seed": 0,
        "num_levels": 4,
    },
    config,
    ["eyes.main_sequence"],
)
problem, design = sample_design(spec)
print(spec["evaluations"], design.shape)  # 12, (12, 2)
```

Morris uses `samples` trajectories and requires an even `num_levels` from 4 to 20. Its design has `samples * (factors + 1)` evaluations. Sobol requires a power of two `samples`, uses scrambling, and fixes second order estimation to false. Its design has `samples * (factors + 2)` evaluations. Both accept base sample counts from 4 to 256. These counts describe design points; the experiment runner separately enforces the full evaluation budget including declared simulation seeds. Canonical validation recomputes the count so an edited saved count cannot bypass that budget.

`sample_design(spec)` returns the actual SALib problem dictionary and an ordered numeric design. Evaluate every row without reordering or dropping it. `analyze_design(spec, problem, X, Y)` accepts one finite scalar result per row, checks the problem, matrix shape and input bounds, and invokes the matching SALib analyzer. The experiment runner can target the mean across the same declared simulation seeds at every design point; this measures sensitivity of that mean rather than treating simulation noise as another independent factor.

Morris results contain `mu`, `mu_star`, `sigma` and `mu_star_conf`. The effects are in output units for changes across the declared input ranges and are not automatically comparable to ranges chosen in a different experiment. Sobol results contain first order `S1`, total order `ST`, and `S1_conf` and `ST_conf`; interaction matrices are not produced. Confidence calculations use 100 bootstrap resamples at a nominal 95 percent level. Small designs can give unstable estimates, negative variance index estimates or undefined confidence intervals; estimates are not clipped to make them appear plausible.

The result reports `status`, an explicit `reason` when needed, parameter names, design counts, seed, confidence settings and the SALib version when an analyzer ran. Constant outputs produce `status="undefined"` and JSON null indices. Nonfinite outputs also produce an explicit undefined result rather than zero sensitivity. If SALib produces an undefined estimate or interval, that entry becomes null and the result remains explicitly undefined while retaining its finite estimates.

Sampling and bootstrap calculations use local NumPy generators, including when the declared seed is zero. SALib's Sobol analyzer has a legacy zero seed branch that otherwise uses the global random generator; passing a local generator avoids that side effect. Output scaling before Sobol analysis protects the normalized variance calculation from overflow without changing the indices.

Sensitivity estimates depend on the declared uniform independent parameter distributions. The chosen bounds require scientific justification. Constructor or benchmark failures at a design point are reported by the runner. The eye example evaluates the module's own coefficients and therefore provides a consistency exercise. Independent human evaluation remains pending. Sensitivity execution preserves the configured model parameters outside that experiment.
