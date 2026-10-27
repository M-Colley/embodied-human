# TouchSim afferent integration

The optional `touchsim` implementation adds the actual TouchSim skin mechanics
and SA1, RA and PC afferent spike generator to the `touch` category. Upstream's
[MIT licence](https://github.com/hsaal/touchsim/blob/c7ece893c1ec75f2de10ade34422e0090a3093b9/LICENSE)
and [package metadata](https://github.com/hsaal/touchsim/blob/c7ece893c1ec75f2de10ade34422e0090a3093b9/pyproject.toml)
were verified on 26 October 2026. The integration invokes the installed package;
it does not vendor its implementation or change this package's Apache licence.
The model publication is [Saal, Delhaye et al. (2017)](https://doi.org/10.1073/pnas.1704856114).

From a source checkout or unpacked source distribution, install the package
and the verified licensed TouchSim revision:

```sh
python -m pip install -e .
python -m pip install -r requirements-touchsim.txt
```

TouchSim is not currently distributed on PyPI. `requirements-touchsim.txt`
therefore pins Git revision `c7ece893c1ec75f2de10ade34422e0090a3093b9`, which
requires Git and network access during this opt-in installation. The file is
included in the source distribution. Core imports and the default human continue
to require only NumPy and SciPy. Keeping the direct Git reference in a separate
requirements file preserves PyPI-compatible package metadata.

```python
import numpy as np
import embodied_human as eh

human = eh.build_human(eh.HumanConfig().replace("touch", "touchsim"), seed=7)
touch = human.touch

# One circular pin, oscillating by 0.2 mm on a flat skin patch for 0.2 s.
time = np.arange(1000) / touch.params.sample_rate
trace = 0.2 * np.sin(2 * np.pi * 100 * time)
response = touch.afferent_response(
    trace,
    pin_locations=[[0.0, 0.0]],
    afferent_locations=[[0.0, 0.0], [2.0, 0.0]],
)
print(response.afferent_classes)  # SA1, RA, PC at each supplied location
print(response.rates)             # mean spikes/s for each afferent
print(response.spikes[0])         # spike times in seconds from trial onset
```

For multiple pins, pass a `(pins, samples)` indentation array and a `(pins, 2)`
location array. Every trace requires at least three original samples and must
span at least three samples at the internal 5000 Hz neural rate. Two-sample
traces leave upstream's centered mechanical derivative undefined and are
rejected. Displacement, pin radius and both sets of skin coordinates are
in **millimetres**, the native TouchSim unit. This is an indentation input,
not acceleration, contact force, or vehicle vibration. Conversion of a force
signal requires a contact-mechanics model and cannot be inferred from force
alone. Pin discs must not overlap; invalid geometry is rejected rather than
letting upstream silently change the pin radius.

Each afferent location receives one fitted model per configured class; this is
a small explicit population, not a reconstruction of hand receptor density.
`model_index` chooses the published fitted neuron within each class. Locations
use upstream's flat `null_surface`; whole-hand anatomical placement and surface
geodesics remain available through the native TouchSim API. Defaults and their
provenance are in `TouchSimParameters` and `touch.params.provenance()`.

Trials are independent. TouchSim's native membrane noise is disabled and neuron
indices are fixed, making responses reproducible without seeding or altering
Python/NumPy global random generators. Upstream parallel Numba noise has its own
random state and cannot honestly be presented as controlled by the human seed.
The adapter also restores NumPy error settings changed by upstream mechanics.

`sensation_level`, `latency` and `perceived_intensity` still use the existing
Morioka–Griffin and Piéron psychophysical baseline. Spikes are an additional
pathway; there is no fitted spike-to-perception decoder, no anatomical whole-body
touch model, and no claim that the TouchSim pathway improves DRT or takeover
benchmarks. Existing aggregate touch benchmarks remain consistency checks of
their own source parameters; a successful optional spike smoke test checks
integration and determinism, not held-out human validity.

Other useful native upstream capabilities include indentation ramps, arbitrary
multi-pin traces, bar/circle shapes, whole-hand populations, and spike histograms.
Use them for explicit tactile experiments; incorporating them into task-level
perception needs independent calibration and held-out evaluation.
