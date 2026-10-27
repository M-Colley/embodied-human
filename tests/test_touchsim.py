import math
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import embodied_human as eh
from embodied_human.touch import TouchSimTouch, VibrotactileTouch


def test_touchsim_registration_is_lazy_and_keeps_psychophysics():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['touchsim'] = None; "
            "import embodied_human as eh; "
            "config = eh.HumanConfig().replace('touch', 'touchsim'); "
            "h = eh.build_human(config, seed=3); "
            "assert h.touch.threshold(125, 'steering_wheel') > 0; "
            "assert 'touchsim' in eh.available('touch')['touch']",
        ],
        check=True,
    )
    touch = TouchSimTouch()
    baseline = VibrotactileTouch()
    level = touch.sensation_level(0.5, 125, "steering_wheel")
    assert level == baseline.sensation_level(0.5, 125, "steering_wheel")
    assert touch.latency(level) == baseline.latency(level)
    assert touch.perceived_intensity(0.5, 125, "steering_wheel") == (
        baseline.perceived_intensity(0.5, 125, "steering_wheel")
    )
    assert all(row["citation"] for row in touch.params.provenance())


def test_missing_touchsim_has_actionable_error(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError("missing touchsim", name="touchsim")

    monkeypatch.setattr("embodied_human.touch.import_module", missing)
    with pytest.raises(ImportError, match="requirements-touchsim.txt"):
        TouchSimTouch().afferent_response(
            [0.0, 0.1, 0.0], pin_locations=[[0, 0]], afferent_locations=[[0, 0]]
        )


def test_touchsim_units_geometry_determinism_and_numpy_settings(monkeypatch):
    afferents = []

    class Afferent:
        affparams = {c: np.zeros((2, 13)) for c in ("SA1", "RA", "PC")}

        def __init__(self, affclass, **kwargs):
            self.affclass = affclass
            afferents.append((affclass, kwargs))

    class Population:
        def __init__(self, *args):
            self.afferents = args

        def response(self, stimulus):
            assert stimulus.fs == 5000.0
            assert stimulus.pin_radius == 0.5
            np.testing.assert_equal(stimulus.trace, [[0.0, 0.2, 0.0]])
            np.testing.assert_equal(stimulus.location, [[1.0, 2.0]])
            np.seterr(all="ignore")  # upstream mutates error handling
            return SimpleNamespace(spikes=[np.array([0.0002])] * len(self.afferents))

    backend = SimpleNamespace(
        Afferent=Afferent,
        AfferentPopulation=Population,
        Stimulus=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr("embodied_human.touch.import_module", lambda _: backend)
    before = np.geterr()
    result = TouchSimTouch(afferent_classes=("RA",)).afferent_response(
        [0, 0.2, 0],
        pin_locations=[[1, 2]],
        afferent_locations=[[0, 0], [3, 4]],
    )
    assert np.geterr() == before
    assert len(result.spikes) == 2
    assert result.duration == pytest.approx(3 / 5000)
    np.testing.assert_allclose(result.rates, [5000 / 3, 5000 / 3])
    np.testing.assert_equal(result.locations, [[0, 0], [3, 4]])
    assert result.afferent_classes == ("RA", "RA")
    assert all(not args["noisy"] and args["idx"] == 0 for _, args in afferents)
    assert all(not args["delay"] for _, args in afferents)


@pytest.mark.parametrize("frequency", [0, -10, math.nan, math.inf])
def test_touch_threshold_rejects_invalid_frequency(frequency):
    with pytest.raises(ValueError, match="frequency"):
        VibrotactileTouch().threshold(frequency, "fingertip")


def test_touch_latency_handles_extreme_levels_without_overflow():
    touch = VibrotactileTouch()
    assert math.isfinite(touch.latency(100000.0))
    assert touch.latency(math.inf) == touch.latency(100000.0)
    assert touch.latency(-math.inf) == math.inf
    with pytest.raises(ValueError, match="NaN"):
        touch.latency(math.nan)
    with pytest.raises(ValueError, match="amplitude"):
        touch.sensation_level(math.nan, 125, "fingertip")
    with pytest.raises(ValueError, match="amplitude"):
        touch.perceived_intensity(math.inf, 125, "steering_wheel")
    with pytest.raises(ValueError, match="latency parameters"):
        TouchSimTouch(latency_exponent=1)


@pytest.mark.parametrize(
    "params",
    [
        {"sample_rate": 0},
        {"sample_rate": math.nan},
        {"pin_radius": -1},
        {"model_index": True},
        {"model_index": 0.5},
        {"afferent_classes": "RA"},
        {"afferent_classes": ()},
        {"afferent_classes": ("SA1", "SA1")},
    ],
)
def test_touchsim_rejects_invalid_protocol(params):
    with pytest.raises(ValueError):
        TouchSimTouch(**params)


@pytest.mark.parametrize(
    "trace,pins,locations",
    [
        ([0], [[0, 0]], [[0, 0]]),
        ([0, 0.1], [[0, 0]], [[0, 0]]),
        ([0, math.nan, 0], [[0, 0]], [[0, 0]]),
        ([[0, 0, 0], [0, 0, 0]], [[0, 0]], [[0, 0]]),
        ([0, 0, 0], [[0, 0]], []),
        ([0, 0, 0], [[0, math.inf]], [[0, 0]]),
        ([[0, 0, 0], [0, 0, 0]], [[0, 0], [0, 0]], [[0, 0]]),
        ([[0, 0, 0], [0, 0, 0]], [[0, 0], [0.5, 0]], [[0, 0]]),
    ],
)
def test_touchsim_rejects_invalid_stimuli(trace, pins, locations):
    with pytest.raises(ValueError):
        TouchSimTouch().afferent_response(
            trace, pin_locations=pins, afferent_locations=locations
        )


def test_real_touchsim_rejects_two_sample_nan_profile():
    pytest.importorskip("touchsim")
    # Upstream's centered derivative leaves both boundary samples NaN for N=2.
    # Reject the trace before invoking mechanics instead of reporting no spikes.
    with pytest.raises(ValueError, match="at least 3 samples"):
        TouchSimTouch().afferent_response(
            [0.0, 0.1], pin_locations=[[0, 0]], afferent_locations=[[0, 0]]
        )


def test_real_touchsim_spikes_repeat_and_preserve_random_streams():
    pytest.importorskip("touchsim")
    numba = pytest.importorskip("numba")
    # Keep the optional smoke test inexpensive on many-core developer machines.
    threads = numba.get_num_threads()
    numba.set_num_threads(1)
    try:
        touch = eh.create("touch", "touchsim")
        trace = 0.2 * np.sin(2 * np.pi * 100 * np.arange(1000) / 5000)
        before = np.random.get_state()
        first = touch.afferent_response(
            trace, pin_locations=[[0, 0]], afferent_locations=[[0, 0]]
        )
        second = touch.afferent_response(
            trace, pin_locations=[[0, 0]], afferent_locations=[[0, 0]]
        )
        after = np.random.get_state()
        assert first.afferent_classes == ("SA1", "RA", "PC")
        assert all(spikes.size > 0 for spikes in first.spikes)
        for a, b in zip(first.spikes, second.spikes):
            np.testing.assert_equal(a, b)
        for a, b in zip(before, after):
            np.testing.assert_equal(a, b)
    finally:
        numba.set_num_threads(threads)
