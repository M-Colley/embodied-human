"""Vision evaluation must exercise the configured perception implementation."""

import pytest

from embodied_human import build_human
from embodied_human.bench.benchmarks import run
from embodied_human.vision.foveation import FoveatedVision


def test_resolution_benchmark_measures_the_returned_cutoff_field():
    class ChangedPerception(FoveatedVision):
        def perceive(self, *args, **kwargs):
            return FoveatedVision(e2=4.6).perceive(*args, **kwargs)

    human = build_human(seed=0)
    baseline = run(human, ["vision.foveation"], seed=0)[0]
    assert baseline.metrics["vision.e2"] == pytest.approx(2.3)
    human.modules["vision"] = ChangedPerception()
    changed = run(human, ["vision.foveation"], seed=0)[0]
    assert human.vision.parameters.e2 == 2.3
    assert changed.metrics["vision.e2"] == pytest.approx(4.6)
    assert not changed.passed


def test_resolution_benchmark_propagates_a_perception_failure():
    class FailedPerception(FoveatedVision):
        def perceive(self, *args, **kwargs):
            raise RuntimeError("perception failed")

    human = build_human(seed=0)
    human.modules["vision"] = FailedPerception()
    with pytest.raises(RuntimeError, match="perception failed"):
        run(human, ["vision.foveation"], seed=0)
