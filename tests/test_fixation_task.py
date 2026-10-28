"""Task geometry and behavior for measured visual fixation protocols."""

import numpy as np
import pytest

from embodied_human import build_human
from embodied_human.agents.driver import AttentiveDriver
from embodied_human.agents.road import HoldLane, StraightRoad
from embodied_human.tasks.fixation import FixationTask


@pytest.mark.parametrize("target", [(0.0, 0.0), (float("nan"), 2), (181, 2)])
def test_target_requires_separate_finite_visual_geometry(target):
    with pytest.raises(ValueError):
        FixationTask(target)


def test_fixation_uses_gaze_without_search_or_manual_responses(monkeypatch):
    human = build_human(seed=4)

    def manual_response(*args, **kwargs):
        raise AssertionError("fixation has no manual response")

    monkeypatch.setattr(human.motor, "plan", manual_response)
    road = StraightRoad(speed=24.0, duration=20.0)
    agent = AttentiveDriver(human, road, FixationTask((8.0, -11.0)), reflex=HoldLane())
    assert agent.area((8.0, -11.0)) == "display"
    assert agent.area((0.0, 0.0)) == "road"

    def run():
        road.reset(4)
        agent.reset(4)
        for _ in range(1000):
            road.step(agent.act())
        assert {"road", "display"} <= set(agent.log.areas)
        assert not agent.log.screen_times
        return np.asarray(agent.log.gaze)

    np.testing.assert_array_equal(run(), run())
