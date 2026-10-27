import math
from types import SimpleNamespace

import numpy as np
import pytest

import embodied_human as eh
from embodied_human.bench.benchmarks import BENCHMARKS, run
from embodied_human.bench.references import REFERENCES, Reference
from embodied_human.memory import ActRMemory
from embodied_human.tasks.drt import DRTConfig, DRTTask
from embodied_human.tasks.nback import NBackConfig, run_nback
from embodied_human.tasks.surt import SuRTConfig, SuRTTask, make_screen, render
from embodied_human.tasks.tapping import (
    TappingCondition,
    iso_order,
    target_positions,
    throughput,
)


def test_surt_screens_have_one_target_and_score_zones():
    rng = np.random.default_rng(0)
    config = SuRTConfig()
    screen = make_screen(config, rng, 0.0)
    assert sum(c.target for c in screen.circles) == 1
    image = render(screen, config)
    assert image.shape == (160, 320, 3) and image.max() == 230
    task = SuRTTask(config, rng=rng)
    zone = task.screen.target_zone
    event = SimpleNamespace(
        source="screen",
        kind="touch_begin",
        time=1.0,
        data={"u": (zone + 0.5) / config.zones},
    )
    outputs = task.update(SimpleNamespace(events=[event]), 1.0)
    assert outputs["surt.correct"] == 1.0 and outputs["screen.image"].shape == (
        160,
        320,
        3,
    )
    with pytest.raises(ValueError):
        make_screen(SuRTConfig(resolution=(40, 40), distractors=200), rng, 0.0)


def test_drt_classifies_responses_per_iso_17488():
    task = DRTTask(rng=np.random.default_rng(0))
    onset = None
    t = 0.0
    while onset is None:
        t += 0.01
        task.step(t)
        onset = task.onset
    assert DRTConfig().interval[0] <= onset <= DRTConfig().interval[1]
    assert task.respond(onset + 0.05) == "premature"
    assert task.respond(onset + 0.5) == "hit"
    assert task.respond(onset + 0.6) == "repeated"
    assert not task.on
    for _ in range(1000):
        t += 0.01
        task.step(t)
    summary = task.result.summary()
    assert summary["stimuli"] >= 2 and task.result.misses >= 1
    assert summary["mean_rt_s"] == pytest.approx(0.5)


def test_nback_errors_grow_when_items_leave_the_focus():
    rng = np.random.default_rng(0)
    easy = run_nback(ActRMemory(), NBackConfig(level=1, digits=30), rng).summary()
    hard = run_nback(
        ActRMemory(capacity=1), NBackConfig(level=2, digits=30), rng
    ).summary()
    assert easy["accuracy"] == 1.0 and hard["accuracy"] < easy["accuracy"]


def test_tapping_geometry_and_throughput_formula():
    assert iso_order(5) == [0, 3, 1, 4, 2]
    with pytest.raises(ValueError):
        iso_order(4)
    condition = TappingCondition(0.1, 0.01, targets=5)
    positions = target_positions(condition)
    assert np.allclose(np.linalg.norm(positions, axis=1), 0.05)
    # Movements of exactly the amplitude with symmetric overshoots.
    starts = np.array([[0.0, 0.0]] * 4)
    targets = np.array([[0.1, 0.0]] * 4)
    ends = np.array([[0.101, 0.0], [0.099, 0.0], [0.101, 0.0], [0.099, 0.0]])
    result = throughput(starts, targets, ends, np.full(4, 0.5), 0.01)
    sd = np.std([0.001, -0.001, 0.001, -0.001], ddof=1)
    assert result.effective_width == pytest.approx(4.133 * sd)
    assert result.throughput == pytest.approx(math.log2(0.1 / (4.133 * sd) + 1) / 0.5)


def test_reference_scores():
    reference = Reference("x.y", 2.0, 1.0, 3.0, "s", "c", None, "")
    assert reference.score(2.5) == 1.0 and reference.score(4.0) == pytest.approx(0.5)
    assert reference.score(-5.0) == 0.0
    assert all(r.low <= r.value <= r.high for r in REFERENCES.values())


@pytest.mark.parametrize(
    "name",
    [
        "eyes.main_sequence",
        "eyes.latency",
        "objects.peripheral_acuity",
        "vision.foveation",
        "memory.span",
    ],
)
def test_module_benchmarks_pass_for_the_baseline(name):
    (result,) = run(eh.build_human(seed=0), [name])
    assert result.passed, result.as_dict()


def test_tapping_benchmark_runs_with_physical_contact():
    pytest.importorskip("embodied_ui")
    (result,) = run(eh.build_human(seed=0), ["motor.touch_tapping"])
    assert result.skipped is None
    assert 4.0 < result.metrics["motor.touch_throughput"] < 9.0
    assert result.details["conditions"][0]["error_rate"] < 0.4


def test_a_replaced_module_changes_the_benchmark():
    config = eh.HumanConfig().replace("eyes", "main_sequence", duration_slope=0.005)
    (result,) = run(eh.build_human(config, seed=0), ["eyes.main_sequence"])
    assert not result.passed and result.metrics["eyes.duration_slope"] == pytest.approx(
        5.0
    )


def test_unknown_benchmarks_are_rejected():
    with pytest.raises(ValueError):
        run(eh.build_human(seed=0), ["nope"])
    assert "driving.takeover" in BENCHMARKS


def test_cli_replacements_accept_parameters():
    from embodied_human.bench.__main__ import _config

    config = _config(["attention=uncertainty_sampler:return_sd=0.8,maximum_glance=4"])
    assert config.attention.params == {"return_sd": 0.8, "maximum_glance": 4}
    with pytest.raises(SystemExit):
        _config(["hands=x"])
    with pytest.raises(SystemExit):
        _config(["attention=seev:value"])
