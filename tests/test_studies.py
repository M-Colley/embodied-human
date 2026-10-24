"""Inference, simulated participants, study statistics and traces."""

import csv
import math

import numpy as np
import pytest

import embodied_human as eh
from embodied_human.agents.road import simulate_glances
from embodied_human.bench.benchmarks import run
from embodied_human.bench.metrics import bootstrap_ci, elasticity, kendall_tau
from embodied_human.core import HumanConfig, Source
from embodied_human.inference import Prior, abc_rejection, distance
from embodied_human.population import (
    BASELINE_VARIATIONS,
    OLDER_ADULTS,
    Population,
    Variation,
    applicable,
)


def test_abc_recovers_a_known_parameter():
    def simulate(params):
        rng = np.random.default_rng(1)
        samples = rng.normal(params["mu"], 1.0, 400)
        return {"mean": float(samples.mean())}

    posterior = abc_rejection(
        simulate, {"mean": 2.0}, {"mean": 0.1}, [Prior("mu", -5.0, 5.0)], samples=256
    )
    assert posterior.median()["mu"] == pytest.approx(2.0, abs=0.2)
    low, high = posterior.interval()["mu"]
    assert low < 2.0 < high and high - low < 1.0
    assert posterior.as_dict()["accepted"] == 26 and len(posterior.all_samples) == 256


def test_priors_and_distances():
    log_prior = Prior("x", 0.1, 10.0, log=True)
    assert log_prior.scale(np.array([0.5]))[0] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        Prior("x", 0.0, 1.0, log=True)
    assert distance({"a": 1.0}, {"a": 0.0}, {"a": 0.5}) == pytest.approx(2.0)
    assert distance({"a": math.nan}, {"a": 0.0}, {"a": 1.0}) == math.inf


def test_population_varies_people_and_shifts_groups():
    population = Population(variations=BASELINE_VARIATIONS)
    people = population.participants(50, seed=3)
    assert people == population.participants(50, seed=3)
    noise = [p.proprioception.params["joint_noise_deg"] for p in people]
    assert np.median(noise) == pytest.approx(3.0, rel=0.1)
    assert 0.08 < np.std(noise) / np.mean(noise) < 0.25
    older = Population(profile=OLDER_ADULTS).typical()
    assert older.proprioception.params["joint_noise_deg"] == pytest.approx(3.6)
    description = population.describe()
    assert description["variations"][0]["kind"] == "measured"
    assert {v["kind"] for v in description["variations"][1:]} == {"assumed"}
    with pytest.raises(ValueError):
        Variation("hands", "x", 0.1)


def test_variations_apply_only_to_modules_that_have_them():
    config = HumanConfig().replace("attention", "seev")
    kept = applicable(config, BASELINE_VARIATIONS)
    assert "attention" not in {v.category for v in kept} and len(kept) == 3
    person = Population(config, kept).sample(np.random.default_rng(0))
    eh.build_human(person, seed=0)


def test_normal_variation_and_sources():
    variation = Variation(
        "motor", "intercept", 0.01, "normal", Source("x", None, "assumed")
    )
    values = [variation.sample(0.15, np.random.default_rng(i)) for i in range(200)]
    assert np.std(values) == pytest.approx(0.01, rel=0.2)


def test_study_statistics():
    low, high = bootstrap_ci(np.random.default_rng(0).normal(1.0, 0.2, 40))
    assert low < 1.0 < high and high - low < 0.3
    assert all(math.isnan(x) for x in bootstrap_ci([1.0, 1.0, 1.0]))
    assert kendall_tau([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert elasticity([10, 20, 40], [4.0, 2.0, 1.0]) == pytest.approx(-1.0)


def test_glances_on_a_straight_road_are_explained():
    log = simulate_glances(eh.build_human(seed=0), speed=14.0, seconds=30.0, seed=0)
    metrics = log.glance_metrics()
    assert metrics["glances"] >= 5
    assert set(metrics["returns_by_reason"]) <= {
        "uncertainty",
        "maximum_glance",
        "task_done",
    }
    assert sum(metrics["returns_by_reason"].values()) >= metrics["glances"]
    trace = log.trace()
    assert len(trace["time_s"]) == len(trace["belief_offset_sd_m"]) == 1500
    assert np.all(trace["true_offset_m"] == 0.0)
    # Off the road the belief's uncertainty grows; on it, it shrinks.
    sd, area = trace["belief_offset_sd_m"], trace["area"]
    growth = np.diff(sd)[area[1:] == "display"]
    assert np.mean(growth > 0) > 0.95


def test_faster_driving_shortens_glances():
    human = eh.build_human(seed=0)
    slow = simulate_glances(human, speed=11.0, seconds=40.0).glance_metrics()
    fast = simulate_glances(human, speed=33.0, seconds=40.0).glance_metrics()
    assert fast["mean_offroad_glance_s"] < slow["mean_offroad_glance_s"]


def test_trace_is_written_as_csv(tmp_path):
    log = simulate_glances(eh.build_human(seed=0), seconds=2.0)
    path = log.write_csv(tmp_path / "trace" / "run.csv")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 100 and rows[0]["area"] == "road"
    assert float(rows[-1]["time_s"]) == pytest.approx(2.0, abs=0.03)


def test_glance_studies_run_without_myosuite():
    population, speed = run(
        eh.build_human(seed=0), ["driving.glance_population", "driving.glance_speed"]
    )
    assert (
        population.skipped is None and len(population.details["per_participant"]) == 20
    )
    low, high = population.details["mean_glance_ci95_s"]
    assert low <= population.metrics["driving.human_mean_glance"] <= high
    # Road uncertainty alone makes glances shorter at speed, as in occlusion.
    assert speed.metrics["driving.occlusion_speed_elasticity"] < -0.3


def test_touch_key_study_replicates_the_analysis():
    pytest.importorskip("embodied_ui")
    pytest.importorskip("mujoco")
    from embodied_human.bench.studies import touch_key_size

    result = touch_key_size(eh.build_human(seed=0), participants=8)
    errors = result.details["error_rate"]
    assert errors["7.5"]["mean"] > errors["27.5"]["mean"]
    assert result.details["differs_from_largest"]["27.5"] is False
    assert result.metrics["design.touch_key_asymptote"] in (7.5, 12.5, 17.5, 22.5, 27.5)
