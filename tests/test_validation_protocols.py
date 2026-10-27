"""Data roles and nested streams must support valid comparisons."""

from embodied_human.bench.protocol import benchmark_seed, protocol_seed, protocol_seeds
from embodied_human.bench.references import REFERENCES
from embodied_human.bench.validation import evaluation_for


def test_every_reference_has_an_explicit_data_role_and_identifier():
    assert all(evaluation_for(name).evaluation_data for name in REFERENCES)
    assert evaluation_for("response.modality_order").role == "consistency"
    assert evaluation_for("affect.emotion_rmse").role == "unclassified"
    assert evaluation_for("driving.takeover_hands").role == "calibration"
    throughput = evaluation_for("motor.touch_throughput")
    assert throughput.role == "held_out"
    assert throughput.evaluation_data == ("mackenzie2015.smartphone_throughput",)
    assert evaluation_for("motor.touch_slope").role == "consistency"
    assert evaluation_for("driving.takeover_gaze").role == "held_out"
    assert evaluation_for("driving.incar_glance_speed_elasticity").role == "exploratory"


def test_nested_replicates_do_not_share_trial_seeds():
    groups = []
    for seed in range(20):
        with benchmark_seed(seed):
            assert protocol_seed() == seed
            groups.append(set(protocol_seeds(64)))
            assert protocol_seeds(64) == protocol_seeds(64)
    assert groups[0] == set(range(64))
    assert sum(map(len, groups)) == len(set().union(*groups))
