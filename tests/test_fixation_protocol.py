"""Independent participants and measured task geometry in fixation fitting."""

import numpy as np
import pytest

pandas = pytest.importorskip("pandas")

from embodied_human.driving_data import DatasetBundle  # noqa: E402
from embodied_human.fixation_protocol import FixationProtocol  # noqa: E402


def _bundle():
    rows = []
    for subject in range(6):
        for condition, angle in (("down1", (8, -12)), ("down2", (26, -24))):
            for trial in range(2):
                for glance in range(5):
                    duration = 0.7 + subject * 0.1 + glance * 0.05
                    rows.append(
                        dict(
                            subject_id=str(subject),
                            trial_id=f"{subject}:{condition}:{trial}",
                            condition=condition,
                            area="target",
                            boundary=False,
                            start_s=glance * 3.0,
                            end_s=glance * 3.0 + duration,
                            duration_s=duration,
                            speed_mps=24.0,
                            heading_deg=angle[0] + subject * 0.1,
                            pitch_deg=angle[1],
                        )
                    )
    return DatasetBundle(
        "test",
        "real_road",
        "CC BY 4.0",
        "fixture",
        glances=pandas.DataFrame(rows),
        trials=pandas.DataFrame([{}]),
    )


def test_test_participants_cannot_define_target_geometry():
    bundle = _bundle()
    original = FixationProtocol(bundle, seconds=10, seed=3)
    held = {source.split(":", 1)[1] for source in original.split.test}
    data = bundle.glances.copy()
    data.loc[data.subject_id.isin(held), "heading_deg"] = 90
    changed = DatasetBundle(
        "test", "real_road", "CC BY 4.0", "fixture", glances=data, trials=bundle.trials
    )
    altered = FixationProtocol(changed, seconds=10, seed=3)
    assert (
        original.metadata["target_centres_deg"]
        == altered.metadata["target_centres_deg"]
    )


def test_component_predictions_are_replayable_and_source_scoped():
    protocol = FixationProtocol(_bundle(), seconds=10, seed=3)
    groups = protocol.split.test[:1]
    parameters = dict(return_sd=0.11, threshold_spread=1.0, maximum_glance=2.0)
    first = protocol(parameters, groups, 100)
    second = protocol(parameters, groups, 100)
    assert first == second
    assert set(first) == set(groups)
    assert all(np.isfinite(v) for row in first.values() for v in row.values())
