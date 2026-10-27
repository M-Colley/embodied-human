import math

import numpy as np
import pytest

from embodied_human.tasks.tapping import throughput


def movements():
    return (
        np.zeros((3, 2)),
        np.tile([0.1, 0], (3, 1)),
        np.array([[0.099, 0], [0.101, 0], [0.1, 0]]),
        np.full(3, 0.5),
        0.01,
    )


def test_perfect_endpoints_report_undefined_throughput_without_division_by_zero():
    starts, targets, _, times, width = movements()
    result = throughput(starts, targets, targets.copy(), times, width)
    assert result.effective_width == 0
    assert result.effective_amplitude == pytest.approx(0.1)
    assert result.error_rate == 0
    assert result.movement_time == 0.5
    assert math.isnan(result.effective_id) and math.isnan(result.throughput)


def test_nonpositive_effective_amplitude_is_undefined():
    starts, targets, endpoints, times, width = movements()
    endpoints[:, 0] -= 0.2
    result = throughput(starts, targets, endpoints, times, width)
    assert result.effective_amplitude < 0
    assert math.isnan(result.effective_id) and math.isnan(result.throughput)
    assert result.error_rate == 1


@pytest.mark.parametrize(
    ("index", "replacement"),
    [
        (0, np.zeros((1, 2))),
        (0, np.zeros(3)),
        (0, np.zeros((3, 3))),
        (1, np.zeros((2, 2))),
        (2, np.zeros((3, 1))),
        (3, np.ones((3, 1))),
        (3, np.ones(2)),
        (3, [0.5, 0.5, 0]),
        (3, [0.5, 0.5, -1]),
        (3, [0.5, 0.5, np.nan]),
        (3, [0.5, 0.5, np.inf]),
        (4, 0),
        (4, -0.01),
        (4, np.nan),
        (4, np.inf),
    ],
)
def test_invalid_measurements_raise_value_error(index, replacement):
    args = list(movements())
    args[index] = replacement
    with pytest.raises(ValueError):
        throughput(*args)


def test_nonfinite_coordinates_and_zero_task_axes_raise_value_error():
    for index in range(3):
        args = list(movements())
        args[index][0, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            throughput(*args)
    starts, targets, ends, times, width = movements()
    starts[1] = targets[1]
    with pytest.raises(ValueError, match="task axis"):
        throughput(starts, targets, ends, times, width)


def test_ordinary_measurements_keep_the_published_formula():
    starts, targets, endpoints, times, width = movements()
    result = throughput(starts, targets, endpoints, times, width)
    effective_width = 4.133 * np.std([-0.001, 0.001, 0], ddof=1)
    assert result.effective_width == pytest.approx(effective_width)
    assert result.throughput == pytest.approx(
        math.log2(0.1 / effective_width + 1) / 0.5
    )
