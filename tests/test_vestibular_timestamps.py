"""Sample time alignment against an independent analytical reduction."""

import math

import numpy as np
import pytest
from scipy.integrate import quad

from embodied_human.bench.senses import SICKNESS_TIME_COURSE, lateral_roll_motion
from embodied_human.vestibular import SubjectiveVertical


def test_svc_reports_the_initial_state_at_the_first_sample():
    model = SubjectiveVertical()
    acceleration = np.array([[0.0, 0.0, 2.0]])
    np.testing.assert_array_equal(model.simulate(0.1, acceleration), [0.0])
    np.testing.assert_array_equal(model.conflict(0.1, acceleration), [0.0])


def test_svc_samples_match_analytic_constant_acceleration_response():
    # With internal acceleration and feedback disabled, the sensed vertical
    # follows a first order lag while the expected vertical stays at rest.
    # The MSI is the exact convolution of its Hill drive with the second
    # order sickness lag. This solution uses the physical sample times.
    model = SubjectiveVertical(
        accel_gain=0.0, accel_feedback=0.0, vertical_feedback=0.0,
        vertical_lag=0.5, sickness_lag=2.0, conflict_scale=0.7,
    )
    dt = 0.01
    times = np.arange(0.0, 8.0 + dt / 2, dt)
    acceleration = np.zeros((len(times), 3))
    acceleration[:, 2] = 2.0
    conflict = model.conflict(dt, acceleration)
    np.testing.assert_allclose(
        conflict, 2.0 * (1.0 - np.exp(-times / 0.5)), atol=2e-9,
    )
    msi = model.simulate(dt, acceleration)
    for second in (0, 1, 2, 4, 8):
        def integrand(u):
            ratio = (2.0 * (1.0 - math.exp(-u / 0.5)) / 0.7) ** 2
            drive = 85.0 * ratio / (1.0 + ratio)
            elapsed = second - u
            return drive * elapsed / 2.0**2 * math.exp(-elapsed / 2.0)

        expected = quad(integrand, 0.0, second, epsabs=1e-10)[0]
        assert msi[round(second / dt)] == pytest.approx(expected, abs=1e-7)


def test_general_vertical_conflict_matches_the_linear_transfer_function():
    # Reducing the original In3 equations to translation alone gives
    # d/a = (1-Ka)s / [tau*s^2 + (1+Kac*tau)*s + Kvc]. The acceleration
    # feedback integrator cannot be dropped when testing the full baseline.
    model = SubjectiveVertical()
    p, dt = model.params, 0.01
    times = np.arange(0.0, 80.0, dt)
    frequency = np.array([0.083, 0.167, 0.333, 0.6, 0.7])
    acceleration = np.zeros((len(frequency), len(times), 3))
    acceleration[..., 2] = np.sin(2 * math.pi * frequency[:, None] * times)
    measured = model.conflict(dt, acceleration)[:, times > 60.0].max(axis=1)
    s = 2j * math.pi * frequency
    expected = np.abs((1.0 - p.accel_gain) * s / (
        p.vertical_lag * s**2
        + (1.0 + p.accel_feedback * p.vertical_lag) * s
        + p.vertical_feedback
    ))
    np.testing.assert_allclose(measured, expected, rtol=0.001)


def test_mccauley_datum_17_uses_the_printed_observed_column():
    # McCauley et al. (1976), Appendix B, printed p. 57. The observed
    # column reads 28; the separately printed +6 difference is inconsistent.
    row = next(row for row in SICKNESS_TIME_COURSE if row[:3] == (0.18, 0.170, 15))
    assert row == (0.18, 0.170, 15, 28, 21)


def test_roll_motion_includes_the_requested_exposure_endpoint():
    motion = lateral_roll_motion(0.2, True, 0.05, 10.0, velocity=1.0)
    assert all(values.shape == (201, 3) for values in motion)
