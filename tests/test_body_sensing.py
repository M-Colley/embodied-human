"""Analytic inertial head kinematics in a moving vehicle frame."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("myosuite_carla")

from myosuite_carla.vehicle_loads import VehicleMotion  # noqa: E402

from embodied_human.agents.body import BodySensing  # noqa: E402


def _sensing(
    *,
    head=(1.0, 0.0, 1.6),
    pelvis=(0.0, 0.0, 0.0),
    com=(0.0, 0.0, 0.0),
    seat_offset=(0.0, 0.0, 0.0),
    velocity=(0.0, 0.0, 0.0),
    motion=None,
):
    def xyz(point):
        return " ".join(map(str, point))

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><option gravity="0 0 0"/><worldbody>'
        f'<body name="pelvis" pos="{xyz(pelvis)}"/>'
        f'<body name="head" pos="{xyz(head)}"><freejoint/>'
        f'<geom size="0.1" mass="1" pos="{xyz(com)}"/>'
        "</body></worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    data.qvel[:3] = velocity
    mujoco.mj_forward(model, data)
    driver = SimpleNamespace(
        model=model,
        data=data,
        config=SimpleNamespace(seat_offset=seat_offset),
        motion=motion or VehicleMotion(),
    )
    return BodySensing(SimpleNamespace(driver=driver))


def test_fixed_head_receives_vehicle_centripetal_acceleration():
    sensing = _sensing(
        motion=VehicleMotion(angular_velocity=(0.0, 0.0, math.degrees(2.0)))
    )
    acceleration, angular, up = sensing.head_motion()
    # Cockpit +x is left; centripetal acceleration is -4x at this point.
    np.testing.assert_allclose(acceleration, [0.0, -4.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(angular, [0.0, 0.0, -2.0], atol=1e-12)
    np.testing.assert_allclose(up, [0.0, 0.0, 9.81], atol=1e-12)


def test_head_com_and_seat_offset_define_the_vehicle_lever_arm():
    sensing = _sensing(
        head=(2.0, 2.0, 2.0),
        pelvis=(2.0, 3.0, 0.4),
        com=(0.2, 0.0, 0.1),
        seat_offset=(1.0, -0.5, 0.2),
        motion=VehicleMotion(
            acceleration=(1.0, 2.0, 3.0),
            angular_velocity=(0.0, 0.0, math.degrees(2.0)),
            angular_acceleration=(0.0, 0.0, math.degrees(3.0)),
        ),
    )
    acceleration, _, _ = sensing.head_motion()
    # Vehicle origin=(1.5,4,.2); head COM=(2.2,2,2.1), so r=(.7,-2,1.9).
    # a0=(-2,-1,3), alpha x r=(-6,-2.1,0), omega x(omega x r)=(-2.8,8,0).
    np.testing.assert_allclose(acceleration, [-4.9, -10.8, 3.0], atol=1e-12)


def test_relative_head_translation_receives_coriolis_acceleration():
    sensing = _sensing(
        velocity=(0.0, 2.0, 0.0),
        motion=VehicleMotion(angular_velocity=(0.0, 0.0, math.degrees(2.0))),
    )
    acceleration, _, _ = sensing.head_motion()
    # Centripetal=-4x and 2 omega x v_rel=+8x in cockpit coordinates.
    np.testing.assert_allclose(acceleration, [0.0, 4.0, 0.0], atol=1e-12)


def test_relative_head_acceleration_is_added_once():
    sensing = _sensing(motion=VehicleMotion(acceleration=(0.0, 0.0, 2.0)))
    driver = sensing.driver
    driver.data.qfrc_applied[:3] = [1.0, 2.0, 3.0]
    mujoco.mj_forward(driver.model, driver.data)
    acceleration, _, _ = sensing.head_motion()
    np.testing.assert_allclose(acceleration, [-2.0, 1.0, 5.0], atol=1e-12)


def test_rotating_head_com_composes_with_vehicle_rotation():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><option gravity="0 0 0"/><worldbody><body name="pelvis"/>'
        '<body name="head" pos="0 0 1.6"><joint axis="0 0 1"/>'
        '<geom size="0.1" mass="1" pos="1 0 0"/>'
        "</body></worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    data.qvel[:] = 1.0
    mujoco.mj_forward(model, data)
    driver = SimpleNamespace(
        model=model,
        data=data,
        config=SimpleNamespace(seat_offset=(0.0, 0.0, 0.0)),
        motion=VehicleMotion(angular_velocity=(0.0, 0.0, math.degrees(2.0))),
    )
    sensing = BodySensing(SimpleNamespace(driver=driver))
    acceleration, angular, _ = sensing.head_motion()
    # The head spins +1z relative to the car's -2z, so its inertial rate is -1z.
    # At radius 1m the complete inertial centripetal acceleration is -1x.
    np.testing.assert_allclose(acceleration, [0.0, -1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(angular, [0.0, 0.0, -1.0], atol=1e-12)


def test_current_head_orientation_rotates_inertial_measurements():
    sensing = _sensing(
        motion=VehicleMotion(angular_velocity=(0.0, 0.0, math.degrees(2.0)))
    )
    driver = sensing.driver
    # Rotate 90 degrees about cockpit z after the reference pose was captured.
    driver.data.qpos[3:] = [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]
    mujoco.mj_forward(driver.model, driver.data)
    acceleration, angular, _ = sensing.head_motion()
    np.testing.assert_allclose(acceleration, [-4.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(angular, [0.0, 0.0, -2.0], atol=1e-12)
