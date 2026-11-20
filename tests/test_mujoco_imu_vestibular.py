"""Online vestibular tracking and standalone MuJoCo IMU/body head sensing."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from embodied_human.vestibular import (
    GRAVITY,
    RUB_TO_FLU,
    SubjectiveVertical,
    VestibularTracker,
    imu_to_head_kinematics,
)


def test_vestibular_tracker_matches_batch_simulate_and_conflict() -> None:
    rng = np.random.default_rng(42)
    dt = 0.02
    steps = 150
    batch = 3
    accel = (
        0.8
        * np.sin(np.linspace(0.0, 6.0, steps))[None, :, None]
        * rng.normal(size=(batch, 1, 3))
    )
    omega = (
        0.15
        * np.cos(np.linspace(0.0, 4.0, steps))[None, :, None]
        * rng.normal(size=(batch, 1, 3))
    )
    tilt = np.linspace(0.0, 0.2, steps)
    gravity = np.broadcast_to(
        np.stack(
            [
                -GRAVITY * np.sin(tilt),
                np.zeros_like(tilt),
                GRAVITY * np.cos(tilt),
            ],
            axis=-1,
        ),
        (batch, steps, 3),
    ).copy()

    svc = SubjectiveVertical()
    expected_msi = svc.simulate(dt, accel, omega, gravity)
    expected_conflict = svc.conflict(dt, accel, omega, gravity)

    tracker = VestibularTracker(svc)
    online_conflict = np.zeros_like(expected_conflict)
    online_msi = np.zeros_like(expected_msi)
    for k in range(steps):
        c_k, m_k = tracker.step(dt, accel[:, k, :], omega[:, k, :], gravity[:, k, :])
        online_conflict[:, k] = c_k
        online_msi[:, k] = m_k

    np.testing.assert_allclose(online_conflict, expected_conflict, atol=1e-13)
    np.testing.assert_allclose(online_msi, expected_msi, atol=1e-13)
    np.testing.assert_allclose(tracker.msi, expected_msi[:, -1], atol=1e-13)
    np.testing.assert_allclose(tracker.conflict, expected_conflict[:, -1], atol=1e-13)

    tracker.reset()
    c0, m0 = tracker.step(dt, accel[:, 0, :], omega[:, 0, :], gravity[:, 0, :])
    np.testing.assert_array_equal(c0, np.zeros(batch))
    np.testing.assert_array_equal(m0, np.zeros(batch))


def test_imu_to_head_kinematics_handles_rub_sensor_and_head_pitch() -> None:
    # Upright rest in RUB (+X right, +Y up, +Z back): upward gravity sits on +Y_sensor.
    # World is FLU (+X fwd, +Y left, +Z up), so columns of sensor in world are
    # [-Y_w, +Z_w, -X_w] = RUB_TO_FLU.T, with unit quaternion (0.5, 0.5, -0.5, -0.5).
    q_ws = np.array([0.5, 0.5, -0.5, -0.5])
    a_h, w_h, g_h = imu_to_head_kinematics(
        specific_force=np.array([0.0, GRAVITY, -1.5]),
        angular_velocity=np.array([-0.2, 0.4, -0.1]),
        orientation_wxyz=q_ws,
        sensor_to_head=RUB_TO_FLU,
    )
    np.testing.assert_allclose(a_h, [1.5, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(w_h, [0.1, 0.2, 0.4], atol=1e-12)
    np.testing.assert_allclose(g_h, [0.0, 0.0, GRAVITY], atol=1e-12)


def test_mujoco_head_sensing_imu_and_body_paths_agree_under_3d_neck_motion() -> None:
    mujoco = pytest.importorskip("mujoco")
    from embodied_human.agents.body import MujocoHeadSensing

    q_cam = f"{math.sqrt(0.5)} 0 {-math.sqrt(0.5)} 0"
    q_head_rest = f"{math.sqrt(0.5)} 0 0 {-math.sqrt(0.5)}"
    xml = f"""
    <mujoco>
      <option gravity="0 0 -9.81"/>
      <worldbody>
        <body name="torso" pos="0 0 1.0">
          <geom size="0.08" mass="10.0"/>
          <body name="head" pos="0 0 0.35" quat="{q_head_rest}">
            <joint name="neck_yaw" axis="0 1 0" limited="true" range="-1 1"/>
            <joint name="neck_pitch" axis="0 0 1" limited="true" range="-0.8 0.8"/>
            <joint name="neck_roll" axis="1 0 0" limited="true" range="-0.6 0.6"/>
            <geom size="0.09" mass="4.5"/>
            <site name="head_imu" pos="0.08 0.06 0" quat="{q_cam}"/>
          </body>
        </body>
      </worldbody>
      <actuator>
        <muscle name="BIClong" joint="neck_pitch"/>
      </actuator>
      <sensor>
        <accelerometer name="head_accel" site="head_imu"/>
        <gyro name="head_gyro" site="head_imu"/>
        <framequat name="head_quat" objtype="site" objname="head_imu"/>
      </sensor>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    sensing = MujocoHeadSensing((model, data))
    assert sensing.has_imu_sensors

    data.qpos[:] = [0.35, -0.25, 0.2]
    data.qvel[:] = [0.7, -0.5, 0.4]
    data.act[:] = [0.65]
    data.qfrc_applied[:] = [1.5, -0.8, 0.5]
    mujoco.mj_forward(model, data)

    imu_out = sensing.head_motion(prefer_imu=True)
    body_out = sensing.head_motion(prefer_imu=False)
    for got, ref in zip(imu_out, body_out, strict=True):
        np.testing.assert_allclose(got, ref, atol=1e-11)

    loads = sensing.loads()
    assert loads["elbow_right"] == pytest.approx(0.65)


def test_mujoco_head_sensing_composes_moving_base_and_drives_tracker() -> None:
    mujoco = pytest.importorskip("mujoco")
    from embodied_human.agents.body import MujocoHeadSensing

    xml = """
    <mujoco>
      <option gravity="0 0 -9.81"/>
      <worldbody>
        <body name="head" pos="1.0 0.0 1.5">
          <joint name="slide_y" type="slide" axis="0 1 0" stiffness="50" damping="4"/>
          <geom size="0.09" mass="4.5"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    env_stub = SimpleNamespace(
        model=model,
        data=data,
        base_kinematics=SimpleNamespace(
            lin_acc=np.array([0.5, 0.0, 1.2]),
            ang_vel=np.array([0.0, 0.0, 2.0]),
            ang_acc=np.array([0.0, 0.0, 0.0]),
        ),
    )
    sensing = MujocoHeadSensing(env_stub, imu_site=None, pivot=(0.0, 0.0, 1.5))
    # At r = (1, 0, 0) with omega = (0, 0, 2) and v_rel = 0, a_rel = 0:
    # a_world = (0.5, 0, 1.2) + (-4, 0, 0) = (-3.5, 0, 1.2)
    accel0, omega0, grav0 = sensing.head_motion()
    np.testing.assert_allclose(accel0, [-3.5, 0.0, 1.2], atol=1e-12)
    np.testing.assert_allclose(omega0, [0.0, 0.0, 2.0], atol=1e-12)
    np.testing.assert_allclose(grav0, [0.0, 0.0, GRAVITY], atol=1e-12)

    tracker = VestibularTracker()
    for _ in range(10):
        mujoco.mj_step(model, data)
        c_val, m_val = tracker.step(model.opt.timestep, *sensing.head_motion())
    assert float(c_val) > 0.0
    assert float(m_val) > 0.0
