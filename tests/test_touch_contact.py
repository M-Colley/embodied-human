import math
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from embodied_human.touch_contact import ContactMapTouch, SensorMap


def two_spheres():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(
        """<mujoco><option gravity="0 0 0"/>
        <worldbody>
          <body name="anchor" euler="0 90 0">
            <geom name="fixed" type="sphere" size=".1" friction="1 .01 .01"/>
          </body>
          <body name="moving" pos=".18 0 0" euler="0 0 90">
            <freejoint/>
            <geom name="mobile" type="sphere" size=".1" mass="1" friction="1 .01 .01"/>
          </body>
        </worldbody></mujoco>"""
    )
    data = mujoco.MjData(model)
    data.qvel[1] = 0.4
    mujoco.mj_forward(model, data)
    assert data.ncon == 1
    return mujoco, model, data


def test_contact_module_import_and_construction_without_mujoco():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['mujoco'] = None; "
            "from embodied_human.touch_contact import ContactMapTouch; "
            "t = ContactMapTouch(); "
            "assert t.threshold(125, 'fingertip') > 0; "
            "assert 'mimoEnv' not in sys.modules; "
            "assert 'trimesh' not in sys.modules; "
            "\ntry: t.bind(None, None)\n"
            "except ImportError as e: assert 'pip install' in str(e)\n"
            "else: raise AssertionError('missing dependency was not reported')",
        ],
        check=True,
    )


@pytest.mark.parametrize("response", ["nearest", "spread_linear"])
def test_real_contacts_preserve_frames_newtons_and_each_body_force(response):
    _, model, data = two_spheres()
    maps = {}
    contact = data.contact[0]
    for name, sign in (("anchor", 1), ("moving", -1)):
        body_id = model.body(name).id
        rotation = data.xmat[body_id].reshape(3, 3)
        point = rotation.T @ (contact.pos - data.xpos[body_id])
        normal = rotation.T @ (sign * contact.frame[:3])
        maps[name] = SensorMap(
            np.array([point + [0, 0.01, 0], point - [0, 0.01, 0], point]),
            np.array([normal, normal, -normal]),
        )
    touch = ContactMapTouch(response=response, spread_radius=0.05)
    before = data.qpos.copy(), data.qvel.copy(), model.geom_pos.copy()
    touch.bind(model, data, sensors=maps)
    reading = touch.sense()
    assert len(reading.contacts) == 2
    a, b = sorted(reading.contacts, key=lambda c: c.body_id)
    np.testing.assert_allclose(a.world_force, -b.world_force)
    assert b.world_force[0] > 0
    assert b.world_force[1] < 0  # friction opposes the moving sphere
    for contact in reading.contacts:
        name = model.body(contact.body_id).name
        rotation = data.xmat[contact.body_id].reshape(3, 3)
        np.testing.assert_allclose(rotation @ contact.force, contact.world_force)
        np.testing.assert_allclose(
            rotation @ contact.position + data.xpos[contact.body_id],
            contact.world_position,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            reading.sensor_forces[name].sum(axis=0), contact.force, atol=1e-12
        )
        np.testing.assert_equal(reading.sensor_forces[name][-1], 0)
    np.testing.assert_equal(data.qpos, before[0])
    np.testing.assert_equal(data.qvel, before[1])
    np.testing.assert_equal(model.geom_pos, before[2])
    assert reading.fallback_count == 0
    reading.sensor_forces["moving"][:] = 0
    assert np.linalg.norm(touch.sense().sensor_forces["moving"]) > 0


def test_linear_distribution_conserves_pure_tangential_force():
    # MIMo's normalization by force[0] can erase these components. Normalizing
    # scalar spatial weights must conserve a force with an exactly zero x value.
    mapping = SensorMap(np.array([[0, 0, 0], [0, 0.02, 0]]))
    touch = ContactMapTouch(spread_radius=0.04)
    allocated, fallback = touch._allocate(
        mapping, np.array([0, 0.01, 0]), np.array([0, 0, 1]), np.array([0, 4, 2])
    )
    np.testing.assert_allclose(allocated, [[0, 2, 1], [0, 2, 1]])
    assert not fallback


def test_sparse_map_falls_back_to_nearest_without_force_loss():
    _, model, data = two_spheres()
    touch = ContactMapTouch(spread_radius=0.001)
    touch.bind(model, data, sensors={"anchor": [[2, 2, 2]], "moving": [[2, 2, 2]]})
    reading = touch.sense()
    assert reading.fallback_count == 2
    for contact in reading.contacts:
        key = model.body(contact.body_id).name
        np.testing.assert_allclose(reading.sensor_forces[key][0], contact.force)


def test_incompatible_sensor_normals_report_fallback_and_keep_force():
    mapping = SensorMap(np.array([[0, 0, 0]]), np.array([[0, 0, -1]]))
    force = np.array([1, 2, 3])
    allocated, fallback = ContactMapTouch()._allocate(
        mapping, np.zeros(3), np.array([0, 0, 1]), force
    )
    assert fallback
    np.testing.assert_equal(allocated[0], force)


@pytest.mark.parametrize(
    "kind,size",
    [
        ("box", ".06 .04 .03"),
        ("sphere", ".06"),
        ("ellipsoid", ".06 .04 .03"),
        ("capsule", ".03 .06"),
        ("cylinder", ".03 .06"),
    ],
)
def test_mimo_primitive_maps_follow_existing_geom_pose_and_surface(kind, size):
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(
        f"""<mujoco><worldbody><body name="skin"><freejoint/>
        <geom type="{kind}" size="{size}" pos=".03 .02 .01" euler="20 40 10"/>
        </body></worldbody></mujoco>"""
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    touch = ContactMapTouch(sensor_spacing=0.03)
    touch.bind(model, data)
    reading = touch.sense()
    rotation_flat = np.empty(9)
    mujoco.mju_quat2Mat(rotation_flat, model.geom_quat[0])
    points = (reading.sensor_positions["skin"] - model.geom_pos[0]) @ (
        rotation_flat.reshape(3, 3)
    )
    sizes = model.geom_size[0]
    if kind == "box":
        on_surface = np.max(np.abs(points) / sizes, axis=1)
    elif kind == "ellipsoid":
        on_surface = np.sum((points / sizes) ** 2, axis=1)
    elif kind == "sphere":
        on_surface = np.linalg.norm(points, axis=1) / sizes[0]
    elif kind == "capsule":
        offset = np.maximum(np.abs(points[:, 2]) - sizes[1], 0)
        on_surface = np.sqrt(np.sum(points[:, :2] ** 2, axis=1) + offset**2) / sizes[0]
    else:
        on_surface = np.maximum(
            np.linalg.norm(points[:, :2], axis=1) / sizes[0],
            np.abs(points[:, 2]) / sizes[1],
        )
    np.testing.assert_allclose(on_surface, 1)
    assert reading.contacts == ()
    np.testing.assert_equal(reading.sensor_forces["skin"], 0)


@pytest.mark.parametrize("kind", ["mjGEOM_CAPSULE", "mjGEOM_CYLINDER"])
def test_primitive_dispatch_handles_asymmetric_native_enum_equality(kind):
    # Some MuJoCo bindings accept Python ints in enum.__eq__ but return
    # False for NumPy integers. Tuple membership calls the enum's equality
    # first, unlike the NumPy scalar comparisons in the other branches.
    class IntegerEnum:
        def __init__(self, value):
            self.value = value

        def __int__(self):
            return self.value

        def __eq__(self, other):
            if isinstance(other, IntegerEnum):
                return self.value == other.value
            return type(other) is int and self.value == other

    enums = SimpleNamespace(**{
        name: IntegerEnum(value) for name, value in (
            ("mjGEOM_SPHERE", 2), ("mjGEOM_CAPSULE", 3),
            ("mjGEOM_ELLIPSOID", 4), ("mjGEOM_CYLINDER", 5),
            ("mjGEOM_BOX", 6), ("mjGEOM_MESH", 7),
        )
    })
    native = getattr(enums, kind)
    scalar = np.int32(int(native))
    assert scalar == native
    assert not native == scalar
    assert scalar not in (enums.mjGEOM_CAPSULE, enums.mjGEOM_CYLINDER)
    model = SimpleNamespace(
        geom_type=np.array([scalar]), geom_size=np.array([[0.03, 0.06, 0.0]]),
    )
    points, normals = ContactMapTouch()._sample_geom(
        model, SimpleNamespace(mjtGeom=enums), 0,
    )
    expected_half_height = 0.09 if kind == "mjGEOM_CAPSULE" else 0.06
    assert np.abs(points[:, 2]).max() == pytest.approx(expected_half_height)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0)


def test_existing_mesh_vertices_and_numeric_body_ids_are_supported():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(
        """<mujoco><asset><mesh name="tetra" vertex="0 0 0 .1 0 0 0 .1 0 0 0 .1"
        face="0 2 1 0 1 3 0 3 2 1 2 3"/></asset>
        <worldbody><body><freejoint/><geom type="mesh" mesh="tetra"/>
        </body></worldbody></mujoco>"""
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    touch = ContactMapTouch()
    touch.bind(model, data, bodies=[1])
    assert touch.sense().sensor_positions[1].shape == (4, 3)


@pytest.mark.parametrize(
    "positions,normals",
    [
        ([], None),
        ([[0, 1]], None),
        ([[0, math.nan, 0]], None),
        ([[0, 0, 0]], [[0, 0, 0]]),
        ([[0, 0, 0]], [[0, 1]]),
    ],
)
def test_invalid_sensor_maps_fail(positions, normals):
    with pytest.raises(ValueError):
        SensorMap(np.asarray(positions), normals)


def test_binding_errors_and_allocation_limits_are_explicit():
    mujoco, model, data = two_spheres()
    touch = ContactMapTouch()
    with pytest.raises(RuntimeError, match="bind"):
        touch.sense()
    with pytest.raises(ValueError, match="belong"):
        touch.bind(model, mujoco.MjData(mujoco.MjModel.from_xml_string("<mujoco/>")))
    with pytest.raises(ValueError, match="unknown body"):
        touch.bind(model, data, bodies=[-1])
    with pytest.raises(ValueError, match="one sensor map"):
        touch.bind(model, data, sensors={1: [[0, 0, 0]], "anchor": [[0, 0, 0]]})
    with pytest.raises(ValueError, match="at least one"):
        touch.bind(model, data, sensors={})
    with pytest.raises(ValueError, match="max_sensors"):
        ContactMapTouch(sensor_spacing=0.00001).bind(model, data)
    touch.bind(model, data)
    touch.reset(np.random.default_rng(3))
    assert len(touch.sense().contacts) == 2


def test_existing_myosuite_fullbody_can_be_sensed_without_rebuilding_it():
    pytest.importorskip("myosuite.core.model_builder")
    pytest.importorskip("myo_sim")
    module = pytest.importorskip("myosuite_carla")
    driver = module.build_fullbody_driver()
    touch = ContactMapTouch()
    geometry_before = driver.model.geom_pos.copy()
    touch.bind(driver.model, driver.data)
    for _ in range(30):
        driver.advance(module.DriverIntent(right_foot=0.03))
    positions_before = driver.data.qpos.copy()
    reading = touch.sense()
    assert len(reading.contacts) > 0
    assert len(reading.sensor_positions) > 20
    assert all(np.isfinite(c.force).all() for c in reading.contacts)
    for key, forces in reading.sensor_forces.items():
        body_id = driver.model.body(key).id
        total = sum(
            (c.force for c in reading.contacts if c.body_id == body_id), np.zeros(3)
        )
        np.testing.assert_allclose(forces.sum(axis=0), total, atol=1e-10)
    np.testing.assert_equal(driver.model.geom_pos, geometry_before)
    np.testing.assert_equal(driver.data.qpos, positions_before)
