"""MetricsHandler: the validation measures are evaluated on the selected joints, bodies and sites and on all
components of every quantity."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from metrx import DistanceMeasures
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from loco_mujoco.core.utils.math import calc_site_velocities, calculate_relative_site_quatities
from loco_mujoco.utils.metrics import MetricsHandler, SUPPORTED_MEASURES

from test_conf import *

jax.config.update("jax_platform_name", "cpu")

N_STEPS, N_ENVS = 6, 3      # length and number of the validation rollouts
SHIFT = 40                  # the rollout is the reference motion, this many samples later

# the selections are neither sorted nor complete, and the first site (the main site) is not site 0
REL_JOINTS = ["left_knee", "abdomen_y", "right_hip_z"]
REL_BODIES = ["left_foot", "torso", "right_shin"]
REL_SITES = ["pelvis_site", "left_foot_site", "torso_site", "right_foot_site"]

_ATOL = 1e-5


@pytest.fixture
def env(falling_trajectory):
    """Environment with a loaded reference motion in which every body moves."""
    env = DummyHumamoidEnv(enable_mjx=False, **DEFAULTS)
    env.process_trajectory(falling_trajectory)
    return env


def make_handler(env, quantities, measures=("EuclideanDistance",), **selection):
    config = OmegaConf.create(dict(experiment=dict(
        num_envs=N_ENVS, validation=dict(quantities=list(quantities), measures=list(measures), **selection))))
    return MetricsHandler(config, env)


def traj_indices():
    """Trajectory samples that the reference of every rollout step points to, shape (N_STEPS, N_ENVS)."""
    return np.arange(N_STEPS)[:, None] + 10 * np.arange(N_ENVS)[None, :]


def make_env_states(env, rollout_data=None):
    """
    States of a validation rollout as the algorithms pass them to the handler, with the leading dimensions
    (N_STEPS, N_ENVS). By default, the simulation is at the reference motion ``SHIFT`` samples later.
    """
    ref_ind = traj_indices()
    if rollout_data is None:
        rollout_data = jax.tree.map(lambda x: x[ref_ind + SHIFT], env._traj.data)
    # the simulation stores the site rotations as matrices, the trajectory stores them flattened
    rollout_data = rollout_data.replace(site_xmat=rollout_data.site_xmat.reshape(N_STEPS, N_ENVS, -1, 3, 3))
    ones = jnp.ones((N_STEPS, N_ENVS))
    metrics = SimpleNamespace(done=ones.astype(bool), returned_episode_returns=ones,
                              returned_episode_lengths=ones, timestep=ones)
    traj_state = SimpleNamespace(traj_no=jnp.zeros((N_STEPS, N_ENVS), dtype=int), subtraj_step_no=jnp.array(ref_ind))
    return SimpleNamespace(data=rollout_data, metrics=metrics,
                           additional_carry=SimpleNamespace(traj_state=traj_state))


def single_states(env, indices):
    """NumPy states of the reference motion at ``indices``, one by one in the layout of ``MjData``."""
    traj_data = env._traj.data.to_numpy()
    return [[SimpleNamespace(xpos=traj_data.xpos[i], cvel=traj_data.cvel[i], subtree_com=traj_data.subtree_com[i],
                             site_xpos=traj_data.site_xpos[i], site_xmat=traj_data.site_xmat[i])
             for i in row] for row in indices]


def expected_quantity(env, name, indices):
    """
    The elements of a quantity that the configuration selects, computed state by state with NumPy.

    Returns:
        np.ndarray: Array of shape (N_STEPS, N_ENVS, n_elements, dim).
    """
    model = env.get_model()
    traj_data = env._traj.data.to_numpy()
    body_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in REL_BODIES])
    site_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in REL_SITES])
    site_body_ids = model.site_bodyid[site_ids]

    if name == "BodyPosition":
        return traj_data.xpos[indices][:, :, body_ids]
    if name == "BodyVelocity":
        return traj_data.cvel[indices][:, :, body_ids]
    if name == "BodyOrientation":
        xquat = traj_data.xquat[indices][:, :, body_ids]
        rotvec = Rotation.from_quat(xquat.reshape(-1, 4), scalar_first=True).as_rotvec()
        return rotvec.reshape(*xquat.shape[:-1], 3)
    if name == "SitePosition":
        return traj_data.site_xpos[indices][:, :, site_ids]
    if name == "SiteOrientation":
        xmat = traj_data.site_xmat[indices][:, :, site_ids]
        return Rotation.from_matrix(xmat.reshape(-1, 3, 3)).as_rotvec().reshape(*xmat.shape[:-1], 3)

    states = single_states(env, indices)
    if name == "SiteVelocity":
        return np.array([[calc_site_velocities(site_ids, s, site_body_ids, model.body_rootid[site_body_ids], np)
                          for s in row] for row in states])
    # the relative site quantities, with the arguments MimicReward passes
    rel = [[calculate_relative_site_quatities(s, site_ids, site_body_ids, model.body_rootid, np)
            for s in row] for row in states]
    k = ["RelSitePosition", "RelSiteOrientation", "RelSiteVelocity"].index(name)
    return np.array([[r[k] for r in row] for row in rel])


def expected_measure(measure, rollout, reference):
    """Mean over elements and environments of a measure between the time series of rollout and reference."""
    fn = DistanceMeasures.create_instance(measure, mean=True) if measure == "EuclideanDistance" \
        else DistanceMeasures.create_instance(measure)
    n_elements = rollout.shape[2]
    return np.mean([float(fn(jnp.array(rollout[:, e, n]), jnp.array(reference[:, e, n])))
                    for n in range(n_elements) for e in range(N_ENVS)])


FIELD_OF_QUANTITY = dict(BodyPosition="xpos", BodyOrientation="xrotvec", BodyVelocity="cvel",
                         SitePosition="site_xpos", SiteOrientation="site_xrotvec", SiteVelocity="site_xvel",
                         RelSitePosition="site_rpos", RelSiteOrientation="site_rrotvec", RelSiteVelocity="site_rvel")
FIELD_OF_MEASURE = dict(EuclideanDistance="euclidean_distance", DynamicTimeWarping="dynamic_time_warping",
                        DiscreteFrechetDistance="discrete_frechet_distance")


@pytest.mark.parametrize("quantity", list(FIELD_OF_QUANTITY))
def test_body_and_site_quantities_use_the_selected_elements(env, quantity):
    """Every measure of a body or site quantity is the one of the selected bodies or sites in all dimensions."""
    mh = make_handler(env, [quantity], SUPPORTED_MEASURES, rel_body_names=REL_BODIES, rel_site_names=REL_SITES)
    summary = mh(make_env_states(env))

    ref_ind = traj_indices()
    rollout = expected_quantity(env, quantity, ref_ind + SHIFT)
    reference = expected_quantity(env, quantity, ref_ind)
    n_elements = len(REL_BODIES) if quantity.startswith("Body") else len(REL_SITES)
    if quantity.startswith("Rel"):
        n_elements -= 1     # the main site is the frame the other sites are expressed in
    assert rollout.shape[:3] == (N_STEPS, N_ENVS, n_elements)

    for measure in SUPPORTED_MEASURES:
        value = getattr(getattr(summary, FIELD_OF_MEASURE[measure]), FIELD_OF_QUANTITY[quantity])
        expected = expected_measure(measure, rollout, reference)
        assert expected > 1e-3     # the rollout does deviate from the reference
        np.testing.assert_allclose(value, expected, rtol=1e-4, atol=_ATOL, err_msg=f"{quantity}, {measure}")

    # the Euclidean distance once more without metrx
    np.testing.assert_allclose(getattr(summary.euclidean_distance, FIELD_OF_QUANTITY[quantity]),
                               np.mean(np.linalg.norm(rollout - reference, axis=-1)), rtol=1e-4, atol=_ATOL)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_site_position_error_along_every_axis_is_measured(env, axis):
    """A rollout that tracks the reference except for an offset of one selected site reports that offset."""
    model = env.get_model()
    mh = make_handler(env, ["SitePosition"], rel_site_names=REL_SITES)
    reference = jax.tree.map(lambda x: x[traj_indices()], env._traj.data)

    def distance(site_name):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        site_xpos = reference.site_xpos.at[:, :, site_id, axis].add(0.3)
        summary = mh(make_env_states(env, reference.replace(site_xpos=site_xpos)))
        return float(summary.euclidean_distance.site_xpos)

    # averaged over the selected sites
    assert distance("left_foot_site") == pytest.approx(0.3 / len(REL_SITES), abs=_ATOL)
    # a site that is not selected does not contribute
    assert "right_thigh_site" not in REL_SITES
    assert distance("right_thigh_site") == pytest.approx(0.0, abs=_ATOL)


def test_site_velocities_use_the_root_of_the_body_of_each_site(env):
    """The site velocities refer to the subtree of the root body a site is attached to, not to the world body."""
    mh = make_handler(env, ["SiteVelocity", "RelSiteVelocity"], rel_site_names=REL_SITES)
    ref_ind = traj_indices()

    # in a model with a single robot the centre of mass of the world subtree coincides with the one of the robot;
    # move it away, as further bodies with mass would do, to tell the two apart
    rollout_data = jax.tree.map(lambda x: x[ref_ind + SHIFT], env._traj.data)
    rollout_data = rollout_data.replace(subtree_com=rollout_data.subtree_com.at[:, :, 0].add(1.0))
    summary = mh(make_env_states(env, rollout_data))

    for quantity in ("SiteVelocity", "RelSiteVelocity"):
        expected = np.mean(np.linalg.norm(expected_quantity(env, quantity, ref_ind + SHIFT)
                                          - expected_quantity(env, quantity, ref_ind), axis=-1))
        np.testing.assert_allclose(getattr(summary.euclidean_distance, FIELD_OF_QUANTITY[quantity]), expected,
                                   rtol=1e-4, atol=_ATOL, err_msg=quantity)


def joint_ids(model, names):
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
    return [int(model.jnt_qposadr[i]) for i in ids], [int(model.jnt_dofadr[i]) for i in ids]


@pytest.mark.parametrize("option", ["rel_joint_names", "joints_to_ignore"])
def test_joint_quantities_use_the_selected_joints(env, option):
    """The joint quantities cover the joints selected by name, also if the free joint is not among them."""
    model = env.get_model()
    traj_data = env._traj.data.to_numpy()
    ref_ind = traj_indices()

    if option == "rel_joint_names":
        selection = dict(rel_joint_names=REL_JOINTS)
        qpos_ids, qvel_ids = joint_ids(model, REL_JOINTS)
    else:
        all_joints = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
        selection = dict(joints_to_ignore=[n for n in all_joints if n not in REL_JOINTS])
        qpos_ids, qvel_ids = joint_ids(model, [n for n in all_joints if n in REL_JOINTS])

    mh = make_handler(env, ["JointPosition", "JointVelocity"], **selection)
    np.testing.assert_array_equal(mh.rel_qpos_ids, qpos_ids)
    np.testing.assert_array_equal(mh.rel_qvel_ids, qvel_ids)

    summary = mh(make_env_states(env))
    for field, values, ids in (("qpos", traj_data.qpos, qpos_ids), ("qvel", traj_data.qvel, qvel_ids)):
        # every joint is a one-dimensional time series
        expected = np.mean(np.abs(values[ref_ind + SHIFT][..., ids] - values[ref_ind][..., ids]))
        assert expected > 1e-3
        np.testing.assert_allclose(getattr(summary.euclidean_distance, field), expected, rtol=1e-4, atol=_ATOL)


def test_rel_joint_names_with_the_free_joint(env):
    """A free joint selected by name contributes its seven positions and its six velocities."""
    model = env.get_model()
    traj_data = env._traj.data.to_numpy()
    ref_ind = traj_indices()
    knee_qpos_id, knee_qvel_id = joint_ids(model, ["left_knee"])

    mh = make_handler(env, ["JointPosition", "JointVelocity"], rel_joint_names=["left_knee", "root"])
    np.testing.assert_array_equal(mh.rel_qpos_ids, knee_qpos_id + list(range(7)))
    np.testing.assert_array_equal(mh.rel_qvel_ids, knee_qvel_id + list(range(6)))

    # the knee and the root position as they are, the root quaternion as a rotation vector
    def expected_qpos(indices):
        qpos = traj_data.qpos[indices]
        rotvec = Rotation.from_quat(qpos[..., 3:7].reshape(-1, 4), scalar_first=True).as_rotvec()
        return np.concatenate([qpos[..., knee_qpos_id], qpos[..., :3], rotvec.reshape(*qpos.shape[:-1], 3)], axis=-1)

    summary = mh(make_env_states(env))
    np.testing.assert_allclose(summary.euclidean_distance.qpos,
                               np.mean(np.abs(expected_qpos(ref_ind + SHIFT) - expected_qpos(ref_ind))),
                               rtol=1e-4, atol=_ATOL)
    qvel_ids = knee_qvel_id + list(range(6))
    np.testing.assert_allclose(summary.euclidean_distance.qvel,
                               np.mean(np.abs(traj_data.qvel[ref_ind + SHIFT][..., qvel_ids]
                                              - traj_data.qvel[ref_ind][..., qvel_ids])), rtol=1e-4, atol=_ATOL)
