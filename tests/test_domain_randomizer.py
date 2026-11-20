from mujoco import mjx
import jax
import numpy as np
import numpy.random
import jax.numpy as jnp
import pytest
import os
from math import isclose

from test_conf import DummyHumamoidEnv
from loco_mujoco.core.domain_randomizer.default import DefaultRandomizerState
from copy import deepcopy

from test_conf import *

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# set Jax-backend to CPU
jax.config.update("jax_platform_name", "cpu")
print(f"Jax backend device: {jax.default_backend()} \n")

_DOMRAND_ATOL = 1e-6
_DOMRAND_RTOL = 1e-6


# load yaml as dict:
path = parent_dir + "/tests/test_conf/default_dom_rand_conf.yaml"
with open(path, "r") as file:
    import yaml

    default_dom_rand_conf = yaml.load(file, Loader=yaml.FullLoader)

DEFAULTS = {"horizon": 1000, "gamma": 0.99, "n_envs": 1}


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_init_state(backend):
    seed = 0
    key = jax.random.PRNGKey(seed)
    numpy.random.seed(seed)

    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        domain_randomization_type="DefaultRandomizer",
        domain_randomization_params=default_dom_rand_conf,
        **DEFAULTS,
    )

    backend = np if backend == "numpy" else jnp

    default_randomizer_state: DefaultRandomizerState = None
    if backend == np:
        # reset the environment in Mujoco
        obs = mjx_env.reset(key)

        default_randomizer_state = mjx_env._domain_randomizer.init_state(
            mjx_env, key, mjx_env._model, mjx_env._data, backend
        )
    else:
        # reset the environment in Mjx
        state = mjx_env.mjx_reset(key)

        default_randomizer_state = mjx_env._domain_randomizer.init_state(
            mjx_env, key, mjx_env._model, state.data, backend
        )

    np.testing.assert_allclose(
        default_randomizer_state.gravity, backend.array([0.0, 0.0, -9.81])
    )
    np.testing.assert_allclose(
        default_randomizer_state.geom_friction,
        backend.array(mjx_env._model.geom_friction.copy()),
    )
    np.testing.assert_allclose(
        default_randomizer_state.geom_stiffness, backend.zeros(mjx_env._model.ngeom)
    )
    np.testing.assert_allclose(
        default_randomizer_state.geom_damping, backend.zeros(mjx_env._model.ngeom)
    )
    np.testing.assert_allclose(
        default_randomizer_state.com_displacement, backend.array([0.0, 0.0, 0.0])
    )
    np.testing.assert_allclose(
        default_randomizer_state.link_mass_multipliers,
        backend.array([1.0] * (mjx_env._model.nbody - 1)),
    )
    np.testing.assert_allclose(
        default_randomizer_state.joint_friction_loss,
        backend.array([0.0] * (mjx_env._model.nv - 6)),
    )
    np.testing.assert_allclose(
        default_randomizer_state.joint_damping,
        backend.array([0.0] * (mjx_env._model.nv - 6)),
    )
    np.testing.assert_allclose(
        default_randomizer_state.joint_armature,
        backend.array([0.0] * (mjx_env._model.nv - 6)),
    )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_reset(backend):
    seed = 0
    key = jax.random.PRNGKey(seed)

    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        domain_randomization_type="DefaultRandomizer",
        domain_randomization_params=default_dom_rand_conf,
        **DEFAULTS,
    )

    backend = np if backend == "numpy" else jnp

    if backend == np:
        # reset the environment in Mujoco
        obs = mjx_env.reset(key)
        initial_carry = deepcopy(mjx_env._additional_carry)

        _, carry = mjx_env._domain_randomizer.reset(
            mjx_env, mjx_env._model, mjx_env._data, mjx_env._additional_carry, backend
        )
    else:
        # reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        initial_carry = deepcopy(state.additional_carry)

        sys = mjx.put_model(mjx_env._model)

        _, carry = mjx_env._domain_randomizer.reset(
            mjx_env, sys, state.data, state.additional_carry, backend
        )

    assert not np.allclose(
        initial_carry.domain_randomizer_state.gravity,
        carry.domain_randomizer_state.gravity,
        atol=_DOMRAND_ATOL, rtol=_DOMRAND_RTOL,
    ), "Gravity should not match"
    assert not np.allclose(
        initial_carry.domain_randomizer_state.geom_friction,
        carry.domain_randomizer_state.geom_friction,
        atol=_DOMRAND_ATOL, rtol=_DOMRAND_RTOL,
    ), "Geom friction should not match"
    assert not np.allclose(
        initial_carry.domain_randomizer_state.geom_stiffness,
        carry.domain_randomizer_state.geom_stiffness,
        atol=_DOMRAND_ATOL, rtol=_DOMRAND_RTOL,
    ), "Geom stiffness should not match"
    assert not np.allclose(
        initial_carry.domain_randomizer_state.geom_damping,
        carry.domain_randomizer_state.geom_damping,
        atol=_DOMRAND_ATOL, rtol=_DOMRAND_RTOL,
    ), "Geom damping should not match"
    assert not np.allclose(
        initial_carry.domain_randomizer_state.base_mass_to_add,
        carry.domain_randomizer_state.base_mass_to_add,
        atol=_DOMRAND_ATOL, rtol=_DOMRAND_RTOL,
    ), "Base mass to add should not match"
    assert not np.allclose(
        initial_carry.domain_randomizer_state.com_displacement,
        carry.domain_randomizer_state.com_displacement,
        atol=_DOMRAND_ATOL, rtol=_DOMRAND_RTOL,
    ), "COM displacement should not match"
    assert not np.allclose(
        initial_carry.domain_randomizer_state.link_mass_multipliers,
        carry.domain_randomizer_state.link_mass_multipliers,
        atol=_DOMRAND_ATOL, rtol=_DOMRAND_RTOL,
    ), "Link mass multipliers should not match"
    assert not np.allclose(
        initial_carry.domain_randomizer_state.joint_friction_loss,
        carry.domain_randomizer_state.joint_friction_loss,
        atol=_DOMRAND_ATOL, rtol=_DOMRAND_RTOL,
    ), "Joint friction loss should not match"
    assert not np.allclose(
        initial_carry.domain_randomizer_state.joint_damping,
        carry.domain_randomizer_state.joint_damping,
        atol=_DOMRAND_ATOL, rtol=_DOMRAND_RTOL,
    ), "Joint damping should not match"
    assert not np.allclose(
        initial_carry.domain_randomizer_state.joint_armature,
        carry.domain_randomizer_state.joint_armature,
        atol=_DOMRAND_ATOL, rtol=_DOMRAND_RTOL,
    ), "Joint armature should not match"


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_update(backend):
    seed = 0
    key = jax.random.PRNGKey(seed)

    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        domain_randomization_type="DefaultRandomizer",
        domain_randomization_params=default_dom_rand_conf,
        **DEFAULTS,
    )

    backend = np if backend == "numpy" else jnp

    if backend == np:
        # reset the environment in Mujoco
        obs = mjx_env.reset(key)
        initial_model = deepcopy(mjx_env._model)
        model, _, _ = mjx_env._domain_randomizer.update(
            mjx_env, mjx_env._model, mjx_env._data, mjx_env._additional_carry, backend
        )
    else:
        # reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        initial_model = deepcopy(mjx_env._model)

        sys = mjx.put_model(mjx_env._model)

        model, _, _ = mjx_env._domain_randomizer.update(
            mjx_env, sys, state.data, state.additional_carry, backend
        )

    # Ensure that values have changed
    assert not np.allclose(
        model.geom_friction, initial_model.geom_friction, atol=_DOMRAND_ATOL, rtol=_DOMRAND_RTOL
    ), "Geom friction should not match"
    assert not np.allclose(
        model.geom_solref, initial_model.geom_solref, atol=_DOMRAND_ATOL, rtol=_DOMRAND_RTOL
    ), "Geom solref should not match"
    assert not np.allclose(
        model.body_ipos, initial_model.body_ipos, atol=_DOMRAND_ATOL, rtol=_DOMRAND_RTOL
    ), "Body position should be modified"
    assert not np.allclose(
        model.body_mass, initial_model.body_mass, atol=_DOMRAND_ATOL, rtol=_DOMRAND_RTOL
    ), "Body mass should be modified"
    assert not np.allclose(
        model.dof_frictionloss, initial_model.dof_frictionloss, atol=_DOMRAND_ATOL, rtol=_DOMRAND_RTOL
    ), "Joint friction loss should be modified"
    assert not np.allclose(
        model.dof_damping, initial_model.dof_damping, atol=_DOMRAND_ATOL, rtol=_DOMRAND_RTOL
    ), "Joint damping should be modified"
    assert not np.allclose(
        model.dof_armature, initial_model.dof_armature, atol=_DOMRAND_ATOL, rtol=_DOMRAND_RTOL
    ), "Joint armature should be modified"


OBSERVATION_SPACE = [
    {"obs_name": "name_obs3", "type": "BodyVel", "xml_name": "left_thigh"},
    {"obs_name": "name_obs4", "type": "BodyVel", "xml_name": "right_shin"},
    {"obs_name": "name_obs5", "type": "ProjectedGravityVector", "xml_name": "root"},
    {"obs_name": "name_obs6", "type": "JointPos", "xml_name": "left_hip_y"},
    {"obs_name": "name_obs7", "type": "JointVel", "xml_name": "left_hip_y"},
    {"obs_name": "name_obs8", "type": "JointPos", "xml_name": "right_hip_y"},
    {"obs_name": "name_obs9", "type": "JointVel", "xml_name": "right_hip_y"},
    {"obs_name": "name_obs10", "type": "FreeJointVel", "xml_name": "root"},
]


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_update_observation(backend, mock_random):
    seed = 0
    key = jax.random.PRNGKey(seed)

    body_name1 = "left_shin"
    body_name2 = "right_thigh"

    # specify the observation space
    observation_spec = [
        {"obs_name": "name_obs1", "type": "BodyPos", "xml_name": body_name1},
        {"obs_name": "name_obs2", "type": "BodyPos", "xml_name": body_name2},
    ]

    observation_spec.extend(OBSERVATION_SPACE)

    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        observation_spec=observation_spec,
        domain_randomization_type="DefaultRandomizer",
        domain_randomization_params=default_dom_rand_conf,
        **DEFAULTS,
    )

    backend = np if backend == "numpy" else jnp

    if backend == np:
        # reset the environment in Mujoco
        obs = mjx_env.reset(key)
        initial_carry = mjx_env._additional_carry

        obs_updated, carry = mjx_env._domain_randomizer.update_observation(
            mjx_env,
            obs,
            mjx_env._model,
            mjx_env._data,
            mjx_env._additional_carry,
            backend,
        )
    else:
        # reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        obs = state.observation
        initial_carry = state.additional_carry

        obs_updated, carry = mjx_env._domain_randomizer.update_observation(
            mjx_env, obs, mjx_env._model, state.data, state.additional_carry, backend
        )
    print(f"\nobs_updated: {obs_updated}")
    # Expected values differ by backend (mujoco vs mjx physics)
    if backend == np:
        expected_obs_updated = [
            -0.02511085,
            0.08873329,
            0.425,
            0.00726929,
            -0.09849757,
            0.828,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0075, 0.0075, -0.9925, 0.0015, 0.04, 0.0015, 0.04, 0.05, 0.05, 0.05,
            0.01, 0.01, 0.01,
        ]
    else:
        # mjx/jax: BodyPos observations are zero at reset
        expected_obs_updated = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0075, 0.0075, -0.9925, 0.0015, 0.04, 0.0015, 0.04, 0.05, 0.05, 0.05,
            0.01, 0.01, 0.01,
        ]
    np.testing.assert_allclose(
        obs_updated,
        expected_obs_updated,
        atol=_DOMRAND_ATOL, rtol=_DOMRAND_RTOL,
    )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_update_action(backend):
    pass


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_sample_geom_friction(backend, mock_random):
    seed = 0
    key = jax.random.PRNGKey(seed)

    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        domain_randomization_type="DefaultRandomizer",
        domain_randomization_params=default_dom_rand_conf,
        **DEFAULTS,
    )

    backend = np if backend == "numpy" else jnp

    if backend == np:
        # reset the environment in Mujoco
        obs = mjx_env.reset(key)

        geom_friction, _ = mjx_env._domain_randomizer._sample_geom_friction(
            mjx_env._model, mjx_env._additional_carry, backend
        )

    else:
        # reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        geom_friction, _ = mjx_env._domain_randomizer._sample_geom_friction(
            mjx_env._model, state.additional_carry, backend
        )

    print(f"\ngeom_friction: {geom_friction}")
    np.testing.assert_allclose(
        geom_friction,
        [
            [9.2000002e-01, 4.2000003e-03, 9.2000002e-05],
            [9.2000002e-01, 4.2000003e-03, 9.2000002e-05],
            [9.2000002e-01, 4.2000003e-03, 9.2000002e-05],
            [9.2000002e-01, 4.2000003e-03, 9.2000002e-05],
            [9.2000002e-01, 4.2000003e-03, 9.2000002e-05],
            [9.2000002e-01, 4.2000003e-03, 9.2000002e-05],
            [9.2000002e-01, 4.2000003e-03, 9.2000002e-05],
            [9.2000002e-01, 4.2000003e-03, 9.2000002e-05],
            [9.2000002e-01, 4.2000003e-03, 9.2000002e-05],
            [9.2000002e-01, 4.2000003e-03, 9.2000002e-05],
            [9.2000002e-01, 4.2000003e-03, 9.2000002e-05],
            [9.2000002e-01, 4.2000003e-03, 9.2000002e-05],
        ],
        atol=_DOMRAND_ATOL, rtol=_DOMRAND_RTOL,
    )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_sample_geom_damping_and_stiffness(backend, mock_random):
    seed = 0
    key = jax.random.PRNGKey(seed)

    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        domain_randomization_type="DefaultRandomizer",
        domain_randomization_params=default_dom_rand_conf,
        **DEFAULTS,
    )

    backend = np if backend == "numpy" else jnp

    if backend == np:
        # reset the environment in Mujoco
        obs = mjx_env.reset(key)

        sampled_damping, sampled_stiffness, _ = (
            mjx_env._domain_randomizer._sample_geom_damping_and_stiffness(
                mjx_env._model, mjx_env._additional_carry, backend
            )
        )
    else:
        # reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        sampled_damping, sampled_stiffness, _ = (
            mjx_env._domain_randomizer._sample_geom_damping_and_stiffness(
                mjx_env._model, state.additional_carry, backend
            )
        )

    print(
        f"\nsampled_damping: {sampled_damping}\nsampled_stiffness: {sampled_stiffness}"
    )
    np.testing.assert_allclose(
        sampled_damping,
        [76.8, 76.8, 76.8, 76.8, 76.8, 76.8, 76.8, 76.8, 76.8, 76.8, 76.8, 76.8],
        atol=_DOMRAND_ATOL, rtol=_DOMRAND_RTOL,
    )

    np.testing.assert_allclose(
        sampled_stiffness,
        [
            960.0,
            960.0,
            960.0,
            960.0,
            960.0,
            960.0,
            960.0,
            960.0,
            960.0,
            960.0,
            960.0,
            960.0,
        ],
        atol=_DOMRAND_ATOL, rtol=_DOMRAND_RTOL,
    )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_sample_joint_friction_loss(backend, mock_random):
    seed = 0
    key = jax.random.PRNGKey(seed)

    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        domain_randomization_type="DefaultRandomizer",
        domain_randomization_params=default_dom_rand_conf,
        **DEFAULTS,
    )

    backend = np if backend == "numpy" else jnp

    if backend == np:
        # reset the environment in Mujoco
        obs = mjx_env.reset(key)

        sampled_friction_loss, _ = (
            mjx_env._domain_randomizer._sample_joint_friction_loss(
                mjx_env._model, mjx_env._additional_carry, backend
            )
        )

    else:
        # reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        sampled_friction_loss, _ = (
            mjx_env._domain_randomizer._sample_joint_friction_loss(
                mjx_env._model, state.additional_carry, backend
            )
        )

    print(f"\nsampled_friction_loss: {sampled_friction_loss}")
    np.testing.assert_allclose(
        sampled_friction_loss,
        [0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06],
        atol=_DOMRAND_ATOL, rtol=_DOMRAND_RTOL,
    )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_sample_joint_damping(backend, mock_random):
    seed = 0
    key = jax.random.PRNGKey(seed)

    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        domain_randomization_type="DefaultRandomizer",
        domain_randomization_params=default_dom_rand_conf,
        **DEFAULTS,
    )

    backend = np if backend == "numpy" else jnp

    if backend == np:
        # reset the environment in Mujoco
        obs = mjx_env.reset(key)

        sampled_damping, _ = mjx_env._domain_randomizer._sample_joint_damping(
            mjx_env._model, mjx_env._additional_carry, backend
        )

    else:
        # reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        sampled_damping, _ = mjx_env._domain_randomizer._sample_joint_damping(
            mjx_env._model, state.additional_carry, backend
        )

    print(f"sampled_damping: {sampled_damping}")
    np.testing.assert_allclose(
        sampled_damping,
        [0.66, 0.66, 0.66, 0.66, 0.66, 0.66, 0.66, 0.66, 0.66, 0.66, 0.66],
        atol=_DOMRAND_ATOL, rtol=_DOMRAND_RTOL,
    )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_sample_joint_armature(backend, mock_random):
    seed = 0
    key = jax.random.PRNGKey(seed)

    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        domain_randomization_type="DefaultRandomizer",
        domain_randomization_params=default_dom_rand_conf,
        **DEFAULTS,
    )

    backend = np if backend == "numpy" else jnp

    if backend == np:
        # reset the environment in Mujoco
        obs = mjx_env.reset(key)

        sampled_armature, _ = mjx_env._domain_randomizer._sample_joint_armature(
            mjx_env._model, mjx_env._additional_carry, backend
        )

    else:
        # reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        sampled_armature, _ = mjx_env._domain_randomizer._sample_joint_armature(
            mjx_env._model, state.additional_carry, backend
        )

    print(f"\nsampled_armature: {sampled_armature}")
    np.testing.assert_allclose(
        sampled_armature,
        [0.092, 0.092, 0.092, 0.092, 0.092, 0.092, 0.092, 0.092, 0.092, 0.092, 0.092],
        atol=_DOMRAND_ATOL, rtol=_DOMRAND_RTOL,
    )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_sample_gravity(backend, mock_random):
    seed = 0
    key = jax.random.PRNGKey(seed)

    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        domain_randomization_type="DefaultRandomizer",
        domain_randomization_params=default_dom_rand_conf,
        **DEFAULTS,
    )

    backend = np if backend == "numpy" else jnp

    if backend == np:
        # reset the environment in Mujoco
        obs = mjx_env.reset(key)

        sampled_gravity, _ = mjx_env._domain_randomizer._sample_gravity(
            mjx_env._model, mjx_env._additional_carry, backend
        )

    else:
        # reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        sampled_gravity, _ = mjx_env._domain_randomizer._sample_gravity(
            mjx_env._model, state.additional_carry, backend
        )

    print(f"\nsampled_gravity: {sampled_gravity}")
    np.testing.assert_allclose(
        sampled_gravity,
        [0.0, 0.0, -9.690001],
        atol=_DOMRAND_ATOL, rtol=_DOMRAND_RTOL,
    )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_sample_base_mass(backend, mock_random):
    seed = 0
    key = jax.random.PRNGKey(seed)

    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        domain_randomization_type="DefaultRandomizer",
        domain_randomization_params=default_dom_rand_conf,
        **DEFAULTS,
    )

    backend = np if backend == "numpy" else jnp

    if backend == np:
        # reset the environment in Mujoco
        obs = mjx_env.reset(key)

        sampled_base_mass, _ = mjx_env._domain_randomizer._sample_base_mass(
            mjx_env._model, mjx_env._additional_carry, backend
        )

        print(f"\nsampled_base_mass: {sampled_base_mass}")
        assert isclose(sampled_base_mass, -0.8)

    else:
        # reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        sampled_base_mass, _ = mjx_env._domain_randomizer._sample_base_mass(
            mjx_env._model, state.additional_carry, backend
        )
        print(f"\nsampled_base_mass: {sampled_base_mass}")
        assert jnp.isclose(sampled_base_mass, -0.7999999523162842)


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_sample_com_displacement(backend, mock_random):
    seed = 0
    key = jax.random.PRNGKey(seed)

    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        domain_randomization_type="DefaultRandomizer",
        domain_randomization_params=default_dom_rand_conf,
        **DEFAULTS,
    )

    backend = np if backend == "numpy" else jnp

    if backend == np:
        # reset the environment in Mujoco
        obs = mjx_env.reset(key)

        sampled_com_displacement, _ = (
            mjx_env._domain_randomizer._sample_com_displacement(
                mjx_env._model, mjx_env._additional_carry, backend
            )
        )

    else:
        # reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        sampled_com_displacement, _ = (
            mjx_env._domain_randomizer._sample_com_displacement(
                mjx_env._model, state.additional_carry, backend
            )
        )

    print(f"\nsampled_com_displacement: {sampled_com_displacement}")
    np.testing.assert_allclose(
        sampled_com_displacement,
        [-0.06, -0.06, -0.06],
        atol=_DOMRAND_ATOL, rtol=_DOMRAND_RTOL,
    )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_sample_link_mass_multipliers(backend, mock_random):
    seed = 0
    key = jax.random.PRNGKey(seed)

    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        domain_randomization_type="DefaultRandomizer",
        domain_randomization_params=default_dom_rand_conf,
        **DEFAULTS,
    )

    backend = np if backend == "numpy" else jnp

    if backend == np:
        # reset the environment in Mujoco
        obs = mjx_env.reset(key)

        mass_multipliers, _ = mjx_env._domain_randomizer._sample_link_mass_multipliers(
            mjx_env._model, mjx_env._additional_carry, backend
        )

    else:
        # reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        mass_multipliers, _ = mjx_env._domain_randomizer._sample_link_mass_multipliers(
            mjx_env._model, state.additional_carry, backend
        )

    print(f"\nmass_multipliers: {mass_multipliers}")
    np.testing.assert_allclose(
        mass_multipliers,
        [0.92, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92],
        atol=_DOMRAND_ATOL, rtol=_DOMRAND_RTOL,
    )


control_params_test = {
    "p_gain": [
        -300,  # back_bkz_actuator -> torso
        -200,  # hip_flexion_r_actuator -> hip_pitch
        -200,  # hip_adduction_r_actuator -> hip_roll
        -200,  # hip_rotation_r_actuator -> hip_yaw
        -300,  # knee_angle_r_actuator -> knee
        -40,  # ankle_angle_r_actuator -> ankle
        -200,  # hip_flexion_l_actuator -> hip_pitch
        -200,  # hip_adduction_l_actuator -> hip_roll
        -200,  # hip_rotation_l_actuator -> hip_yaw
        -300,  # knee_angle_l_actuator -> knee
        -40,  # ankle_angle_l_actuator -> ankle
    ],
    "d_gain": [
        -6,  # back_bkz_actuator -> torso
        -5,  # hip_flexion_r_actuator -> hip_pitch
        -5,  # hip_adduction_r_actuator -> hip_roll
        -5,  # hip_rotation_r_actuator -> hip_yaw
        -6,  # knee_angle_r_actuator -> knee
        -2,  # ankle_angle_r_actuator -> ankle
        -5,  # hip_flexion_l_actuator -> hip_pitch
        -5,  # hip_adduction_l_actuator -> hip_roll
        -5,  # hip_rotation_l_actuator -> hip_yaw
        -6,  # knee_angle_l_actuator -> knee
        -2,  # ankle_angle_l_actuator -> ankle
    ],
}


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_sample_p_gains_noise(backend, mock_random):
    seed = 0
    key = jax.random.PRNGKey(seed)

    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        domain_randomization_type="DefaultRandomizer",
        domain_randomization_params=default_dom_rand_conf,
        control_type="PDControl",
        control_params=control_params_test,
        **DEFAULTS,
    )

    backend = np if backend == "numpy" else jnp

    if backend == np:
        # reset the environment in Mujoco
        obs = mjx_env.reset(key)

        p_noise, _ = mjx_env._domain_randomizer._sample_p_gains_noise(
            mjx_env, mjx_env._model, mjx_env._additional_carry, backend
        )

    else:
        # reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        p_noise, _ = mjx_env._domain_randomizer._sample_p_gains_noise(
            mjx_env, mjx_env._model, state.additional_carry, backend
        )

    print(f"\np_noise: {p_noise}")
    np.testing.assert_allclose(
        p_noise,
        [12.0, 8.0, 8.0, 8.0, 12.0, 1.6, 8.0, 8.0, 8.0, 12.0, 1.6],
        atol=_DOMRAND_ATOL, rtol=_DOMRAND_RTOL,
    )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_sample_d_gains_noise(backend, mock_random):
    seed = 0
    key = jax.random.PRNGKey(seed)

    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        domain_randomization_type="DefaultRandomizer",
        domain_randomization_params=default_dom_rand_conf,
        control_type="PDControl",
        control_params=control_params_test,
        **DEFAULTS,
    )

    backend = np if backend == "numpy" else jnp

    if backend == np:
        # reset the environment in Mujoco
        obs = mjx_env.reset(key)

        d_noise, _ = mjx_env._domain_randomizer._sample_d_gains_noise(
            mjx_env, mjx_env._model, mjx_env._additional_carry, backend
        )

    else:
        # reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        d_noise, _ = mjx_env._domain_randomizer._sample_d_gains_noise(
            mjx_env, mjx_env._model, state.additional_carry, backend
        )

    print(f"\nd_noise: {d_noise}")
    np.testing.assert_allclose(
        d_noise,
        [
            0.24000001,
            0.2,
            0.2,
            0.2,
            0.24000001,
            0.08000001,
            0.2,
            0.2,
            0.2,
            0.24000001,
            0.08000001,
        ],
        atol=_DOMRAND_ATOL, rtol=_DOMRAND_RTOL,
    )


# ---------------------------------------------------------------------------
# Partial configurations: switches that are not given are off.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_partial_config(backend):
    partial_conf = {"randomize_gravity": True, "gravity_range": [5.0, 6.0],
                    "add_joint_pos_noise": True, "joint_pos_noise_scale": 0.01}
    env = DummyHumamoidEnv(enable_mjx=backend == "jax",
                           domain_randomization_type="DefaultRandomizer",
                           domain_randomization_params=partial_conf,
                           **DEFAULTS)
    rand_conf = env._domain_randomizer.rand_conf
    for switch, (value_key, _) in env._domain_randomizer.SWITCHES.items():
        assert switch in rand_conf and value_key in rand_conf
        assert rand_conf[switch] == (switch in partial_conf)

    key = jax.random.PRNGKey(0)
    if backend == "jax":
        state = env.mjx_reset(key)
        state = env.mjx_step(state, jnp.zeros(env.info.action_space.shape[0]))
        dr_state = state.additional_carry.domain_randomizer_state
        model = env._model
    else:
        numpy.random.seed(0)
        env.reset(key)
        env.step(np.zeros(env.info.action_space.shape[0]))
        dr_state = env._additional_carry.domain_randomizer_state
        model = env._model

    # gravity is randomized within the given range ...
    assert 5.0 <= -float(dr_state.gravity[2]) <= 6.0
    # ... and the switched-off parameters keep their neutral values
    assert np.allclose(np.asarray(dr_state.link_mass_multipliers), 1.0)
    assert np.allclose(np.asarray(dr_state.base_mass_to_add), 0.0)
    assert np.allclose(np.asarray(dr_state.com_displacement), 0.0)


def test_empty_config_is_valid():
    env = DummyHumamoidEnv(enable_mjx=False,
                           domain_randomization_type="DefaultRandomizer",
                           domain_randomization_params={},
                           **DEFAULTS)
    assert not any(env._domain_randomizer.rand_conf[switch] for switch in env._domain_randomizer.SWITCHES)
    env.reset(jax.random.PRNGKey(0))
    env.step(np.zeros(env.info.action_space.shape[0]))


def test_enabled_switch_without_range_raises():
    with pytest.raises(ValueError, match="gravity_range"):
        DummyHumamoidEnv(enable_mjx=False,
                         domain_randomization_type="DefaultRandomizer",
                         domain_randomization_params={"randomize_gravity": True},
                         **DEFAULTS)
