from typing import Any, Union, Tuple
from types import ModuleType

import numpy as np
import jax
import jax.numpy as jnp
from flax import struct
import mujoco
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from loco_mujoco.core.domain_randomizer import DomainRandomizer
from loco_mujoco.core.control_functions import PDControl
from loco_mujoco.core.utils.backend import assert_backend_is_supported


@struct.dataclass
class DefaultRandomizerState:
    """
    Represents the state of the default randomizer.

    """

    gravity: Union[np.ndarray, jax.Array]
    geom_friction: Union[np.ndarray, jax.Array]
    geom_stiffness: Union[np.ndarray, jax.Array]
    geom_damping: Union[np.ndarray, jax.Array]
    base_mass_to_add: float
    com_displacement: Union[np.ndarray, jax.Array]
    link_mass_multipliers: Union[np.ndarray, jax.Array]
    joint_friction_loss: Union[np.ndarray, jax.Array]
    joint_damping: Union[np.ndarray, jax.Array]
    joint_armature: Union[np.ndarray, jax.Array]


class DefaultRandomizer(DomainRandomizer):
    """
    A domain randomizer class that modifies typical simulation parameters.

    """

    def __init__(self, env, **kwargs):

        # store initial values for reset (only needed for numpy backend)
        self._init_gravity = None
        self._init_geom_friction = None
        self._init_geom_solref = None
        self._init_body_ipos = None
        self._init_body_mass = None
        self._init_dof_frictionloss = None
        self._init_dof_damping = None
        self._init_dof_armature = None

        info_props = env._get_all_info_properties()
        root_body_name = info_props["root_body_name"]
        self._root_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, root_body_name)

        self._other_body_masks = np.ones(env.model.nbody, dtype=bool)
        self._other_body_masks[0] = False # exclude worldbody
        self._other_body_masks[self._root_body_id] = False

        # some observations are not allowed to be randomized, filter them out
        self._allowed_to_be_randomized = env.obs_container.get_randomizable_obs_indices()

        super().__init__(env, **kwargs)

    def init_state(self,
                   env: Any,
                   key: Any,
                   model: Union[MjModel, Model],
                   data: Union[MjData, Data],
                   backend: ModuleType,
                   traj=None) -> DefaultRandomizerState:
        """
        Initialize the randomizer state.

        Args:
            env (Any): The environment instance.
            key (Any): Random seed key.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            DefaultRandomizerState: The initialized randomizer state.

        """

        assert_backend_is_supported(backend)
        return DefaultRandomizerState(gravity=backend.array([0.0, 0.0, -9.81]),
                                      geom_friction=backend.array(model.geom_friction.copy()),
                                      geom_stiffness=backend.zeros(model.ngeom),
                                      geom_damping=backend.zeros(model.ngeom),
                                      base_mass_to_add=0.0,
                                      com_displacement=backend.array([0.0, 0.0, 0.0]),
                                      link_mass_multipliers=backend.array([1.0] * (model.nbody-1)), #exclude worldbody
                                      joint_friction_loss=backend.array([0.0] * (model.nv-6)), #exclude freejoint 6 dofs
                                      joint_damping=backend.array([0.0] * (model.nv-6)), #exclude freejoint 6 dofs
                                      joint_armature=backend.array([0.0] * (model.nv-6)), #exclude freejoint 6 dofs
                                      )

    def reset(self,
              env: Any,
              model: Union[MjModel, Model],
              data: Union[MjData, Data],
              carry: Any,
              backend: ModuleType,
              traj=None) -> Tuple[Union[MjData, Data], Any]:
        """
        Reset the randomizer, applying domain randomization.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[MjData, Data], Any]: The updated simulation data and carry.

        """

        assert_backend_is_supported(backend)
        domain_randomizer_state = carry.domain_randomizer_state

        if backend == np and self._init_body_ipos is None:
            # store initial values for reset
            self._init_gravity = model.opt.gravity.copy()
            self._init_geom_solref = model.geom_solref.copy()
            self._init_body_ipos = model.body_ipos.copy()
            self._init_body_mass = model.body_mass.copy()
            self._init_dof_frictionloss = model.dof_frictionloss.copy()
            self._init_dof_damping = model.dof_damping.copy()
            self._init_dof_armature = model.dof_armature.copy()

        # update different randomization parameters
        gravity, carry = self._sample_gravity(model, carry, backend)
        geom_friction, carry = self._sample_geom_friction(model, carry, backend)
        geom_damping, geom_stiffness, carry = self._sample_geom_damping_and_stiffness(model, carry, backend)
        base_mass_to_add, carry = self._sample_base_mass(model, carry, backend)
        com_displacement, carry = self._sample_com_displacement(model, carry, backend)
        link_mass_multipliers, carry = self._sample_link_mass_multipliers(model, carry, backend)
        joint_friction_loss, carry = self._sample_joint_friction_loss(model, carry, backend)
        joint_damping, carry = self._sample_joint_damping(model, carry, backend)
        joint_armature, carry = self._sample_joint_armature(model, carry, backend)

        if isinstance(env._control_func, PDControl):
            control_func_state = carry.control_func_state

            p_noise, carry = self._sample_p_gains_noise(env, model, carry, backend)
            d_noise, carry = self._sample_d_gains_noise(env, model, carry, backend)
            carry = carry.replace(control_func_state=control_func_state.replace(p_gain_noise=p_noise,
                                                                                d_gain_noise=d_noise,
                                                                                pos_offset=backend.zeros_like(env._control_func._nominal_joint_positions),
                                                                                ctrl_mult=backend.ones_like(env._control_func._nominal_joint_positions)))



        carry = carry.replace(domain_randomizer_state=domain_randomizer_state.replace(gravity=gravity,
                                                                                      geom_friction=geom_friction,
                                                                                      geom_stiffness=geom_stiffness,
                                                                                      geom_damping=geom_damping,
                                                                                      base_mass_to_add=base_mass_to_add,
                                                                                      com_displacement=com_displacement,
                                                                                      link_mass_multipliers=link_mass_multipliers,
                                                                                      joint_friction_loss=joint_friction_loss,
                                                                                      joint_damping=joint_damping,
                                                                                      joint_armature=joint_armature,
                                                                                      ))

        return data, carry

    def update(self,
               env: Any,
               model: Union[MjModel, Model],
               data: Union[MjData, Data],
               carry: Any,
               backend: ModuleType,
               traj=None) -> Tuple[Union[MjModel, Model], Union[MjData, Data], Any]:
        """
        Update the randomizer by applying the state changes to the model.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[MjModel, Model], Union[MjData, Data], Any]: The updated simulation model, data, and carry.

        """

        assert_backend_is_supported(backend)

        domrand_state = carry.domain_randomizer_state

        sampled_base_mass_multiplier = domrand_state.link_mass_multipliers[0]
        sampled_other_bodies_mass_multipliers = domrand_state.link_mass_multipliers[1:]

        root_body_id = self._root_body_id
        other_body_masks = self._other_body_masks

        if backend == jnp:
            geom_solref = model.geom_solref.at[:, 0].set(-domrand_state.geom_stiffness)
            geom_solref = geom_solref.at[:, 1].set(-domrand_state.geom_damping)
            body_ipos = model.body_ipos.at[root_body_id].set(model.body_ipos[root_body_id] + domrand_state.com_displacement)
            body_mass = model.body_mass.at[root_body_id].set(model.body_mass[root_body_id] * sampled_base_mass_multiplier)
            body_mass = body_mass.at[root_body_id].set(body_mass[root_body_id] + domrand_state.base_mass_to_add)
            body_mass = body_mass.at[other_body_masks].set(body_mass[other_body_masks] * sampled_other_bodies_mass_multipliers)
            dof_frictionloss = model.dof_frictionloss.at[6:].set(domrand_state.joint_friction_loss)
            dof_damping = model.dof_damping.at[6:].set(domrand_state.joint_damping)
            dof_armature = model.dof_armature.at[6:].set(domrand_state.joint_armature)
            if self.rand_conf["randomize_gravity"]:
                model = self._set_attribute_in_model(model, "opt.gravity", domrand_state.gravity, backend)
        else:
            geom_solref = self._init_geom_solref.copy()
            geom_solref[:, 0] = -domrand_state.geom_stiffness
            geom_solref[:, 1] = -domrand_state.geom_damping
            body_ipos = self._init_body_ipos.copy()
            body_ipos[root_body_id] += domrand_state.com_displacement
            body_mass = self._init_body_mass.copy()
            body_mass[root_body_id] *= sampled_base_mass_multiplier
            body_mass[root_body_id] += domrand_state.base_mass_to_add
            body_mass[other_body_masks] *= sampled_other_bodies_mass_multipliers
            dof_frictionloss = self._init_dof_frictionloss.copy()
            dof_frictionloss[6:] = domrand_state.joint_friction_loss
            dof_damping = self._init_dof_damping.copy()
            dof_damping[6:] = domrand_state.joint_damping
            dof_armature = self._init_dof_armature.copy()
            dof_armature[6:] = domrand_state.joint_armature
            if self.rand_conf["randomize_gravity"]:
                model.opt.gravity = domrand_state.gravity

        if (self.rand_conf["randomize_geom_friction_tangential"] or self.rand_conf["randomize_geom_friction_torsional"]
                or self.rand_conf["randomize_geom_friction_rolling"]):
            model = self._set_attribute_in_model(model, "geom_friction", domrand_state.geom_friction, backend)
        if self.rand_conf["randomize_geom_damping"] or self.rand_conf["randomize_geom_stiffness"]:
            model = self._set_attribute_in_model(model, "geom_solref", geom_solref, backend)
        if self.rand_conf["randomize_com_displacement"]:
            model = self._set_attribute_in_model(model, "body_ipos", body_ipos, backend)
        if self.rand_conf["randomize_link_mass"] or self.rand_conf["randomize_base_mass"]:
            model = self._set_attribute_in_model(model, "body_mass", body_mass, backend)
        if self.rand_conf["randomize_joint_friction_loss"]:
            model = self._set_attribute_in_model(model, "dof_frictionloss", dof_frictionloss, backend)
        if self.rand_conf["randomize_joint_damping"]:
            model = self._set_attribute_in_model(model, "dof_damping", dof_damping ,backend)
        if self.rand_conf["randomize_joint_armature"]:
            model = self._set_attribute_in_model(model, "dof_armature", dof_armature, backend)

        return model, data, carry

    def update_observation(self,
                           env: Any,
                           obs: Union[np.ndarray, jnp.ndarray],
                           model: Union[MjModel, Model],
                           data: Union[MjData, Data],
                           carry: Any,
                           backend: ModuleType,
                           traj=None) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Update the observation with randomization effects.

        Args:
            env (Any): The environment instance.
            obs (Union[np.ndarray, jnp.ndarray]): The observation to be updated.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: The updated observation and carry.

        """

        assert_backend_is_supported(backend)

        # get indices of all the observation components
        total_len_noise_vec = 0
        ind_of_all_joint_pos = env._obs_indices.JointPos
        ind_of_all_joint_pos_noise = np.arange(total_len_noise_vec, total_len_noise_vec+len(ind_of_all_joint_pos))
        total_len_noise_vec += len(ind_of_all_joint_pos)
        ind_of_all_joint_vel = env._obs_indices.JointVel
        ind_of_all_joint_vel_noise = np.arange(total_len_noise_vec, total_len_noise_vec+len(ind_of_all_joint_vel))
        total_len_noise_vec += len(ind_of_all_joint_vel)
        ind_of_gravity_vec = env._obs_indices.ProjectedGravityVector
        ind_of_gravity_vec_noise = np.arange(total_len_noise_vec, total_len_noise_vec+len(ind_of_gravity_vec))
        total_len_noise_vec += len(ind_of_gravity_vec)
        ind_of_lin_vel = env._obs_indices.FreeJointVel[:3]
        ind_of_lin_vel_noise = np.arange(total_len_noise_vec, total_len_noise_vec+len(ind_of_lin_vel))
        total_len_noise_vec += len(ind_of_lin_vel)
        ind_of_ang_vel = env._obs_indices.FreeJointVel[3:]
        ind_of_ang_vel_noise = np.arange(total_len_noise_vec, total_len_noise_vec+len(ind_of_ang_vel))
        total_len_noise_vec += len(ind_of_ang_vel)

        # get randomization parameters
        joint_pos_noise_scale = self.rand_conf["joint_pos_noise_scale"]
        joint_vel_noise_scale = self.rand_conf["joint_vel_noise_scale"]
        gravity_noise_scale = self.rand_conf["gravity_noise_scale"]
        lin_vel_noise_scale = self.rand_conf["lin_vel_noise_scale"]
        ang_vel_noise_scale = self.rand_conf["ang_vel_noise_scale"]

        if backend == jnp:
            key = carry.key
            key, _k = jax.random.split(key)
            noise = jax.random.normal(_k, shape=(total_len_noise_vec,))

            randomized_obs = obs.copy()

            # Add noise to joint positions
            if self.rand_conf["add_joint_pos_noise"]:
                randomized_obs = randomized_obs.at[ind_of_all_joint_pos].add(noise[ind_of_all_joint_pos_noise] * joint_pos_noise_scale)
            
            # Add noise to joint velocities
            if self.rand_conf["add_joint_vel_noise"]:
                randomized_obs = randomized_obs.at[ind_of_all_joint_vel].add(noise[ind_of_all_joint_vel_noise] * joint_vel_noise_scale)
            
            # Add noise to gravity vector
            if self.rand_conf["add_gravity_noise"]:
                randomized_obs = randomized_obs.at[ind_of_gravity_vec].add(noise[ind_of_gravity_vec_noise] * gravity_noise_scale)
            
            # Add noise to linear velocities
            if self.rand_conf["add_free_joint_lin_vel_noise"]:
                randomized_obs = randomized_obs.at[ind_of_lin_vel].add(noise[ind_of_lin_vel_noise] * lin_vel_noise_scale)

            # Add noise to angular velocities
            if self.rand_conf["add_free_joint_ang_vel_noise"]:
                randomized_obs = randomized_obs.at[ind_of_ang_vel].add(noise[ind_of_ang_vel_noise] * ang_vel_noise_scale)

            obs = obs.at[self._allowed_to_be_randomized].set(randomized_obs[self._allowed_to_be_randomized])
            carry = carry.replace(key=key)

        else:
            noise = np.random.normal(size=(total_len_noise_vec,))

            randomized_obs = obs.copy()

             # Add noise to joint positions
            if self.rand_conf["add_joint_pos_noise"]:
                randomized_obs[ind_of_all_joint_pos] += noise[ind_of_all_joint_pos_noise] * joint_pos_noise_scale

            # Add noise to joint velocities
            if self.rand_conf["add_joint_vel_noise"]:
                randomized_obs[ind_of_all_joint_vel] += noise[ind_of_all_joint_vel_noise] * joint_vel_noise_scale

            # Add noise to gravity vector
            if self.rand_conf["add_gravity_noise"]:
                randomized_obs[ind_of_gravity_vec] += noise[ind_of_gravity_vec_noise] * gravity_noise_scale

            # Add noise to linear velocities
            if self.rand_conf["add_free_joint_lin_vel_noise"]:
                randomized_obs[ind_of_lin_vel] += noise[ind_of_lin_vel_noise] * lin_vel_noise_scale

            # Add noise to angular velocities
            if self.rand_conf["add_free_joint_ang_vel_noise"]:
                randomized_obs[ind_of_ang_vel] += noise[ind_of_ang_vel_noise] * ang_vel_noise_scale

            obs[self._allowed_to_be_randomized] = randomized_obs[self._allowed_to_be_randomized]

        return obs, carry

    def update_action(self,
                      env: Any,
                      action: Union[np.ndarray, jnp.ndarray],
                      model: Union[MjModel, Model],
                      data: Union[MjData, Data],
                      carry: Any,
                      backend: ModuleType,
                      traj=None) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Update the action with randomization effects.

        Args:
            env (Any): The environment instance.
            action (Union[np.ndarray, jnp.ndarray]): The action to be updated.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: The updated action and carry.

        """

        assert_backend_is_supported(backend)
        return action, carry

    def _sample_geom_friction(self,
                              model: Union[MjModel, Model],
                              carry: Any,
                              backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Samples the geometry friction parameters.

        Args:
            model (Union[MjModel, Model]): The simulation model.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: The randomized geometry friction parameters and carry.

        """

        assert_backend_is_supported(backend)

        fric_tan_min, fric_tan_max = self.rand_conf["geom_friction_tangential_range"]
        fric_tor_min, fric_tor_max = self.rand_conf["geom_friction_torsional_range"]
        fric_roll_min, fric_roll_max = self.rand_conf["geom_friction_rolling_range"]

        if backend == jnp:
            key = carry.key
            key, _k = jax.random.split(key)
            interpolation = jax.random.uniform(_k, shape=(len(model.geom_friction),))
            carry = carry.replace(key=key)
        else:
            interpolation = np.random.uniform(size=(len(model.geom_friction),))

        sampled_friction_tangential = (
            fric_tan_min + (fric_tan_max - fric_tan_min) * interpolation
            if self.rand_conf["randomize_geom_friction_tangential"]
            else model.geom_friction[:, 0]
        )
        sampled_friction_torsional = (
            fric_tor_min + (fric_tor_max - fric_tor_min) * interpolation
            if self.rand_conf["randomize_geom_friction_torsional"]
            else model.geom_friction[:, 1]
        )
        sampled_friction_rolling = (
            fric_roll_min + (fric_roll_max - fric_roll_min) * interpolation
            if self.rand_conf["randomize_geom_friction_rolling"]
            else model.geom_friction[:, 2]
        )
        geom_friction = backend.array([
            sampled_friction_tangential,
            sampled_friction_torsional,
            sampled_friction_rolling,
        ]).T

        return geom_friction, carry

    def _sample_geom_damping_and_stiffness(self,
                                           model: Union[MjModel, Model],
                                           carry: Any,
                                           backend: ModuleType) \
            -> Tuple[Union[np.ndarray, jnp.ndarray], Union[np.ndarray, jnp.ndarray], Any]:
        """
        Samples the geometry damping and stiffness parameters.

        Args:
            model (Union[MjModel, Model]): The simulation model.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Union[np.ndarray, jnp.ndarray], Any]: The randomized geometry damping
            and stiffness parameters and carry.

        """

        assert_backend_is_supported(backend)

        damping_min, damping_max = self.rand_conf["geom_damping_range"]
        n_geoms = model.ngeom
        stiffness_min, stiffness_max = self.rand_conf["geom_stiffness_range"]

        if backend == jnp:
            key = carry.key
            key, _k_damp, _k_stiff = jax.random.split(key, 3)
            interpolation_damping = jax.random.uniform(_k_damp, shape=(n_geoms,))
            interpolation_stiff = jax.random.uniform(_k_stiff, shape=(n_geoms,))
            carry = carry.replace(key=key)
        else:
            interpolation_damping = np.random.uniform(size=(n_geoms,))
            interpolation_stiff = np.random.uniform(size=(n_geoms,))

        sampled_damping = (
            damping_min + (damping_max - damping_min) * interpolation_damping
            if self.rand_conf["randomize_geom_damping"]
            else model.geom_solref[:, 1]
        )
        sampled_stiffness = (
            stiffness_min + (stiffness_max - stiffness_min) * interpolation_stiff
            if self.rand_conf["randomize_geom_stiffness"]
            else model.geom_solref[:, 0]
        )

        return sampled_damping, sampled_stiffness, carry
    
    def _sample_joint_friction_loss(self,
                                    model: Union[MjModel, Model],
                                    carry: Any,
                                    backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Samples the joint friction loss parameters.

        Args:
            model (Union[MjModel, Model]): The simulation model.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: The randomized joint friction loss parameters.

        """

        assert_backend_is_supported(backend)

        friction_min, friction_max = self.rand_conf["joint_friction_loss_range"]
        n_dofs = model.nv - 6 #exclude freejoint 6 degrees of freedom

        if backend == jnp:
            key = carry.key
            key, _k = jax.random.split(key)
            interpolation = jax.random.uniform(_k, shape=(n_dofs,))
            carry = carry.replace(key=key)
        else:
            interpolation = np.random.uniform(size=(n_dofs,))

        sampled_friction_loss = (
            friction_min + (friction_max - friction_min) * interpolation
            if self.rand_conf["randomize_joint_friction_loss"]
            else model.dof_frictionloss[6:]
        )

        return sampled_friction_loss, carry
    
    def _sample_joint_damping(self, model: Union[MjModel, Model],
                              carry: Any,
                              backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Samples the joint damping parameters.

        Args:
            model (Union[MjModel, Model]): The simulation model.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: The randomized joint damping and carry.

        """

        assert_backend_is_supported(backend)

        damping_min, damping_max = self.rand_conf["joint_damping_range"]
        n_dofs = model.nv - 6 #exclude freejoint 6 degrees of freedom

        if backend == jnp:
            key = carry.key
            key, _k = jax.random.split(key)
            interpolation = jax.random.uniform(_k, shape=(n_dofs,))
            carry = carry.replace(key=key)
        else:
            interpolation = np.random.uniform(size=(n_dofs,))

        sampled_damping = (
            damping_min + (damping_max - damping_min) * interpolation
            if self.rand_conf["randomize_joint_damping"]
            else model.dof_damping[6:]
        )

        return sampled_damping, carry
    
    def _sample_joint_armature(self,
                               model: Union[MjModel, Model],
                               carry: Any,
                               backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Samples the joint armature parameters.

        Args:
            model (Union[MjModel, Model]): The simulation model.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: The randomized joint aramture paramters and carry.

        """

        assert_backend_is_supported(backend)

        armature_min, armature_max = self.rand_conf["joint_armature_range"]
        n_dofs = model.nv - 6 #exclude freejoint 6 degrees of freedom

        if backend == jnp:
            key = carry.key
            key, _k = jax.random.split(key)
            interpolation = jax.random.uniform(_k, shape=(n_dofs,))
            carry = carry.replace(key=key)
        else:
            interpolation = np.random.uniform(size=(n_dofs,))

        sampled_armature = (
            armature_min + (armature_max - armature_min) * interpolation
            if self.rand_conf["randomize_joint_armature"]
            else model.dof_armature[6:]
        )

        return sampled_armature, carry

    def _sample_gravity(self,
                        model: Union[MjModel, Model],
                        carry: Any, backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
         Samples the gravity vector.

         Args:
            model (Union[MjModel, Model]): The simulation model.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: The randomized gravity vector and carry.

        """

        assert_backend_is_supported(backend)

        gravity_min, gravity_max = self.rand_conf["gravity_range"]

        if backend == jnp:
            key = carry.key
            key, _k = jax.random.split(key)
            interpolation = jax.random.uniform(_k)
            carry = carry.replace(key=key)
        else:
            interpolation = np.random.uniform()

        sampled_gravity_z = (
            gravity_min + (gravity_max - gravity_min) * interpolation
            if self.rand_conf["randomize_gravity"]
            else model.opt.gravity[2])

        return backend.array([0.0, 0.0, -sampled_gravity_z]), carry

    def _sample_base_mass(self,
                          model: Union[MjModel, Model],
                          carry: Any, backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
         Samples a base mass to add to the robot.

         Args:
            model (Union[MjModel, Model]): The simulation model.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: The randomized gravity vector and carry.

        """

        assert_backend_is_supported(backend)

        base_mass_min, base_mass_max = self.rand_conf["base_mass_to_add_range"]

        if backend == jnp:
            key = carry.key
            key, _k = jax.random.split(key)
            interpolation = jax.random.uniform(_k)
            carry = carry.replace(key=key)
        else:
            interpolation = np.random.uniform()

        sampled_base_mass = (
            base_mass_min + (base_mass_max - base_mass_min) * interpolation
            if self.rand_conf["randomize_base_mass"]
            else 0.0)

        return sampled_base_mass, carry

    def _sample_com_displacement(self,
                                 model: Union[MjModel, Model],
                                 carry: Any,
                                 backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Samples a center-of-mass (COM) displace.

        Args:
            model (Union[MjModel, Model]): The simulation model.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: The randomized COM displacement vector and carry.

        """
        assert_backend_is_supported(backend)

        displ_min, displ_max = self.rand_conf["com_displacement_range"]

        if backend == jnp:
            key = carry.key
            key, _k = jax.random.split(key)
            interpolation = jax.random.uniform(_k, shape=(3,))
            carry = carry.replace(key=key)
        else:
            interpolation = np.random.uniform(size=3)

        sampled_com_displacement = (
            displ_min + (displ_max - displ_min) * interpolation
            if self.rand_conf["randomize_com_displacement"]
            else backend.array([0.0, 0.0, 0.0])
        )

        return sampled_com_displacement, carry
    
    def _sample_link_mass_multipliers(self,
                                      model: Union[MjModel, Model],
                                      carry: Any,
                                      backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Samples the link mass multipliers.

        Args:
            model (Union[MjModel, Model]): The simulation model.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: The randomized link mass multipliers and carry.

        """

        assert_backend_is_supported(backend)

        multiplier_dict = self.rand_conf["link_mass_multiplier_range"]

        mult_base_min, mult_base_max = multiplier_dict["root_body"]
        mult_other_min, mult_other_max = multiplier_dict["other_bodies"]

        if backend == jnp:
            key = carry.key
            key, _k = jax.random.split(key)
            interpolation = jax.random.uniform(_k, shape=(model.nbody-1,)) #exclude worldbody 
            carry = carry.replace(key=key)
        else:
            interpolation = np.random.uniform(size=model.nbody-1)

        sampled_base_mass_multiplier = (
            mult_base_min + (mult_base_max - mult_base_min) * interpolation[0]
            if self.rand_conf["randomize_link_mass"]
            else backend.array([1.0])
        )

        sampled_base_mass_multiplier = jnp.expand_dims(sampled_base_mass_multiplier, axis=0)

        sampled_other_bodies_mass_multipliers = (
            mult_other_min + (mult_other_max - mult_other_min) * interpolation[1:]
            if self.rand_conf["randomize_link_mass"]
            else backend.array([1.0]*(model.nbody-2))
        )

        mass_multipliers = backend.concatenate([
            sampled_base_mass_multiplier,
            sampled_other_bodies_mass_multipliers,
        ])

        return mass_multipliers, carry
    
    def _sample_p_gains_noise(self,
                              env, model: Union[MjModel, Model],
                              carry: Any,
                              backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Samples p_gains_noise for the PDControl control function.

        Args:
            model (Union[MjModel, Model]): The simulation model.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: The p_gains_noise and carry.
        """
        assert_backend_is_supported(backend)

        init_p_gain = env._control_func._init_p_gain

        noise_shape = (len(init_p_gain),) if init_p_gain.size > 1 else (1,)

        if backend == jnp:
            key = carry.key
            key, _k = jax.random.split(key)
            interpolation = jax.random.uniform(_k, shape=noise_shape, minval=-1.0, maxval=1.0)
            carry = carry.replace(key=key)
        else:
            interpolation = np.random.uniform(size=noise_shape, low=-1.0, high=1.0)

        p_noise_scale = self.rand_conf["p_gains_noise_scale"]

        p_noise = (
            interpolation * (p_noise_scale * init_p_gain) 
            if self.rand_conf["add_p_gains_noise"]
            else backend.array([0.0]*len(init_p_gain))
            )

        return p_noise, carry
    
    def _sample_d_gains_noise(self,
                              env, model: Union[MjModel, Model],
                              carry: Any,
                              backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Samples d_gains_noise for the PDControl control function..

        Args:
            model (Union[MjModel, Model]): The simulation model.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: The d_gains_noise and carry.

        """

        assert_backend_is_supported(backend)

        init_d_gain = env._control_func._init_d_gain

        noise_shape = (len(init_d_gain),) if init_d_gain.size > 1 else (1,)

        if backend == jnp:
            key = carry.key
            key, _k = jax.random.split(key)
            interpolation = jax.random.uniform(_k, shape=noise_shape, minval=-1.0, maxval=1.0)
            carry = carry.replace(key=key)
        else:
            interpolation = np.random.uniform(size=noise_shape, low=-1.0, high=1.0)

        d_noise_scale = self.rand_conf["d_gains_noise_scale"]

        d_noise = (
            interpolation * (d_noise_scale * init_d_gain) 
            if self.rand_conf["add_d_gains_noise"]
            else backend.array([0.0]*len(init_d_gain))
            )

        return d_noise, carry