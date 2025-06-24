from typing import Any, Dict, Tuple
from types import ModuleType

import mujoco
from mujoco import mjx
from mujoco.mjx import Model, Data
from flax import struct
import numpy as np
import jax
import jax.numpy as jnp

from loco_mujoco.core.mujoco_base import Mujoco, AdditionalCarry
from loco_mujoco.core.visuals import MujocoViewer
from loco_mujoco.trajectory import TrajectoryData


@struct.dataclass
class MjxAdditionalCarry(AdditionalCarry):
    """
    Additional carry for the Mjx environment.

    """
    final_observation: jax.Array
    final_info: Dict[str, Any]


@struct.dataclass
class MjxState:
    """
    State of the Mjx environment.

    Args:
        data (Data): Mjx data structure.
        observation (jax.Array): Observation of the environment.
        reward (float): Reward of the environment.
        absorbing (bool): Whether the state is absorbing.
        done (bool): Whether the episode is done.
        additional_carry (Any): Additional carry information.
        info (Dict[str, Any]): Information dictionary.

    """
    data: Data
    observation: jax.Array
    reward: float
    absorbing: bool
    done: bool
    additional_carry: MjxAdditionalCarry
    info: Dict[str, Any] = struct.field(default_factory=dict)


class Mjx(Mujoco):
    """
    Base class for Mujoco environments using JAX.

    Args:
        n_envs (int): Number of environments to run in parallel.
        **kwargs: Additional arguments to pass to the Mujoco base class.

    """

    def __init__(self, **kwargs):

        # call base mujoco env
        super().__init__(**kwargs)

        # add information to mdp_info
        self._mdp_info.mjx_env = True

        # setup mjx model and data
        mujoco.mj_resetData(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)
        self.sys = mjx.put_model(self._model)
        data = mjx.put_data(self._model, self._data)
        self._first_data = mjx.forward(self.sys, data)

    def mjx_reset(self, key: jax.random.PRNGKey, env_id: int = None) -> MjxState:
        """
        Resets the environment.

        Args:
            key (jax.random.PRNGKey): Random key for the reset.

        Returns:
            MjxState: The reset state of the environment.

        """

        key, subkey = jax.random.split(key)

        # reset data
        data = self._first_data

        carry = self._init_additional_carry(key, self._model, data, jnp, env_id)

        data, carry = self._mjx_reset_carry(self.sys, data, carry)

        # reset all stateful entities
        data, carry = self.obs_container.reset_state(self, self._model, data, carry, jnp)

        obs, carry = self._mjx_create_observation(self._model, data, carry)
        reward = 0.0
        absorbing = jnp.array(False, dtype=bool)
        done = jnp.array(False, dtype=bool)
        info = self._mjx_reset_info_dictionary(obs, data, subkey)

        return MjxState(data=data, observation=obs, reward=reward, absorbing=absorbing, done=done,
                        info=info, additional_carry=carry)

    def _mjx_reset_in_step(self, state: MjxState) -> MjxState:
        """
        Resets the environment if the episode is done. This function is called in the step function for asynchronous
        resetting of the environments.

        Args:
            state (MjxState): Current state of the environment.

        Returns:
            MjxState: The reset state of the environment.

        """

        carry = state.additional_carry

        # reset data
        data = self._first_data

        data, carry = self._mjx_reset_carry(self.sys, data, carry)

        # reset carry
        carry = carry.replace(cur_step_in_episode=1,
                              final_observation=state.observation,
                              last_action=jnp.zeros_like(carry.last_action),
                              final_info=state.info)

        # update all stateful entities
        data, carry = self.obs_container.reset_state(self, self._model, data, carry, jnp)

        # create new observation
        obs, carry = self._mjx_create_observation(self._model, data, carry)

        return state.replace(data=data, observation=obs, additional_carry=carry)

    def mjx_step(self, state: MjxState, action: jax.Array) -> MjxState:
        """

        Args:
            state (MjxState): Current state of the environment.
            action (jax.Array): Action to take in the environment.

        Returns:
            MjxState: The next state of the environment.

        """

        data = state.data
        cur_info = state.info
        carry = state.additional_carry
        carry = carry.replace(last_action=action)

        # reset dones
        state = state.replace(done=jnp.zeros_like(state.done, dtype=bool))

        # preprocess action
        processed_action, carry = self._mjx_preprocess_action(action, self._model, data, carry)

        # modify data and model *before* step if needed
        sys, data, carry = self._mjx_simulation_pre_step(self.sys, data, carry)

        def _inner_loop(idx, _runner_state):

            _data, _carry = _runner_state

            ctrl_action, _carry = self._mjx_compute_action(processed_action, self._model, _data, _carry)

            # step in the environment using the action
            ctrl = _data.ctrl.at[jnp.array(self._action_indices)].set(ctrl_action)
            _data = _data.replace(ctrl=ctrl)
            step_fn = lambda _, x: mjx.step(sys, x)
            _data = jax.lax.fori_loop(0, self._n_substeps, step_fn, _data)

            return _data, _carry

        # run inner loop
        data, carry = jax.lax.fori_loop(0, self._n_intermediate_steps, _inner_loop, (data, carry))

        # modify data *after* step if needed (does nothing by default)
        data, carry = self._mjx_simulation_post_step(self._model, data, carry)

        # create the observation
        cur_obs, carry = self._mjx_create_observation(sys, data, carry)

        # modify the observation and the data if needed (does nothing by default)
        cur_obs, data, cur_info, carry = self._mjx_step_finalize(cur_obs, self._model, data, cur_info, carry)

        # create info
        cur_info = self._mjx_update_info_dictionary(cur_info, cur_obs, data, carry)

        # check if the next obs is an absorbing state
        absorbing, carry = self._mjx_is_absorbing(cur_obs, cur_info, data, carry)

        # calculate the reward
        reward, carry = self._mjx_reward(state.observation, action, cur_obs, absorbing, cur_info, self._model, data, carry)

        # check if done
        done = self._mjx_is_done(cur_obs, absorbing, cur_info, data, carry)

        done = jnp.logical_or(done, jnp.any(jnp.isnan(cur_obs)))
        cur_obs = jnp.nan_to_num(cur_obs, nan=0.0)

        # create state
        carry = carry.replace(cur_step_in_episode=carry.cur_step_in_episode + 1)
        state = state.replace(data=data, observation=cur_obs, reward=reward,
                              absorbing=absorbing, done=done, info=cur_info, additional_carry=carry)

        # reset state if done
        state = jax.lax.cond(state.done, self._mjx_reset_in_step, lambda x: x, state)

        return state

    def _mjx_create_observation(self, model: Model,
                                data: Data,
                                carry: MjxAdditionalCarry) -> jax.Array:
        """
        Creates the observation for the environment.

        Args:
            model (Model): Mjx model.
            data (Data): Mjx data structure.
            carry (MjxAdditionalCarry): Additional carry information.

        Returns:
            jax.Array: The observation of the environment.

        """
        return self._create_observation_compat(model, data, carry, jnp)

    def _mjx_reset_info_dictionary(self, obs: jnp.ndarray,
                                   data: Data,
                                   key: jax.random.PRNGKey) -> Dict:
        """
        Resets the info dictionary.

        Args:
            obs (jnp.ndarray): Observation of the environment.
            data (Data): Mjx data structure.
            key (jax.random.PRNGKey): Random key.

        Returns:
            Dict: The updated info dictionary.

        """
        return {}

    def _mjx_update_info_dictionary(self, info: Dict,
                                    obs: jnp.ndarray,
                                    data: Data,
                                    carry: MjxAdditionalCarry) -> Dict:
        """
        Updates the info dictionary.

        Args:
            obs (jnp.ndarray): Observation of the environment.
            data (Data): Mjx data structure.
            carry (MjxAdditionalCarry): Additional carry information.

        Returns:
            Dict: The updated info dictionary.

        """
        return info

    def _mjx_reward(self, obs: jnp.ndarray,
                    action: jnp.ndarray,
                    next_obs: jnp.ndarray,
                    absorbing: bool,
                    info: Dict,
                    model: Model,
                    data: Data,
                    carry: MjxAdditionalCarry) -> Tuple[float, MjxAdditionalCarry]:
        """
        Calls the reward function of the environment.

        Args:
            obs (jnp.ndarray): Observation of the environment.
            action (jnp.ndarray): Action taken in the environment.
            next_obs (jnp.ndarray): Next observation of the environment.
            absorbing (bool): Whether the next state is absorbing.
            info (Dict): Information dictionary.
            model (Model): Mjx model.
            data (Data): Mjx data structure.
            carry (MjxAdditionalCarry): Additional carry information.

        Returns:
            Tuple[float, MjxAdditionalCarry]: The reward and the updated carry.

        """
        reward, carry = self._reward_function(obs, action, next_obs, absorbing, info, self, model, data, carry, jnp)
        return reward, carry

    def _mjx_is_absorbing(self, obs: jnp.ndarray,
                          info: Dict,
                          data: Data,
                          carry: MjxAdditionalCarry) -> bool:
        """
        Determines if the current state is absorbing.

        Args:
            obs (jnp.ndarray): Current observation.
            info (Dict): Information dictionary.
            data (Data): Mujoco data structure.
            carry (MjxAdditionalCarry): Additional carry information.

        Returns:
            bool: True if the state is absorbing, False otherwise.
        """
        return self._terminal_state_handler.mjx_is_absorbing(self, obs, info, data, carry)

    def _mjx_is_done(self, obs: jnp.ndarray,
                     absorbing: bool,
                     info: Dict,
                     data: Data,
                     carry: MjxAdditionalCarry) -> bool:
        """
        Determines if the episode is done.

        Args:
            obs (jnp.ndarray): Current observation.
            absorbing (bool): Whether the next state is absorbing.
            info (Dict): Information dictionary.
            data (Data): Mujoco data structure.
            carry (MjxAdditionalCarry): Additional carry information.

        Returns:
            bool: True if the episode is done, False otherwise.
        """
        done = jnp.greater_equal(carry.cur_step_in_episode, self.info.horizon)
        done = jnp.logical_or(done, absorbing)
        return done

    def _mjx_simulation_pre_step(self, model: Model,
                                 data: Data,
                                 carry: MjxAdditionalCarry) -> Tuple[Model, Data, MjxAdditionalCarry]:
        """
        Applies pre-step modifications to the model, data, and carry.

        Args:
            model (Model): Mujoco model.
            data (Data): Mujoco data structure.
            carry (MjxAdditionalCarry): Additional carry information.

        Returns:
            Tuple[Model, Data, MjxAdditionalCarry]: Updated model, data, and carry.
        """
        model, data, carry = self._terrain.update(self, model, data, carry, jnp)
        model, data, carry = self._domain_randomizer.update(self, model, data, carry, jnp)
        return model, data, carry

    def _mjx_simulation_post_step(self, model: Model,
                                  data: Data,
                                  carry: MjxAdditionalCarry) -> Tuple[Data, MjxAdditionalCarry]:
        """
        Applies post-step modifications to the data and carry.

        Args:
            model (Model): Mujoco model.
            data (Data): Mujoco data structure.
            carry (MjxAdditionalCarry): Additional carry information.

        Returns:
            Tuple[Data, MjxAdditionalCarry]: Updated data and carry.
        """
        return data, carry

    def _mjx_preprocess_action(self, action: jnp.ndarray,
                               model: Model,
                               data: Data,
                               carry: MjxAdditionalCarry) -> Tuple[jnp.ndarray, MjxAdditionalCarry]:
        """
        Transforms the action before applying it to the environment.

        Args:
            action (jnp.ndarray): Action input.
            model (Model): Mujoco model.
            data (Data): Mujoco data structure.
            carry (MjxAdditionalCarry): Additional carry information.

        Returns:
            Tuple[jnp.ndarray, MjxAdditionalCarry]: Processed action and updated carry.
        """
        action, carry = self._domain_randomizer.update_action(self, action, model, data, carry, jnp)
        return action, carry

    def _mjx_compute_action(self, action: jnp.ndarray,
                            model: Model,
                            data: Data,
                            carry: MjxAdditionalCarry) -> Tuple[jnp.ndarray, MjxAdditionalCarry]:
        """
        Applies transformations to the action at intermediate steps.

        Args:
            action (jnp.ndarray): Action at the current step.
            model (Model): Mujoco model.
            data (Data): Mujoco data structure.
            carry (MjxAdditionalCarry): Additional carry information.

        Returns:
            Tuple[jnp.ndarray, MjxAdditionalCarry]: Computed action and updated carry.
        """
        action, carry = self._control_func.generate_action(self, action, model, data, carry, jnp)
        return action, carry

    def _mjx_reset_carry(self, model: Model,
                         data: Data,
                         carry: MjxAdditionalCarry) -> Tuple[Data, MjxAdditionalCarry]:
        """
        Resets the additional carry and allows modification to the Mujoco data.

        Args:
            model (Model): Mujoco model.
            data (Data): Mujoco data structure.
            carry (MjxAdditionalCarry): Additional carry information.

        Returns:
            Tuple[Data, MjxAdditionalCarry]: Updated data and carry.
        """
        data, carry = self._terminal_state_handler.reset(self, model, data, carry, jnp)
        data, carry = self._terrain.reset(self, model, data, carry, jnp)
        data, carry = self._init_state_handler.reset(self, model, data, carry, jnp)
        data, carry = self._control_func.reset(self, model, data, carry, jnp)
        data, carry = self._domain_randomizer.reset(self, model, data, carry, jnp)
        data, carry = self._reward_function.reset(self, model, data, carry, jnp)
        return data, carry

    def _mjx_step_finalize(self, obs: jnp.ndarray,
                           model: Model,
                           data: Data,
                           info: Dict,
                           carry: MjxAdditionalCarry) -> Tuple[jnp.ndarray, Data, Dict, MjxAdditionalCarry]:
        """
        Allows information to be accessed at the end of a step.

        Args:
            obs (jnp.ndarray): Observation.
            model (Model): Mujoco model.
            data (Data): Mujoco data structure.
            info (Dict): Information dictionary.
            carry (MjxAdditionalCarry): Additional carry information.

        Returns:
            Tuple[jnp.ndarray, Data, Dict, MjxAdditionalCarry]: Updated observation, data, info, and carry.
        """
        obs, carry = self._domain_randomizer.update_observation(self, obs, model, data, carry, jnp)
        return obs, data, info, carry

    @staticmethod
    def mjx_set_sim_state_from_traj_data(data: Data,
                                         traj_data: TrajectoryData,
                                         carry: MjxAdditionalCarry) -> Data:
        """
        Sets the simulation state from the trajectory data.

        Args:
            data (Data): Current Mujoco data.
            traj_data (TrajectoryData): Data from the trajectory.
            carry (MjxAdditionalCarry): Additional carry information.

        Returns:
            Data: Updated Mujoco data.
        """
        return data.replace(
            xpos=traj_data.xpos if traj_data.xpos.size > 0 else data.xpos,
            xquat=traj_data.xquat if traj_data.xquat.size > 0 else data.xquat,
            cvel=traj_data.cvel if traj_data.cvel.size > 0 else data.cvel,
            qpos=traj_data.qpos if traj_data.qpos.size > 0 else data.qpos,
            qvel=traj_data.qvel if traj_data.qvel.size > 0 else data.qvel)

    def _mjx_set_sim_state_from_obs(self, data: Data,
                                    obs: jnp.ndarray) -> Data:
        """
        Updates the simulation state from an observation.

        .. note:: This may not fully set the state of the simulation if the observation does not contain all the
                  necessary information.

        Args:
            data (Data): Current Mujoco data.
            obs (jnp.ndarray): Observation containing state information.

        Returns:
            Data: Updated Mujoco data.
        """
        data = data.replace(
            qpos=data.qpos.at[self._data_indices.free_joint_qpos].set(obs[self._obs_indices.free_joint_qpos]),
            qvel=data.qvel.at[self._data_indices.free_joint_qvel].set(obs[self._obs_indices.free_joint_qvel]))

        return data.replace(
            xpos=data.xpos.at[self._data_indices.body_xpos].set(obs[self._obs_indices.body_xpos].reshape(-1, 3)),
            xquat=data.xquat.at[self._data_indices.body_xquat].set(obs[self._obs_indices.body_xquat].reshape(-1, 4)),
            cvel=data.cvel.at[self._data_indices.body_cvel].set(obs[self._obs_indices.body_cvel].reshape(-1, 6)),
            qpos=data.qpos.at[self._data_indices.joint_qpos].set(obs[self._obs_indices.joint_qpos]),
            qvel=data.qvel.at[self._data_indices.joint_qvel].set(obs[self._obs_indices.joint_qvel]),
            site_xpos=data.site_xpos.at[self._data_indices.site_xpos].set(
                obs[self._obs_indices.site_xpos].reshape(-1, 3)),
            site_xmat=data.site_xmat.at[self._data_indices.site_xmat].set(
                obs[self._obs_indices.site_xmat].reshape(-1, 9)))

    def mjx_render(self, state,
                   record: bool = False) -> np.ndarray:
        """
        Renders all environments in parallel.

        Args:
            state: Current environment state.
            record (bool): Whether to record the rendering.

        Returns:
            np.ndarray: Rendered image.
        """
        if self._viewer is None:
            if "default_camera_mode" not in self._viewer_params.keys():
                self._viewer_params["default_camera_mode"] = "static"
            self._viewer = MujocoViewer(self._model, self.dt, record=record, **self._viewer_params)

        if self._terrain.is_dynamic:
            terrain_state = state.additional_carry.terrain_state
            assert hasattr(terrain_state, "height_field_raw"), "Terrain state does not have height_field_raw."
            assert self._terrain.hfield_id is not None, "Terrain hfield id is not set."
            hfield_data = np.array(terrain_state.height_field_raw)
            self._model.hfield_data = hfield_data[0]
            self._viewer.upload_hfield(self._model, hfield_id=self._terrain.hfield_id)

        return self._viewer.parallel_render(state, record)

    def mjx_render_trajectory(self, trajectory,
                              record: bool = False) -> None:
        """
        Renders a trajectory sequence.

        Args:
            trajectory: A sequence of environment states.
            record (bool): Whether to record the rendering.

        """
        assert len(trajectory) > 0, "Mjx render got provided with an empty trajectory."

        if self._viewer is None:
            self._viewer = MujocoViewer(self._model, self.dt, record=record, **self._viewer_params)

        n_envs = trajectory[0].data.qpos.shape[0]

        for i in range(n_envs):
            for state in trajectory:
                self._data.qpos, self._data.qvel = state.data.qpos[i, :], state.data.qvel[i, :]
                mujoco.mj_forward(self._model, self._data)
                self._viewer.render(self._data, record)

    def _init_additional_carry(self, key,
                               model: Model,
                               data: Data,
                               backend: ModuleType,
                               env_id: int = None) -> MjxAdditionalCarry:
        """
        Initializes additional carry parameters.

        Args:
            key: Random key for initialization.
            model (Model): Mujoco model.
            data (Data): Mujoco data structure.
            backend (ModuleType): Computational backend (either numpy or jax.numpy).

        Returns:
            MjxAdditionalCarry: Initialized carry object.
        """
        carry = super()._init_additional_carry(key, model, data, backend, env_id)
        return MjxAdditionalCarry(final_observation=backend.zeros(self.info.observation_space.shape),
                                  final_info={},
                                  **vars(carry))

    @property
    def n_envs(self) -> int:
        """Returns the number of environments."""
        return self._n_envs

    @property
    def mjx_env(self) -> bool:
        """Indicates whether this is an MJX environment."""
        return True

