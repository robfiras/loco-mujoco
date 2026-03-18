from typing import List, Any, Union, Tuple, Dict
from types import ModuleType

import numpy as np
import jax
import jax.numpy as jnp
from flax import struct
from dataclasses import asdict

from mujoco import MjSpec, MjModel, MjData
from mujoco.mjx import Model, Data

from loco_mujoco import RLFactory
from loco_mujoco.environments import UnitreeG1
from loco_mujoco.core.initial_state_handler import InitialStateHandler
from loco_mujoco.core.observations import Observation, StatefulObservation
from loco_mujoco.core.control_functions.pd import PDControl, PDControlState
from loco_mujoco.core.terminal_state_handler import TerminalStateHandler
from loco_mujoco.core.reward import Reward


# ----- CUSTOM ENVIRONMENT -----
# custom environment class with fixed waist yaw joint
class CustomUnitreeG1(UnitreeG1):
    def __init__(self, spec=None, **kwargs):
        self._waist_yaw_joint_name = "waist_yaw_joint"
        self._waist_yaw_actuator_name = "waist_yaw"
        super().__init__(spec=spec or self.get_default_xml_file_path(), **kwargs)

    @staticmethod
    def fix_waist_yaw(spec: MjSpec):
        spec.joints = [j for j in spec.joints if j.name != "waist_yaw_joint"]

    def _get_observation_specification(self, spec: MjSpec) -> List[Observation]:
        obs = super()._get_observation_specification(spec)
        return [o for o in obs if o.xml_name != self._waist_yaw_joint_name]

    def _get_action_specification(self, spec: MjSpec) -> List[str]:
        actions = super()._get_action_specification(spec)
        return [a for a in actions if a != self._waist_yaw_actuator_name]


# ----- CUSTOM INITIAL STATE HANDLER -----
# custom initial state handler to set the initial height of the robot between 2.0 and 2.5
class CustomInitialStateHandler(InitialStateHandler):
    def reset(self, env: Any, model: Union[MjModel, Model],
              data: Union[MjData, Data], carry: Any,
              backend: ModuleType) -> Tuple[Union[MjData, Data], Any]:
        if backend == np:
            data.qpos[2] = np.random.uniform(2.0, 2.5)
        else:
            key, subkey = jax.random.split(carry.key)
            z = jax.random.uniform(subkey, (1,), minval=2.0, maxval=2.5)
            data = data.replace(qpos=data.qpos.at[2].set(z))
            carry = carry.replace(key=key)
        return data, carry


# ----- CUSTOM CONTROL FUNCTION -----
@struct.dataclass
class CustomControlFunctionState(PDControlState):
    moving_average: Union[np.ndarray, jnp.ndarray]

class CustomControlFunction(PDControl):
    def generate_action(self, env, action, model, data, carry, backend):
        orig_action, carry = super().generate_action(env, action, model, data, carry, backend)
        ma = 0.99 * carry.control_func_state.moving_average + 0.01 * orig_action
        state = carry.control_func_state.replace(moving_average=ma)
        carry = carry.replace(control_func_state=state)
        return ma, carry

    def init_state(self, env, key, model, data, backend):
        orig_state = super().init_state(env, key, model, data, backend)
        dim = env.info.action_space.shape[0]
        ma = backend.zeros_like(dim)
        return CustomControlFunctionState(moving_average=ma, **asdict(orig_state))


# ----- CUSTOM REWARD -----
# custom reward function to penalize the deviation from the robots zero joint position
class CustomReward(Reward):
    def __call__(self, state, action, next_state, absorbing, info, env,
                 model, data, carry, backend):
        reward = backend.exp(-backend.linalg.norm(data.qpos[7:]))
        return reward, carry


# ----- CUSTOM TERMINAL STATE HANDLER -----
# custom terminal state handler to terminate the episode with a probability of 0.05
class CustomTerminalStateHandler(TerminalStateHandler):
    def reset(self, env, model, data, carry, backend):
        return data, carry

    def is_absorbing(self, env, obs, info, data, carry):
        return np.random.uniform() < 0.05, carry

    def mjx_is_absorbing(self, env, obs, info, data, carry):
        key, subkey = jax.random.split(carry.key)
        absorbing = jax.random.uniform(subkey) < 0.05
        return absorbing, carry.replace(key=key)


# ----- CUSTOM OBSERVATIONS -----
# custom observation class to observe the center of mass position of the pelvis
class CustomBodyCOMPos(Observation):
    dim = 3

    def __init__(self, name: str, xml_name: str):
        self.xml_name = xml_name
        super().__init__(name)

    def _init_from_mj(self, env, model, data, current_obs_size):
        self.min, self.max = [-np.inf] * self.dim, [np.inf] * self.dim
        self.data_type_ind = np.array(self.to_list(data.body(self.xml_name).id))
        self.obs_ind = np.arange(current_obs_size, current_obs_size + self.dim)
        self._initialized_from_mj = True

    @classmethod
    def data_type(cls):
        return "xipos"


# custom observation class to observe the moving average of the center of mass position of the pelvis
@struct.dataclass
class CustomBodyCOMPosMovingAverageState:
    moving_average: Union[np.ndarray, jnp.ndarray]

class CustomBodyCOMPosMovingAverage(StatefulObservation):
    dim = 3

    def __init__(self, name: str, xml_name: str):
        self.xml_name = xml_name
        super().__init__(name)

    def _init_from_mj(self, env, model, data, current_obs_size):
        self.min, self.max = [-np.inf] * self.dim, [np.inf] * self.dim
        self.obs_ind = np.arange(current_obs_size, current_obs_size + self.dim)
        self.data_type_ind = np.array(self.to_list(data.body(self.xml_name).id))
        self._initialized_from_mj = True

    def init_state(self, env, key, model, data, backend):
        return CustomBodyCOMPosMovingAverageState(moving_average=backend.zeros(self.dim))

    def get_obs_and_update_state(self, env, model, data, carry, backend):
        obs_states = carry.observation_states
        obs_state = getattr(obs_states, self.name)
        xipos = backend.squeeze(data.xipos[self.data_type_ind])
        ma = 0.9 * obs_state.moving_average + 0.1 * xipos
        obs_states = obs_states.replace(**{self.name: obs_state.replace(moving_average=ma)})
        carry = carry.replace(observation_states=obs_states)
        return backend.ravel(ma), carry


# ----- REGISTRATION -----
CustomUnitreeG1.register()
CustomInitialStateHandler.register()
CustomControlFunction.register()
CustomReward.register()
CustomTerminalStateHandler.register()
CustomBodyCOMPos.register()
CustomBodyCOMPosMovingAverage.register()


# ----- ENVIRONMENT SETUP -----
observation_spec = [
    CustomBodyCOMPos("pelvis_com", "pelvis"),
    CustomBodyCOMPosMovingAverage("pelvis_com_mov_avg", "pelvis"),
]

env, traj = RLFactory.make(
    "CustomUnitreeG1",
    init_state_type="CustomInitialStateHandler",
    control_type="CustomControlFunction",
    control_params={"p_gain": 100, "d_gain": 1},
    reward_type="CustomReward",
    terminal_state_type="CustomTerminalStateHandler",
    observation_spec=observation_spec,
)

# ----- MAIN LOOP -----
action_dim = env.info.action_space.shape[0]
env.reset()
env.render()

i, absorbing = 0, False

while True:
    if i == 1000 or absorbing:
        env.reset()
        i = 0
    action = np.random.randn(action_dim)
    _, _, absorbing, _, _ = env.step(action)
    env.render()
    i += 1
