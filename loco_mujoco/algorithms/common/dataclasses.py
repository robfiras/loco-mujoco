from typing import NamedTuple
from typing import Any, Optional

import jax
import jax.numpy as jnp
from flax import struct
from flax.training import train_state

from loco_mujoco.environments.base import TrajState
from loco_mujoco.core.wrappers.mjx import Metrics


class Transition(NamedTuple):
    done: jnp.ndarray
    absorbing: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    traj_state: TrajState
    metrics: Metrics


class MetricHandlerTransition(NamedTuple):
    env_state: Any
    logged_metrics: Metrics


@struct.dataclass
class AdaptiveLRState:
    learning_rate: jnp.ndarray

class TrainState(train_state.TrainState):
    run_stats: Any
    adaptive_lr_state: Optional[AdaptiveLRState]

@struct.dataclass
class TrainStateBuffer:
    train_states: TrainState
    n: int
    size: int   # buffer size

    @classmethod
    def create(cls, train_state: TrainState, size: int):
        return TrainStateBuffer(
            train_states=jax.tree.map(lambda x: jnp.stack([x] * size), train_state),
            n=0,
            size=size
        )

    @classmethod
    def add(cls, train_state_buffer, train_state: TrainState):
        index = train_state_buffer.n
        # Add the new train state at index n
        train_states_updated = jax.tree.map(
            lambda buffer, new: buffer.at[index].set(new),
            train_state_buffer.train_states,
            train_state
        )
        return train_state_buffer.replace(
            train_states=train_states_updated,
            n=index + 1,
        )


@struct.dataclass
class BestTrainStates:
    train_states: TrainState
    metrics: jnp.array
    iterations: jnp.array
    cur_worst_perf: float
    step: int
    n: int
    size: int

    @classmethod
    def create(cls, train_state: TrainState, n: int):
        return BestTrainStates(
            train_states=jax.tree.map(lambda x: jnp.stack([x] * n), train_state),
            metrics=jnp.full((n,), -jnp.inf),
            iterations=jnp.zeros((n,)),
            cur_worst_perf=-jnp.inf,
            n=n,
            size=0
        )
