"""Base class for environment components that keep a state in the carry of the environment."""

from typing import List

from flax import struct


@struct.dataclass
class EmptyState:
    """State of a stateful object that does not need to store anything."""

    pass


class StatefulObject:
    """
    Base class for components of an environment that own a state and, optionally, visual geoms.

    The environments are functional: everything that changes over time is stored in the additional carry instead
    of on the object. A stateful object (an observation, a reward, a terrain, a domain randomizer, ...) creates its
    state with :meth:`init_state` and resets it at the beginning of an episode with :meth:`reset_state`.

    Every instance is recorded in a class-level list shared by all subclasses, which
    :meth:`~loco_mujoco.core.visuals.scene.MjvScene.init_for_all_stateful_objects` uses to assign each object its
    slots in the user scene.

    Attributes:
        n_visual_geoms (int): Number of visual geoms the object draws into the user scene.
        visual_geoms_idx (np.ndarray): Indices of these geoms in the user scene; None until the scene is created.

    """

    _instances: List["StatefulObject"] = []

    def __init__(self, n_visual_geoms: int =0):
        """
        Constructor.

        Args:
            n_visual_geoms (int): Number of visual geoms the object draws into the user scene.

        """
        self.n_visual_geoms = n_visual_geoms
        self.visual_geoms_idx = None
        self._instances.append(self)

    def reset_state(self, env, model, data, carry, backend, traj=None):
        """
        Reset the state of the object at the beginning of an episode. The default implementation does nothing.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Additional carry.
            backend (ModuleType): Backend module, either ``numpy`` or ``jax.numpy``.
            traj (Trajectory, optional): The trajectory of the environment.

        Returns:
            Tuple[Union[MjData, Data], Any]: The (possibly updated) simulation data and carry.

        """
        return data, carry

    def init_state(self, env, key, model, data, backend, traj=None):
        """
        Create the initial state of the object. The default implementation returns an :class:`EmptyState`.

        Args:
            env (Any): The environment instance.
            key (jax.Array): JAX random key.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            backend (ModuleType): Backend module, either ``numpy`` or ``jax.numpy``.
            traj (Trajectory, optional): The trajectory of the environment.

        Returns:
            EmptyState: The initial state.

        """
        return EmptyState()

    @classmethod
    def get_all_instances(cls) -> List["MjvGeom"]:
        """
        Return all stateful objects created so far, of this class and of every other subclass.

        Returns:
            List[StatefulObject]: The list of instances, in the order of their creation.

        """
        return cls._instances
