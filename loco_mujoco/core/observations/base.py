"""
Observation types and the container that assembles them into the observation vector of an environment.

Simple observation types read entries of the Mujoco data structure (joint positions, body poses, ...);
stateful observations compute their values from the model, the data and the carry, and may keep a state.

"""

from __future__ import annotations
from copy import deepcopy
from typing import List, Sequence, Union
from collections import UserDict

import numpy as np
import mujoco
import jax
import jax.numpy as jnp
from dataclasses import make_dataclass
from flax import struct
from scipy.spatial.transform import Rotation as np_R
from jax.scipy.spatial.transform import Rotation as jnp_R

from loco_mujoco.core.stateful_object import StatefulObject
from loco_mujoco.core.utils.mujoco import (mj_jnt_name2id, mj_jntname2qposid, mj_jntname2qvelid,
                                           mj_jntid2qposid, mj_jntid2qvelid)
from loco_mujoco.core.utils.math import quat_scalarfirst2scalarlast, calculate_relative_site_quatities


class ObservationIndexContainer:
    """
    Container for indices of different non-stateful observation types, used to store indices
    related to observations within Mujoco data structures or observations
    created in an environment.

    The container has one attribute per non-stateful observation type, named like the type, that holds the
    indices of all observations of that type.

    Attributes:
        concatenated_indices (np.ndarray): Permutation that sorts the concatenation of all index lists; None
            until :meth:`convert_to_numpy` is called.

    """

    def __init__(self):

        # add attributes for the different observation types
        for obs_type in ObservationType.list_all_non_stateful():
            setattr(self, obs_type.__name__, [])

        self.concatenated_indices = None

    def convert_to_numpy(self):
        """Converts all list attributes of the class to NumPy arrays in place and computes ``concatenated_indices``."""
        ind = []
        # Iterate through all attributes of the class
        for attr_name in vars(self):
            attr_value = getattr(self, attr_name)
            if attr_value is not None:
                ind += attr_value
            # Check if the attribute is a list before converting
            if isinstance(attr_value, list):
                setattr(self, attr_name, np.array(attr_value, dtype=int))

        # this array concatenates all indices in the order of this class
        self.concatenated_indices = np.argsort(np.array(ind))


class ObservationContainer(UserDict):
    """
    Container for the observations of an environment.

    This is a dictionary from observation names to :class:`Observation` instances with additional functionality:
    it sets the container reference of each observation, keeps track of the stateful observations, and gives
    access to the indices of observation groups. The order of insertion defines the layout of the observation
    vector.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stateful_obs = []
        self._locked = False

    def __setitem__(self, key, observation: Observation):
        """
        Add an observation to the container.

        Args:
            key (str): Name under which the observation is stored.
            observation (Observation): The observation to add.

        Raises:
            KeyError: If an observation with the same name is already in the container.
            ValueError: If the container is locked.

        """

        if not self._locked:

            if observation.name in self.keys():
                raise KeyError("Duplicate keys are not allowed. Key: ", observation.name)

            # Set the container reference before adding to dict
            if isinstance(observation, Observation):
                observation.obs_container = self
            # Add the observation to the container
            super().__setitem__(key, observation)
            # Add the observation also to the stateful_obs container if it is a stateful observation
            if isinstance(observation, StatefulObservation):
                self._stateful_obs.append(observation)
        else:
            raise ValueError("Container is locked and cannot be modified.")

    def names(self):
        """
        Return a view of the dictionary's keys, same as keys().

        Returns:
            KeysView: The names of the observations.

        """
        return self.keys()

    def entries(self):
        """
        Return a view of the dictionary's values, same as values().

        Returns:
            ValuesView: The observations.

        """
        return self.values()

    def __eq__(self, other):
        """Two containers are equal if they hold the same names and the same observation types in the same order."""
        if not isinstance(other, ObservationContainer):
            return False

        keys_are_the_same = set(self.keys()) == set(other.keys())
        type_self = [type(v) for v in self.values()]
        type_other = [type(v) for v in other.values()]
        types_are_the_same = type_self == type_other

        return keys_are_the_same and types_are_the_same

    def list_all_stateful(self):
        """
        Returns a list of all stateful observations in the container.

        Returns:
            List[StatefulObservation]: The stateful observations, in the order of insertion.

        """
        return self._stateful_obs

    def get_all_stateful_indices(self):
        """
        Get the indices of all stateful observations in the container.

        Returns:
            np.ndarray: The indices of the stateful observations.

        """
        return np.concatenate([obs.obs_ind for obs in self._stateful_obs]).astype(int)

    def init_state(self, env, key, model, data, backend, traj=None):
        """
        Builds a dataclass from the states of the stateful observations in the container.

        Args:
            env (Any): The environment instance.
            key (jax.Array): JAX random key.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            backend (ModuleType): Backend module, either ``numpy`` or ``jax.numpy``.
            traj (Trajectory, optional): The trajectory of the environment.

        Returns:
            Any: Instance of a dynamically created flax dataclass with one field per stateful observation, named
            like the observation and holding its initial state.

        """

        # Get all stateful observations
        stateful_obs = self.list_all_stateful()
        # Create a dictionary with the stateful observations
        stateful_obs_dict = {obs.name: obs.init_state(env, key, model, data, backend, traj) for obs in stateful_obs}
        # Dynamically create a class with fields from the dictionary
        dynamic_class = make_dataclass("ObservationStates", stateful_obs_dict.keys())
        # convert to flax dataclass
        dynamic_class = struct.dataclass(dynamic_class, frozen=False)
        # create instance
        return dynamic_class(**stateful_obs_dict)

    def get_all_group_names(self):
        """
        Get all group names in the container.

        Returns:
            List[str]: The list of group names.

        """
        return list(set(group for obs in self.values() for group in obs.group))

    def filter_by_group(self, obs: Union[np.ndarray, jnp.ndarray], group_name: str = None):
        """
        Filter observations by group name.

        Args:
            obs (Union[np.ndarray, jnp.ndarray]): The observation array, shape ``(..., obs_dim)``.
            group_name (str | list[str]): The group name, or several names, to filter by.

        Returns:
            Union[np.ndarray, jnp.ndarray]: The entries of the observation that belong to the group(s).

        """
        obs_ind_group = self.get_obs_ind_by_group(group_name)
        return obs[..., obs_ind_group]

    def get_obs_ind_by_group(self, group_name=None):
        """
        Get the indices of the observations by group name.

        Args:
            group_name (str | list[str]): A group name, or several. With several, the
                groups are concatenated IN THE ORDER GIVEN, and within each group the
                observations keep container order. That order is part of the network's
                input layout, so changing it invalidates existing checkpoints.

                Passing overlapping groups duplicates the shared indices, which is
                usually a mistake -- keep groups disjoint, or rely on the fact that a
                term meant to be read by two consumers (e.g. last_action) can simply be
                listed in both and requested by only one of them at a time.

        Returns:
            np.ndarray: The indices of the observations.

        """
        groups = [group_name] if isinstance(group_name, str) else list(group_name)
        obs_ind_group = [obs.obs_ind for g in groups
                         for obs in self.values() if g in obs.group]
        return np.concatenate(obs_ind_group) if len(obs_ind_group) > 0 else np.array([])

    def get_randomizable_obs_indices(self):
        """
        Get the indices of the observations that are allowed to be randomized.

        Returns:
            np.ndarray: The indices of the observations.

        """
        return np.concatenate([obs.obs_ind for obs in self.values() if obs.allow_randomization]).astype(int)

    def reset_state(self, env, model, data, carry, backend, traj=None):
        """
        Reset the states of all stateful observations in the container.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Additional carry.
            backend (ModuleType): Backend module, either ``numpy`` or ``jax.numpy``.
            traj (Trajectory, optional): The trajectory of the environment.

        Returns:
            Tuple[Union[MjData, Data], Any]: The updated simulation data and carry.

        """
        # Get all stateful observations
        stateful_obs = self.list_all_stateful()
        # Reset the state of each stateful observation
        for obs in stateful_obs:
             data, carry = obs.reset_state(env, model, data, carry, backend, traj)
        return data, carry

    def lock(self):
        """Lock the container to prevent further modifications."""
        self._locked = True


class Observation:
    """
    Base class for all observation types.

    Args:
        obs_name (str): The name of the observation.
        group (Union[str, List[str]]): The group name(s) of the observation. Groups are used to select parts
            of the observation, for example for the actor and the critic.
        allow_randomization (bool): Whether the observation is allowed to be randomized.

    Attributes:
        registered (dict): Registry of observation classes by name.
        name (str): The name of the observation.
        obs_container (ObservationContainer): The container the observation was added to.
        group (List[str]): The groups the observation belongs to.
        allow_randomization (bool): Whether the observation is allowed to be randomized.
        obs_ind (np.ndarray): Indices of this observation in the observation vector.
        data_type_ind (np.ndarray): Indices of this observation in the respective attribute of the Mujoco data
            structure.
        min (list): Lower limits of the observation.
        max (list): Upper limits of the observation.

    """

    registered = dict()

    def __init__(self, obs_name: str, group: Union[str, List(str)] = None, allow_randomization: bool = True):
        self.name = obs_name
        self.obs_container = None
        self.group = [group] if isinstance(group, str) or group is None else group
        self.allow_randomization = allow_randomization

        # these attributes *must* be initialized in the _init_from_mj method
        self.obs_ind = None
        self.data_type_ind = None
        self.min, self.max = None, None
        self._initialized_from_mj = False

    def init_from_mj(self, env, model, data, current_obs_size, data_ind_cont: ObservationIndexContainer,
                     obs_ind_cont: ObservationIndexContainer):
        """
        Initialize the observation type from the Mujoco data structure and model.

        Args:
            env (Any): Environment instance.
            model (MjModel): The Mujoco model.
            data (MjData): The Mujoco data structure.
            current_obs_size (int): The current size of the observation space, i.e. the index at which this
                observation starts.
            data_ind_cont (ObservationIndexContainer): The data indices container.
            obs_ind_cont (ObservationIndexContainer): The observation indices container.

        """
        # extract all information from data and model
        self._init_from_mj(env, model, data, current_obs_size)

        # store the indices in the ObservationIndexContainer
        self._add_to_data_and_obs_cont(data_ind_cont, obs_ind_cont)

    def _init_from_mj(self, env, model, data, current_obs_size):
        """
        Initialize the observation type from the Mujoco data structure and model.
        This method *must* initialize the following attributes:

        - obs_ind: Indices of this observation in the observation space.
        - data_type_ind: Indices of this observation in respective attribute
          in the Mujoco data structure (like qpos, qvel, xpos, etc.).
        - min: Minimum values of the observation in the observation space.
        - max: Maximum values of the observation in the observation space.

        Args:
            env (Any): Environment instance.
            model (MjModel): The Mujoco model.
            data (MjData): The Mujoco data structure.
            current_obs_size (int): The current size of the observation space, i.e. the index at which this
                observation starts.

        """
        raise NotImplementedError

    def _add_to_data_and_obs_cont(self, data_ind_cont: ObservationIndexContainer,
                                  obs_ind_cont: ObservationIndexContainer):
        """
        Adds the indices corresponding to this observation to the specified
        `ObservationIndexContainer` for both the MuJoCo data structure and the
        observation itself.

        Args:
            data_ind_cont (ObservationIndexContainer): The container holding
                the indices for the MuJoCo data structure.
            obs_ind_cont (ObservationIndexContainer): The container holding
                the indices for the observation.
        """

        # get the obs type name
        obs_type_name = self.__class__.__name__

        # store the indices in the ObservationIndexContainer
        data_ind_cont_attr = getattr(data_ind_cont, obs_type_name)
        obs_ind_cont_attr = getattr(obs_ind_cont, obs_type_name)
        data_ind_cont_attr.extend(deepcopy(self.data_type_ind.tolist()))
        obs_ind_cont_attr.extend(deepcopy(self.obs_ind.tolist()))

    @classmethod
    def get_all_obs_of_type(cls, env, model, data, data_ind_cont, backend):
        """
        Default getter for all the observations of this type from the Mujoco data structure.

        Args:
            env (Any): The environment.
            model (Union[MjModel, Model]): The Mujoco model.
            data (Union[MjData, Data]): The Mujoco data structure.
            data_ind_cont (ObservationIndexContainer): The indices of *all* observations of all types.
            backend (ModuleType): The backend to use, either ``numpy`` or ``jax.numpy``.

        Returns:
            Union[np.ndarray, jax.Array]: The flattened values of all observations of this type.

        """
        obs_type_name = cls.__name__
        data_type_name = cls.data_type()

        assert data_type_name is not None, (f"Observation type {cls.__name__} does not have a default data_type. "
                                            f"Default get_obs method can not be used, please implement"
                                            f" a dedicated one.")

        # get the attribute from the mujoco data structure
        data_attr = getattr(data, data_type_name)
        # get the indices of all observations of this type
        data_ind = getattr(data_ind_cont, obs_type_name)
        # return the observation
        return backend.ravel(data_attr[data_ind])

    @classmethod
    def data_type(cls):
        """
        Attribute name in the mujoco data structure. If this is provided, the default get_obs method can be used.
        If not provided, a dedicated get_obs method must be implemented.

        Returns:
            str: The attribute name in the Mujoco data structure, or None.

        """
        return None

    @property
    def initialized_from_mj(self):
        """
        Whether the observation has been initialized from a Mujoco model and data.

        Returns:
            bool: True after the indices and limits have been set.

        """
        return self._initialized_from_mj

    @staticmethod
    def to_list(val):
        """
        Convert the input to a list of integers.

        Args:
            val (Union[int, np.ndarray]): An integer or an integer array.

        Returns:
            List[int]: The input as a list.

        Raises:
            ValueError: If the input is neither an integer nor an integer array.

        """
        if isinstance(val, int):
            return [val]
        elif isinstance(val, np.ndarray) and val.dtype == int:
            return val.tolist()
        else:
            raise ValueError("Input must be an integer or a numpy array of integers")

    @classmethod
    def register(cls):
        """Register the observation class in the registry and as an attribute of :class:`ObservationType`."""
        obs_type_name = cls.__name__

        if obs_type_name not in Observation.registered:
            Observation.registered[obs_type_name] = cls

        # register it also in observation type class
        ObservationType.register(cls)

    @staticmethod
    def list_registered():
        """
        List registered Observations.

        Returns:
            List[str]: The names of the registered observation classes.

        """
        return list(Observation.registered.keys())


class SimpleObs(Observation):
    """
    Base class of observations that read the quantities of a single named element of the model.

    Args:
        obs_name (str): The name of the observation.
        xml_name (str): The name of the element (joint, body, site, ...) in the Mujoco XML.
        **kwargs: Further arguments of :class:`Observation` (``group`` and ``allow_randomization``).

    See also:
        :class:`Observation` for the base observation class.

    """

    def __init__(self, obs_name: str, xml_name: str, **kwargs):
        self.xml_name = xml_name
        super().__init__(obs_name, **kwargs)


class StatefulObservation(Observation, StatefulObject):
    """
    Base class of observations that are computed by a function instead of being read from the data structure.

    A stateful observation implements :meth:`get_obs_and_update_state`, which has access to the environment, the
    model, the data and the carry, and may keep its own state in the carry
    (see :class:`~loco_mujoco.core.stateful_object.StatefulObject`).

    """

    def init_from_mj(self, env, model, data, current_obs_size, data_ind_cont: ObservationIndexContainer,
                     obs_ind_cont: ObservationIndexContainer):
        """
        Initialize the observation type from the Mujoco data structure and model.

        Args:
            env (Any): Environment instance.
            model (MjModel): The Mujoco model.
            data (MjData): The Mujoco data structure.
            current_obs_size (int): The current size of the observation space, i.e. the index at which this
                observation starts.
            data_ind_cont (ObservationIndexContainer): The data indices container (unused).
            obs_ind_cont (ObservationIndexContainer): The observation indices container (unused).

        """
        # extract all information from data and model
        self._init_from_mj(env, model, data, current_obs_size)

    @classmethod
    def get_all_obs_of_type(cls, env, model, data, data_ind_cont, backend):
        """Not supported by stateful observations; raises ``NotImplementedError``."""
        raise NotImplementedError("Stateful observations do not support this function.")

    def get_obs_and_update_state(self, env, model, data, carry, backend, traj=None):
        """
        Get the observation and update the state.

        Args:
            env (Any): The environment.
            model (Union[MjModel, Model]): The Mujoco model.
            data (Union[MjData, Data]): The Mujoco data structure.
            carry (Any): The additional carry.
            backend (ModuleType): The backend to use, either ``numpy`` or ``jax.numpy``.
            traj (Trajectory, optional): Trajectory to use.

        Returns:
            Tuple[Union[np.ndarray, jax.Array], Any]: The observation and the updated carry.

        """
        raise NotImplementedError


class BodyPos(SimpleObs):
    """
    Observation Type holding x, y, z position of the body.

    Args:
        obs_name (str): The name of the observation.
        xml_name (str): The name of the body in the Mujoco XML.
        **kwargs: Further arguments of :class:`Observation` (``group`` and ``allow_randomization``).

    Attributes:
        dim (int): Dimension of the observation.

    See also:
        :class:`Observation` for the base observation class.

    """

    dim = 3

    def _init_from_mj(self, env, model, data, current_obs_size):
        """Set the indices of this observation in the observation vector and in ``data.xpos`` as well as its limits."""
        dim = len(data.body(self.xml_name).xpos)
        assert dim == self.dim
        self.min, self.max = [-np.inf] * dim, [np.inf] * dim
        self.data_type_ind = np.array(self.to_list(data.body(self.xml_name).id))
        self.obs_ind = np.array([j for j in range(current_obs_size, current_obs_size + dim)])
        self._initialized_from_mj = True

    @classmethod
    def data_type(cls):
        """
        Name of the attribute of the Mujoco data structure this observation is read from.

        Returns:
            str: ``"xpos"``.

        """
        return "xpos"


class BodyRot(SimpleObs):
    """
    Observation Type holding the quaternion of the body.

    Args:
        obs_name (str): The name of the observation.
        xml_name (str): The name of the body in the Mujoco XML.
        **kwargs: Further arguments of :class:`Observation` (``group`` and ``allow_randomization``).

    Attributes:
        dim (int): Dimension of the observation.

    See also:
        :class:`Observation` for the base observation class.

    """

    dim = 4

    def _init_from_mj(self, env, model, data, current_obs_size):
        """Set the indices of this observation in the observation vector and in ``data.xquat`` as well as its limits."""
        dim = len(data.body(self.xml_name).xquat)
        assert dim == self.dim
        self.min, self.max = [-np.inf] * dim, [np.inf] * dim
        self.data_type_ind = np.array(self.to_list(data.body(self.xml_name).id))
        self.obs_ind = np.array([j for j in range(current_obs_size, current_obs_size + dim)])
        self._initialized_from_mj = True

    @classmethod
    def data_type(cls):
        """
        Name of the attribute of the Mujoco data structure this observation is read from.

        Returns:
            str: ``"xquat"``.

        """
        return "xquat"


class BodyVel(SimpleObs):
    """
    Observation Type holding the angular velocity around x, y, z and the linear velocity for x, y, z.

    Args:
        obs_name (str): The name of the observation.
        xml_name (str): The name of the body in the Mujoco XML.
        **kwargs: Further arguments of :class:`Observation` (``group`` and ``allow_randomization``).

    Attributes:
        dim (int): Dimension of the observation.

    See also:
        :class:`Observation` for the base observation class.

    """

    dim = 6

    def _init_from_mj(self, env, model, data, current_obs_size):
        """Set the indices of this observation in the observation vector and in ``data.cvel`` as well as its limits."""
        dim = len(data.body(self.xml_name).cvel)
        assert dim == self.dim
        self.min, self.max = [-np.inf] * dim, [np.inf] * dim
        self.data_type_ind = np.array(self.to_list(data.body(self.xml_name).id))
        self.obs_ind = np.array([j for j in range(current_obs_size, current_obs_size + dim)])
        self._initialized_from_mj = True

    @classmethod
    def data_type(cls):
        """
        Name of the attribute of the Mujoco data structure this observation is read from.

        Returns:
            str: ``"cvel"``.

        """
        return "cvel"


class FreeJointPos(SimpleObs):
    """
    Observation Type holding the 3D position and the 4D quaternion of a free joint.

    Args:
        obs_name (str): The name of the observation.
        xml_name (str): The name of the free joint in the Mujoco XML.
        **kwargs: Further arguments of :class:`Observation` (``group`` and ``allow_randomization``).

    Attributes:
        dim (int): Dimension of the observation.

    See also:
        :class:`Observation` for the base observation class.

    """

    dim = 7

    def _init_from_mj(self, env, model, data, current_obs_size):
        """Set the indices of this observation in the observation vector and in ``data.qpos`` as well as its limits."""
        # note: free joints do not have limits
        self.min, self.max = [-np.inf] * FreeJointPos.dim, [np.inf] * FreeJointPos.dim
        self.data_type_ind = np.array(mj_jntname2qposid(self.xml_name, model))
        self.obs_ind = np.array([j for j in range(current_obs_size, current_obs_size + FreeJointPos.dim)])
        self._initialized_from_mj = True

    @classmethod
    def data_type(cls):
        """
        Name of the attribute of the Mujoco data structure this observation is read from.

        Returns:
            str: ``"qpos"``.

        """
        return "qpos"


class EntryFromFreeJointPos(FreeJointPos):
    """
    Observation Type holding *a single entry* of a free joint pose.

    Args:
        entry_index (int): Index of the entry in the 7D pose (position, then quaternion).
        **kwargs: Arguments of :class:`FreeJointPos`.

    See also:
        :class:`Observation` for the base observation class.

    """

    dim = 1

    def __init__(self, entry_index: int, **kwargs):
        assert type(entry_index) == int, "entry_index must be an integer."
        self._entry_index = entry_index
        super().__init__(**kwargs)

    def _init_from_mj(self, env, model, data, current_obs_size):
        """Initialize like :class:`FreeJointPos` and keep only the selected entry of the pose."""
        super()._init_from_mj(None, model, data, current_obs_size)
        self.min, self.max = [self.min[self._entry_index]], [self.max[self._entry_index]]
        self.data_type_ind = np.array([self.data_type_ind[self._entry_index]])
        self.obs_ind = np.array([self.obs_ind[0]])
        self._initialized_from_mj = True

    @classmethod
    def data_type(cls):
        """
        Name of the attribute of the Mujoco data structure this observation is read from.

        Returns:
            str: ``"qpos"``.

        """
        return "qpos"


class FreeJointPosNoXY(FreeJointPos):
    """
    Observation Type holding the height and the 4D quaternion of a free joint pose, neglecting the x and y position.

    Args:
        obs_name (str): The name of the observation.
        xml_name (str): The name of the free joint in the Mujoco XML.
        **kwargs: Further arguments of :class:`Observation` (``group`` and ``allow_randomization``).

    Attributes:
        dim (int): Dimension of the observation.

    See also:
        :class:`Observation` for the base observation class.

    """

    dim = 5

    def _init_from_mj(self, env, model, data, current_obs_size):
        """Initialize like :class:`FreeJointPos` and drop the x and y position."""
        super()._init_from_mj(None, model, data, current_obs_size)
        self.min, self.max = self.min[2:], self.max[2:]
        self.data_type_ind = self.data_type_ind[2:]
        self.obs_ind = self.obs_ind[:-2]
        self._initialized_from_mj = True

    @classmethod
    def data_type(cls):
        """
        Name of the attribute of the Mujoco data structure this observation is read from.

        Returns:
            str: ``"qpos"``.

        """
        return "qpos"


class JointPos(SimpleObs):
    """
    Observation Type holding the position of a joint with a single degree of freedom.

    The limits are taken from the joint range, if the joint is limited.

    Args:
        obs_name (str): The name of the observation.
        xml_name (str): The name of the joint in the Mujoco XML.
        **kwargs: Further arguments of :class:`Observation` (``group`` and ``allow_randomization``).

    Attributes:
        dim (int): Dimension of the observation.

    See also:
        :class:`Observation` for the base observation class.

    """

    dim = 1

    def _init_from_mj(self, env, model, data, current_obs_size):
        """Set the indices of this observation in the observation vector and in ``data.qpos`` as well as its limits."""
        dim = len(data.joint(self.xml_name).qpos)
        assert dim == self.dim
        jh = model.joint(mj_jnt_name2id(self.xml_name, model))
        if jh.limited:
            self.min, self.max = [jh.range[0]], [jh.range[1]]
        else:
            self.min, self.max = [-np.inf] * dim, [np.inf] * dim
        self.data_type_ind = np.array(mj_jntid2qposid(jh.id, model))
        self.obs_ind = np.array([j for j in range(current_obs_size, current_obs_size + dim)])
        self._initialized_from_mj = True

    @classmethod
    def data_type(cls):
        """
        Name of the attribute of the Mujoco data structure this observation is read from.

        Returns:
            str: ``"qpos"``.

        """
        return "qpos"


class JointPosArray(Observation):
    """
    Observation Type holding the positions of an array of joints.

    Args:
        obs_name (str): The name of the observation.
        xml_names (List[str]): The names of the joints in the Mujoco XML.
        **kwargs: Further arguments of :class:`Observation` (``group`` and ``allow_randomization``).

    See also:
        :class:`Observation` for the base observation class.

    """

    def __init__(self, obs_name: str, xml_names: List[str], **kwargs):
        super().__init__(obs_name, **kwargs)
        self._xml_names = xml_names
        self.dim = None

    def _init_from_mj(self, env, model, data, current_obs_size):
        """Set the indices of this observation in the observation vector and in ``data.qpos`` as well as its limits."""
        dim = 0
        self.min, self.max, self.data_type_ind, self.obs_ind = [], [], [], []
        for name in self._xml_names:
            sdim = len(data.joint(name).qpos)
            jh = model.joint(mj_jnt_name2id(name, model))
            if jh.limited:
                min, max = [jh.range[0]] * sdim, [jh.range[1]] * sdim
            else:
                min, max = [-np.inf] * sdim, [np.inf] * sdim
            self.min.extend(min)
            self.max.extend(max)
            self.data_type_ind.extend(mj_jntid2qposid(jh.id, model))
            self.obs_ind.extend([j for j in range(current_obs_size, current_obs_size + sdim)])
            current_obs_size += sdim
            dim += sdim
        self.dim = dim
        self.data_type_ind = np.array(self.data_type_ind)
        self.obs_ind = np.array(self.obs_ind)
        self._initialized_from_mj = True

    @classmethod
    def data_type(cls):
        """
        Name of the attribute of the Mujoco data structure this observation is read from.

        Returns:
            str: ``"qpos"``.

        """
        return "qpos"


class FreeJointVel(SimpleObs):
    """
    Observation Type holding the 3D linear velocity and the 3D angular velocity of a free joint.

    Note: Different to the :class:`BodyVel` observation type!

    Args:
        obs_name (str): The name of the observation.
        xml_name (str): The name of the free joint in the Mujoco XML.
        **kwargs: Further arguments of :class:`Observation` (``group`` and ``allow_randomization``).

    Attributes:
        dim (int): Dimension of the observation.

    See also:
        :class:`Observation` for the base observation class.

    """

    dim = 6

    def _init_from_mj(self, env, model, data, current_obs_size):
        """Set the indices of this observation in the observation vector and in ``data.qvel`` as well as its limits."""
        dim = len(data.joint(self.xml_name).qvel)
        # note: free joints do not have limits
        self.min, self.max = [-np.inf] * dim, [np.inf] * dim
        self.data_type_ind = np.array(mj_jntname2qvelid(self.xml_name, model))
        self.obs_ind = np.array([j for j in range(current_obs_size, current_obs_size + dim)])
        self._initialized_from_mj = True

    @classmethod
    def data_type(cls):
        """
        Name of the attribute of the Mujoco data structure this observation is read from.

        Returns:
            str: ``"qvel"``.

        """
        return "qvel"


class EntryFromFreeJointVel(FreeJointVel):
    """
    Observation Type holding a single entry of a free joint velocity.

    Args:
        entry_index (int): Index of the entry in the 6D velocity (linear, then angular).
        **kwargs: Arguments of :class:`FreeJointVel`.

    See also:
        :class:`Observation` for the base observation class.

    """
    dim = 1

    def __init__(self, entry_index: int, **kwargs):
        assert type(entry_index) == int, "entry_index must be an integer."
        self._entry_index = entry_index
        super().__init__(**kwargs)

    def _init_from_mj(self, env, model, data, current_obs_size):
        """Initialize like :class:`FreeJointVel` and keep only the selected entry of the velocity."""
        super()._init_from_mj(None, model, data, current_obs_size)
        self.min, self.max = [self.min[self._entry_index]], [self.max[self._entry_index]]
        self.data_type_ind = np.array([self.data_type_ind[self._entry_index]])
        self.obs_ind = np.array([self.obs_ind[0]])

    @classmethod
    def data_type(cls):
        """
        Name of the attribute of the Mujoco data structure this observation is read from.

        Returns:
            str: ``"qvel"``.

        """
        return "qvel"


class JointVel(SimpleObs):
    """
    Observation Type holding the velocity of the joint.

    Args:
        obs_name (str): The name of the observation.
        xml_name (str): The name of the joint in the Mujoco XML.
        **kwargs: Further arguments of :class:`Observation` (``group`` and ``allow_randomization``).

    Attributes:
        dim (int): Dimension of the observation.

    See also:
        :class:`Observation` for the base observation class.

    """

    dim = 1

    def _init_from_mj(self, env, model, data, current_obs_size):
        """Set the indices of this observation in the observation vector and in ``data.qvel`` as well as its limits."""
        dim = len(data.joint(self.xml_name).qvel)
        assert dim == self.dim
        self.min, self.max = [-np.inf] * dim, [np.inf] * dim
        self.data_type_ind = np.array([model.jnt_dofadr[data.joint(self.xml_name).id]])
        self.obs_ind = np.array([j for j in range(current_obs_size, current_obs_size + dim)])
        self._initialized_from_mj = True

    @classmethod
    def data_type(cls):
        """
        Name of the attribute of the Mujoco data structure this observation is read from.

        Returns:
            str: ``"qvel"``.

        """
        return "qvel"


class JointVelArray(Observation):
    """
    Observation Type holding the velocity of an array of joints.

    Args:
        obs_name (str): The name of the observation.
        xml_names (List[str]): The names of the joints in the Mujoco XML.
        **kwargs: Further arguments of :class:`Observation` (``group`` and ``allow_randomization``).

    See also:
        :class:`Observation` for the base observation class.

    """

    def __init__(self, obs_name: str, xml_names: List[str], **kwargs):
        super().__init__(obs_name, **kwargs)
        self._xml_names = xml_names
        self.dim = None

    def _init_from_mj(self, env, model, data, current_obs_size):
        """Set the indices of this observation in the observation vector and in ``data.qvel`` as well as its limits."""
        dim = 0
        self.min, self.max, self.data_type_ind, self.obs_ind = [], [], [], []
        for name in self._xml_names:
            sdim = len(data.joint(name).qvel)
            jh = model.joint(mj_jnt_name2id(name, model))
            min, max = [-np.inf] * sdim, [np.inf] * sdim
            self.min.extend(min)
            self.max.extend(max)
            self.data_type_ind.extend(mj_jntid2qvelid(jh.id, model))
            self.obs_ind.extend([j for j in range(current_obs_size, current_obs_size + sdim)])
            current_obs_size += sdim
            dim += sdim
        self.dim = dim
        self.data_type_ind = np.array(self.data_type_ind)
        self.obs_ind = np.array(self.obs_ind)
        self._initialized_from_mj = True

    @classmethod
    def data_type(cls):
        """
        Name of the attribute of the Mujoco data structure this observation is read from.

        Returns:
            str: ``"qvel"``.

        """
        return "qvel"


class SitePos(SimpleObs):
    """
    Observation Type holding the x, y, z position of the site.

    Args:
        obs_name (str): The name of the observation.
        xml_name (str): The name of the site in the Mujoco XML.
        **kwargs: Further arguments of :class:`Observation` (``group`` and ``allow_randomization``).

    Attributes:
        dim (int): Dimension of the observation.

    See also:
        :class:`Observation` for the base observation class.

    """

    dim = 3

    def _init_from_mj(self, env, model, data, current_obs_size):
        """
        Set the indices of this observation in the observation vector and in ``data.site_xpos`` as well as its limits.

        """
        dim = len(data.site(self.xml_name).xpos)
        assert dim == self.dim
        self.min, self.max = [-np.inf] * dim, [np.inf] * dim
        self.data_type_ind = np.array(self.to_list(data.site(self.xml_name).id))
        self.obs_ind = np.array([j for j in range(current_obs_size, current_obs_size + dim)])
        self._initialized_from_mj = True

    @classmethod
    def data_type(cls):
        """
        Name of the attribute of the Mujoco data structure this observation is read from.

        Returns:
            str: ``"site_xpos"``.

        """
        return "site_xpos"


class SiteRot(SimpleObs):
    """
    Observation Type holding the flattened rotation matrix of the site.

    Args:
        obs_name (str): The name of the observation.
        xml_name (str): The name of the site in the Mujoco XML.
        **kwargs: Further arguments of :class:`Observation` (``group`` and ``allow_randomization``).

    Attributes:
        dim (int): Dimension of the observation.

    See also:
        :class:`Observation` for the base observation class.

    """

    dim = 9

    def _init_from_mj(self, env, model, data, current_obs_size):
        """
        Set the indices of this observation in the observation vector and in ``data.site_xmat`` as well as its limits.

        """
        # Sites don't have rotation quaternion for some reason...
        # x_mat is rotation matrix with shape (9, )
        dim = len(data.site(self.xml_name).xmat)
        assert dim == self.dim
        self.min, self.max = [-np.inf] * dim, [np.inf] * dim
        self.data_type_ind = np.array(self.to_list(data.site(self.xml_name).id))
        self.obs_ind = np.array([j for j in range(current_obs_size, current_obs_size + dim)])
        self._initialized_from_mj = True

    @classmethod
    def data_type(cls):
        """
        Name of the attribute of the Mujoco data structure this observation is read from.

        Returns:
            str: ``"site_xmat"``.

        """
        return "site_xmat"


class ProjectedGravityVector(Observation):
    """
    Observation Type holding the gravity direction expressed in the frame of a free joint.

    The unit vector ``(0, 0, -1)`` is rotated with the inverse of the orientation of the free joint.

    Args:
        obs_name (str): The name of the observation.
        xml_name (str): The name of the free joint in the Mujoco XML to calculate the gravity vector from.
        **kwargs: Further arguments of :class:`Observation` (``group`` and ``allow_randomization``).

    See also:
        :class:`Observation` for the base observation class.

    """

    dim = 3

    def __init__(self, obs_name: str, xml_name: str, **kwargs):
        self.xml_name = xml_name
        super().__init__(obs_name, **kwargs)

    def _init_from_mj(self, env, model, data, current_obs_size):
        """
        Set the observation indices, the indices of the quaternion of the free joint in ``data.qpos`` and the limits.

        """
        self.min, self.max = [-np.inf] * self.dim, [np.inf] * self.dim
        self.data_type_ind = np.array(mj_jntname2qposid(self.xml_name, model))[3:]  # only the quaternion part is needed
        self.obs_ind = np.array([j for j in range(current_obs_size, current_obs_size + self.dim)])
        self._initialized_from_mj = True

    @classmethod
    def get_all_obs_of_type(cls, env, model, data, data_ind_cont, backend):
        """
        Default getter for all the projected gravity vectors from the Mujoco data structure.

        Args:
            env (Any): The environment.
            model (Union[MjModel, Model]): The Mujoco model.
            data (Union[MjData, Data]): The Mujoco data structure.
            data_ind_cont (ObservationIndexContainer): The indices of *all* observations of all types.
            backend (ModuleType): The backend to use, either ``numpy`` or ``jax.numpy``.

        Returns:
            Union[np.ndarray, jax.Array]: The flattened projected gravity vectors of all observations of this type.

        """

        if backend == np:
            R = np_R
        elif backend == jnp:
            R = jnp_R
        else:
            raise ValueError(f"Unknown backend {backend}.")

        if data_ind_cont.ProjectedGravityVector.size == 0:
            return backend.empty(shape=(0,))
        else:
            xquats = data.qpos[data_ind_cont.ProjectedGravityVector]
            xquats = xquats.reshape(-1, 4)
            rots = R.from_quat(quat_scalarfirst2scalarlast(xquats))

            # get the gravity vector from the quaternions
            proj_grav = rots.inv().apply(np.array([0, 0, -1]))

            # return the observation
            return backend.ravel(proj_grav)


class Force(Observation):
    """
    Observation Type holding the collision forces/torques [3D force + 3D torque]
    between two geoms.

    The forces are only computed in the CPU (MuJoCo) backend; in the MJX backend the observation is zero.

    Args:
        obs_name (str): The name of the observation.
        xml_name_geom1 (str): The name of the first geom in the Mujoco XML.
        xml_name_geom2 (str): The name of the second geom in the Mujoco XML.
        **kwargs: Further arguments of :class:`Observation` (``group`` and ``allow_randomization``).

    See also:
        :class:`Observation` for the base observation class.

    """

    dim = 6

    def __init__(self, obs_name: str, xml_name_geom1: str, xml_name_geom2: str, **kwargs):
        self.xml_name_geom1 = xml_name_geom1
        self.xml_name_geom2 = xml_name_geom2
        super().__init__(obs_name, **kwargs)

        self.mjx_contact_id = None
        self.data_geom_id1 = None
        self.data_geom_id2 = None

    def _init_from_mj(self, env, model, data, current_obs_size):
        """Look up the ids of the two geoms and set the observation indices and limits."""
        # get all required information from data
        self.min, self.max = [-np.inf] * self.dim, [np.inf] * self.dim
        self.data_geom_id1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, self.xml_name_geom1)
        self.data_geom_id2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, self.xml_name_geom2)
        self.data_type_ind = np.array([self.data_geom_id1, self.data_geom_id2])
        self.obs_ind = np.array([j for j in range(current_obs_size, current_obs_size + self.dim)])
        self._initialized_from_mj = True

    @classmethod
    def get_all_obs_of_type(cls, env, model, data, data_ind_cont, backend):
        """
        Getter for the collision forces of all observations of this type.

        Args:
            env (Any): The environment.
            model (Union[MjModel, Model]): The Mujoco model.
            data (Union[MjData, Data]): The Mujoco data structure.
            data_ind_cont (ObservationIndexContainer): The indices of *all* observations of all types.
            backend (ModuleType): The backend to use, either ``numpy`` or ``jax.numpy``.

        Returns:
            Union[np.ndarray, jax.Array]: The flattened forces and torques, 6 entries per observation.

        """
        ind = data_ind_cont.Force
        if backend == np:
            return backend.ravel(cls.mj_collision_force(model, data, ind))
        elif backend == jnp:
            return backend.ravel(cls.mjx_collision_force(model, data, ind))
        else:
            raise ValueError(f"Unknown backend {backend}.")

    @staticmethod
    def mj_collision_force(model, data, ind):
        """
        Compute the contact forces between pairs of geoms with MuJoCo.

        Args:
            model (MjModel): The Mujoco model.
            data (MjData): The Mujoco data structure.
            ind (np.ndarray): Pairs of geom ids, shape ``(n_pairs, 2)``. A contact is only matched if its geoms are
                in the same order as in the pair.

        Returns:
            np.ndarray: Contact force and torque in the contact frame (``mujoco.mj_contactForce``) for every pair,
            shape ``(n_pairs, 6)``; zero for pairs that are not in contact.

        """

        c_array = np.zeros((len(ind), 6), dtype=np.float64)
        for i, geom_ids in enumerate(ind):

            for con_i in range(0, data.ncon):
                con = data.contact[con_i]
                con_geom_ids = (con.geom1, con.geom2)

                if np.array_equal(geom_ids, con_geom_ids):
                    mujoco.mj_contactForce(model, data,
                                           con_i, c_array[i])

        return c_array

    @staticmethod
    def mjx_collision_force(model, data, ind):
        """
        Placeholder for the contact forces in MJX, which are not implemented yet.

        Args:
            model (Model): The Mujoco model.
            data (Data): The Mujoco data structure.
            ind (np.ndarray): Pairs of geom ids, shape ``(n_pairs, 2)``.

        Returns:
            np.ndarray: Zeros of shape ``(n_pairs, 6)``.

        """
        # will be added once mjx adds the collision force function to the official release
        c_array = np.zeros((len(ind), 6), dtype=np.float64)
        return c_array


class LastAction(StatefulObservation):
    """
    Observation Type holding the action of the previous step, taken from ``carry.last_action``.

    Args:
        obs_name (str): The name of the observation.
        **kwargs: Further arguments of :class:`Observation` (``group`` and ``allow_randomization``).

    """

    def __init__(self, obs_name: str, **kwargs):
        super().__init__(obs_name, **kwargs)
        self.dim = None

    def _init_from_mj(self, env, model, data, current_obs_size):
        """Set the dimension to the action dimension of the environment, and the observation indices and limits."""
        self.dim = env.action_dim
        self.min, self.max = [-np.inf] * self.dim, [np.inf] * self.dim
        self.obs_ind = np.array([j for j in range(current_obs_size, current_obs_size + self.dim)])
        self._initialized_from_mj = True

    def get_obs_and_update_state(self, env, model, data, carry, backend, traj=None):
        """
        Get the last action from the carry.

        Args:
            env (Any): The environment.
            model (Union[MjModel, Model]): The Mujoco model.
            data (Union[MjData, Data]): The Mujoco data structure.
            carry (Any): The additional carry.
            backend (ModuleType): The backend to use, either ``numpy`` or ``jax.numpy``.
            traj (Trajectory, optional): Trajectory to use.

        Returns:
            Tuple[Union[np.ndarray, jax.Array], Any]: The observation and the unchanged carry.

        """
        return backend.ravel(carry.last_action), carry


class ModelInfo(StatefulObservation):
    """
    Observation Type holding attributes of the model, for example quantities that are randomized.

    Args:
        obs_name (str): The name of the observation.
        model_attributes (Union[str, List[str]]): Name(s) of the attributes of the Mujoco model to observe. The
            flattened attributes are concatenated in the given order.
        **kwargs: Further arguments of :class:`Observation` (``group`` and ``allow_randomization``).

    """

    def __init__(self, obs_name: str, model_attributes: Union[str, List[str]], **kwargs):
        self._model_attributes = [model_attributes] if isinstance(model_attributes, str) else model_attributes
        super().__init__(obs_name, **kwargs)
        self.dim = None

    def _init_from_mj(self, env, model, data, current_obs_size):
        """
        Set the dimension to the total size of the observed model attributes, and the observation indices and limits.

        """
        dim = 0
        for attr in self._model_attributes:
            dim += getattr(model, attr).size
        self.dim = dim
        self.min, self.max = [-np.inf] * self.dim, [np.inf] * self.dim
        self.obs_ind = np.array([j for j in range(current_obs_size, current_obs_size + self.dim)])
        self._initialized_from_mj = True

    def get_obs_and_update_state(self, env, model, data, carry, backend, traj=None):
        """
        Get the flattened model attributes.

        Args:
            env (Any): The environment.
            model (Union[MjModel, Model]): The Mujoco model.
            data (Union[MjData, Data]): The Mujoco data structure.
            carry (Any): The additional carry.
            backend (ModuleType): The backend to use, either ``numpy`` or ``jax.numpy``.
            traj (Trajectory, optional): Trajectory to use.

        Returns:
            Tuple[Union[np.ndarray, jax.Array], Any]: The observation and the unchanged carry.

        """
        obs = []
        for attr in self._model_attributes:
            obs.append(backend.ravel(getattr(model, attr)))

        return backend.concatenate(obs), carry


class RelativeSiteQuantaties(StatefulObservation):
    """
    Observation Type holding the position, rotation and velocity of all sites for mimic relatively to the main site.
    This observation type is typically included in the observation space of imitation learning algorithms like
    AMP or DeepMimic, where the agent is supposed to mimic the trajectory of a reference agent.

    For every site except the main site, the observation holds 3 entries for the relative position, 3 for the
    relative rotation and 6 for the relative velocity.

    Args:
        obs_name (str): The name of the observation.
        site_names (List[str]): List of site names to calculate the relative quantities from. The first site in
            the list is considered as the main site. If None, the environment's sites for mimic are used instead.
        **kwargs: Further arguments of :class:`Observation` (``group`` and ``allow_randomization``).

    See also:
        :class:`Observation` for the base observation class.

    """

    def __init__(self, obs_name: str, site_names: List[str] = None, **kwargs):
        self.site_names = site_names

        # will be initialized in the _init_from_mj method
        self.dim = None
        self.rel_site_ids = None
        self.site_bodyid = None
        self.body_rootid = None
        super().__init__(obs_name, **kwargs)

    def _init_from_mj(self, env, model, data, current_obs_size):
        """Look up the site ids and set the dimension, observation indices and limits."""
        if self.site_names is None:
            self.site_names = env.sites_for_mimic
            assert len(self.site_names) > 0, "No sites for mimic are defined in the environment."
        self.rel_site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in self.site_names]
        self.rel_site_ids = np.array(self.rel_site_ids)
        self.site_bodyid = model.site_bodyid
        self.body_rootid = model.body_rootid
        n_sites = len(self.site_names) - 1
        self.dim = (3 + 3 + 6) * n_sites    # 3 for position, 3 for rotation, 6 for velocity
        self.min, self.max = [-np.inf] * self.dim, [np.inf] * self.dim
        self.obs_ind = np.array([j for j in range(current_obs_size, current_obs_size + self.dim)])
        self._initialized_from_mj = True

    def get_obs_and_update_state(self, env, model, data, carry, backend, traj=None):
        """
        Get the relative site quantities.

        Args:
            env (Any): The environment.
            model (Union[MjModel, Model]): The Mujoco model.
            data (Union[MjData, Data]): The Mujoco data structure.
            carry (Any): The additional carry.
            backend (ModuleType): The backend to use, either ``numpy`` or ``jax.numpy``.
            traj (Trajectory, optional): Trajectory to use.

        Returns:
            Tuple[Union[np.ndarray, jax.Array], Any]: The observation (all relative positions, then all relative
            rotations, then all relative velocities) and the unchanged carry.

        """

        rel_body_ids = self.site_bodyid[self.rel_site_ids]
        site_rpos, site_rangles, site_rvel = calculate_relative_site_quatities(data, self.rel_site_ids, rel_body_ids,
                                                                               self.body_rootid, backend)

        site_obs = backend.concatenate([
            backend.ravel(site_rpos),
            backend.ravel(site_rangles),
            backend.ravel(site_rvel)
        ])

        return backend.ravel(site_obs), carry


class HeightScan(StatefulObservation):
    """
    Terrain heights on a grid around the root, in the root's heading frame (Isaac Lab / mjlab style height scan).

    Each entry is ``root_height - terrain_height(point) - offset``, clipped to ``clip``. The grid spans ``size``
    (forward, lateral) metres with ``resolution`` spacing, centred ``forward_offset`` metres ahead of the root.
    The heights come from the terrain's ``get_height_at``, so the observation works with all built-in terrains,
    also in jitted code on MJX and MjWarp.

    Args:
        obs_name (str): The name of the observation.
        size (Sequence[float]): Grid extent (forward, lateral) in metres.
        resolution (float): Grid spacing in metres.
        forward_offset (float): Shift of the grid centre along the heading, in metres.
        offset (float): Subtracted from every entry (e.g. the nominal root height), in metres.
        clip (Sequence[float]): (low, high) clipping range of the output.
        xml_name (str): Name of the free joint giving the root pose. Defaults to the env's root free joint.
        differentiable (bool): Let gradients flow through the scan into the root pose. Off by default: for
            methods that backpropagate through the observation (e.g. SuperMimic's BPTT through a world model),
            step edges turn the scan into near-discontinuities whose gradients with respect to the root position
            are spurious.
        **kwargs: Further arguments of :class:`Observation` (``group`` and ``allow_randomization``).

    """

    def __init__(self, obs_name: str, size: Sequence[float] = (1.6, 1.0), resolution: float = 0.1,
                 forward_offset: float = 0.0, offset: float = 0.0, clip: Sequence[float] = (-1.0, 1.0),
                 xml_name: str = None, differentiable: bool = False, **kwargs):
        super().__init__(obs_name, **kwargs)
        self._differentiable = differentiable
        nx = int(round(size[0] / resolution)) + 1
        ny = int(round(size[1] / resolution)) + 1
        gx = np.linspace(-size[0] / 2, size[0] / 2, nx) + forward_offset
        gy = np.linspace(-size[1] / 2, size[1] / 2, ny)
        self._grid = np.stack(np.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2)   # (n, 2) in heading frame
        self.dim = len(self._grid)
        self._offset = offset
        self._clip = tuple(clip)
        self._xml_name = xml_name

    def _init_from_mj(self, env, model, data, current_obs_size):
        """Look up the indices of the root free joint in ``data.qpos`` and set the observation indices and limits."""
        # the terrain is set up after the observations, so it is only looked up when the observation is computed
        name = env.root_free_joint_xml_name if self._xml_name is None else self._xml_name
        self._qpos_ids = np.array(mj_jntname2qposid(name, model))
        self.min, self.max = [self._clip[0]] * self.dim, [self._clip[1]] * self.dim
        self.obs_ind = np.arange(current_obs_size, current_obs_size + self.dim)
        self._initialized_from_mj = True

    def get_obs_and_update_state(self, env, model, data, carry, backend, traj=None):
        """
        Compute the height scan around the current root pose.

        Args:
            env (Any): The environment.
            model (Union[MjModel, Model]): The Mujoco model.
            data (Union[MjData, Data]): The Mujoco data structure.
            carry (Any): The additional carry.
            backend (ModuleType): The backend to use, either ``numpy`` or ``jax.numpy``.
            traj (Trajectory, optional): Trajectory to use.

        Returns:
            Tuple[Union[np.ndarray, jax.Array], Any]: The scan, shape ``(dim,)`` with the grid flattened forward
            index first, and the unchanged carry.

        """
        q = data.qpos[self._qpos_ids]
        w, x, y, z = q[3], q[4], q[5], q[6]
        yaw = backend.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        c, s = backend.cos(yaw), backend.sin(yaw)
        g = backend.asarray(self._grid)
        pts = backend.stack([q[0] + c * g[:, 0] - s * g[:, 1], q[1] + s * g[:, 0] + c * g[:, 1]], axis=-1)
        h = env._terrain.get_height_at(pts, backend, env=env, carry=carry)
        scan = backend.clip(q[2] - h - self._offset, *self._clip)
        if backend == jnp and not self._differentiable:
            scan = jax.lax.stop_gradient(scan)
        return scan, carry


class ObservationType:
    """
    Namespace for all observation types for easy access.

    Every registered observation class is available as an attribute with the name of the class, for example
    ``ObservationType.JointPos``.

    """
    BodyPos = BodyPos
    BodyRot = BodyRot
    BodyVel = BodyVel
    JointPos = JointPos
    JointVel = JointVel
    JointPosArray = JointPosArray
    JointVelArray = JointVelArray
    FreeJointPos = FreeJointPos
    EntryFromFreeJointPos = EntryFromFreeJointPos
    FreeJointPosNoXY = FreeJointPosNoXY
    FreeJointVel = FreeJointVel
    EntryFromFreeJointVel = EntryFromFreeJointVel
    SitePos = SitePos
    SiteRot = SiteRot
    ProjectedGravityVector = ProjectedGravityVector
    Force = Force
    LastAction = LastAction
    ModelInfo = ModelInfo
    RelativeSiteQuantaties = RelativeSiteQuantaties
    HeightScan = HeightScan

    @classmethod
    def get(cls, obs_name):
        """
        Get an observation type by name.

        Args:
            obs_name (str): The name of the observation type.

        Returns:
            type: The observation type.

        """
        return getattr(cls, obs_name)

    @classmethod
    def register(cls, new_obs_type):
        """
        Register a new observation type.

        Args:
            new_obs_type (type): The new observation type to register.

        Raises:
            TypeError: If the argument is not a class or does not inherit from :class:`Observation`.

        """
        # Check if new_obs_type is a class
        if not isinstance(new_obs_type, type):
            raise TypeError(f"{new_obs_type} must be a class.")

        # Check if new_obs_type inherits from Observation
        if not issubclass(new_obs_type, Observation):
            raise TypeError(f"{new_obs_type.__name__} must inherit from Observation.")

        setattr(cls, new_obs_type.__name__, new_obs_type)

    @classmethod
    def list_all(cls):
        """
        List all observation types.

        Returns:
            List[type]: All registered observation classes.

        """
        return [getattr(ObservationType, obs_type) for obs_type in dir(cls) if not obs_type.startswith("__")
                and obs_type not in ["register", "list_all", "list_all_non_stateful", "get"]]

    @classmethod
    def list_all_non_stateful(cls):
        """
        List all observation types, which are not stateful.

        Returns:
            List[type]: All registered observation classes that do not inherit from :class:`StatefulObservation`.

        """
        return [getattr(ObservationType, obs_type) for obs_type in dir(cls) if not obs_type.startswith("__")
                and obs_type not in ["register", "list_all", "list_all_non_stateful", "get"]
                and not issubclass(getattr(ObservationType, obs_type), StatefulObservation)]
