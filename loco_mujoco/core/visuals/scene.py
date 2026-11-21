"""
Functional counterparts of MuJoCo's ``mjvGeom`` and ``mjvScene`` for user-defined visual geoms.

Stateful objects (for example goals) write the geoms they want to visualize into the scene stored in the carry,
and the viewer draws them on top of the simulation.

"""

from typing import Union
from dataclasses import field

import mujoco
import numpy as np
import jax
from jax import numpy as jnp
from flax import struct

from loco_mujoco.core.stateful_object import StatefulObject


@struct.dataclass
class MjvGeom:
    """
    Visual geom, mirroring the fields of MuJoCo's ``mjvGeom`` (text labels are not supported).

    Attributes:
        size (Union[np.ndarray, jnp.ndarray]): Size parameters, shape ``(3,)``.
        type (int): Geom type (``mjtGeom``).
        pos (Union[np.ndarray, jnp.ndarray]): Cartesian position, shape ``(3,)``.
        mat (Union[np.ndarray, jnp.ndarray]): Cartesian orientation as a flattened rotation matrix, shape
            ``(9,)``.
        dataid (int): Mesh, hfield, or plane id; -1: none.
        objtype (int): MuJoCo object type; ``mjOBJ_UNKNOWN`` for decor.
        objid (int): MuJoCo object id; -1 for decor.
        category (int): Visual category.
        matid (int): Material id; -1: no textured material.
        texcoord (int): Whether a mesh or flex geom has texture coordinates.
        segid (int): Segmentation id; -1: not shown.
        rgba (Union[np.ndarray, jnp.ndarray]): Color and transparency, shape ``(4,)``.
        emission (float): Emission coefficient.
        specular (float): Specular coefficient.
        shininess (float): Shininess coefficient.
        reflectance (float): Reflectance coefficient.
        camdist (float): Distance to camera (used by the sorter; set internally).
        modelrbound (float): Geom rbound from the model, 0 if not a model geom (set internally).
        transparent (int): Whether to treat the geom as transparent (set internally).

    """

    # Required
    size: Union[np.ndarray, jnp.ndarray] = field(default_factory=lambda: np.ones(3) * 0.1)  # size parameters
    type: int = 0  # geom type (mjtGeom)

    # Spatial transform
    pos: Union[np.ndarray, jnp.ndarray] = field(default_factory=lambda: np.zeros(3))  # Cartesian position
    mat: Union[np.ndarray, jnp.ndarray] = field(default_factory=lambda: np.eye(3).reshape(-1))  # Cartesian orientation

    # Type info
    dataid: int = -1  # mesh, hfield, or plane id; -1: none
    objtype: int = 0  # mujoco object type; mjOBJ_UNKNOWN for decor
    objid: int = -1  # mujoco object id; -1 for decor
    category: int = 0  # visual category
    matid: int = -1  # material id; -1: no textured material
    texcoord: int = 0  # mesh or flex geom has texture coordinates
    segid: int = 0  # segmentation id; -1: not shown

    # Material properties
    rgba: Union[np.ndarray, jnp.ndarray] = field(default_factory=lambda: np.ones(4))  # color and transparency
    emission: float = 0.0  # emission coefficient
    specular: float = 0.5  # specular coefficient
    shininess: float = 0.5  # shininess coefficient
    reflectance: float = 0.0  # reflectance coefficient

    # label: str = ""  # text label NOT SUPPORTED

    # Transparency rendering (set internally)
    camdist: float = 0.0  # distance to camera (used by sorter)
    modelrbound: float = 0.0  # geom rbound from model, 0 if not model geom
    transparent: int = 0  # treat geom as transparent (mjtByte in C, represented as int here)


@struct.dataclass
class MjvScene:
    """
    Visualization scene. For now, limited to geoms.

    Attributes:
        ngeoms (int): Number of geoms in the scene.
        geoms (MjvGeom): The geoms, stored as one :class:`MjvGeom` whose fields have a leading dimension of size
            ``ngeoms``.

    """
    ngeoms: int
    geoms: MjvGeom

    @classmethod
    def init_n_geoms(cls, n_geoms: int, backend):
        """
        Create a scene with a given number of default geoms.

        Args:
            n_geoms (int): Number of geoms.
            backend (ModuleType): Backend module, either ``numpy`` or ``jax.numpy``.

        Returns:
            MjvScene: The scene, in which every field of the geoms is tiled to ``(n_geoms, field_dim)``.

        """
        geom = MjvGeom()
        geoms = jax.tree.map(lambda x: backend.tile(x, (n_geoms, 1)), geom)
        return cls(n_geoms, geoms)

    @classmethod
    def init_for_all_stateful_objects(cls, backend):
        """
        Create a scene with one slot for every visual geom of every stateful object.

        As a side effect, ``visual_geoms_idx`` of every
        :class:`~loco_mujoco.core.stateful_object.StatefulObject` created so far is set to the indices of its geoms
        in the scene.

        Args:
            backend (ModuleType): Backend module, either ``numpy`` or ``jax.numpy``.

        Returns:
            MjvScene: The scene.

        """
        idx = 0
        for instance in StatefulObject.get_all_instances():
            instance.visual_geoms_idx = np.arange(idx, idx + instance.n_visual_geoms)
            idx += instance.n_visual_geoms

        total_geoms = idx
        return cls.init_n_geoms(total_geoms, backend)
