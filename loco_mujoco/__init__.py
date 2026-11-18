"""
LocoMuJoCo: an imitation learning benchmark for whole-body control.

The top-level package exports the simulation base classes (:class:`~loco_mujoco.core.mujoco_base.Mujoco`,
:class:`~loco_mujoco.core.mujoco_mjx.Mjx`), the environment base class
(:class:`~loco_mujoco.environments.base.LocoEnv`) and the task factories
(:class:`~loco_mujoco.task_factories.base.TaskFactory`, :class:`~loco_mujoco.task_factories.rl_factory.RLFactory`,
:class:`~loco_mujoco.task_factories.imitation_factory.ImitationFactory`).

User-specific settings such as dataset and cache paths are stored in the file ``LOCOMUJOCO_VARIABLES.yaml`` inside
the package directory (``PATH_TO_VARIABLES``) and can be read with :func:`get_variable`.
"""

from pathlib import Path
import os
import yaml

__version__ = '1.2.0'


try:
    PATH_TO_VARIABLES = Path(__file__).resolve().parent / "LOCOMUJOCO_VARIABLES.yaml"
    PATH_TO_SMPL_ROBOT_CONF = Path(__file__).resolve().parent / "smpl" / "robot_confs"

    # PATH_TO_CUSTOM_MODELS: used for MyoSkeleton and other custom models
    # (e.g. loco-mujoco-myomodel-init). Resolved from LOCOMUJOCO_MODELS_PATH in the yaml.
    def _resolve_path_to_custom_models() -> Path:
        default_path = Path(os.path.expanduser("~")) / ".loco-mujoco-caches" / "models"
        if not PATH_TO_VARIABLES.exists():
            PATH_TO_VARIABLES.parent.mkdir(parents=True, exist_ok=True)
            with open(PATH_TO_VARIABLES, "w") as f:
                yaml.dump({"LOCOMUJOCO_MODELS_PATH": str(default_path)}, f, default_flow_style=False)
            return default_path
        with open(PATH_TO_VARIABLES) as f:
            data = yaml.load(f, Loader=yaml.FullLoader) or {}
        raw = data.get("LOCOMUJOCO_MODELS_PATH")
        if isinstance(raw, list):
            raw = raw[0] if raw else None
        if isinstance(raw, str) and raw.strip():
            return Path(os.path.expanduser(raw.strip()))
        # Use default and persist it in the yaml so the path is always set there
        data["LOCOMUJOCO_MODELS_PATH"] = str(default_path)
        with open(PATH_TO_VARIABLES, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
        return default_path

    PATH_TO_CUSTOM_MODELS = _resolve_path_to_custom_models()

    from .core import Mujoco, Mjx
    from .environments import LocoEnv
    from .task_factories import (TaskFactory, RLFactory, ImitationFactory)

    def get_registered_envs():
        """
        Get all registered environments.

        Returns:
            dict: Mapping from environment name (the class name, e.g. ``"UnitreeH1"`` or ``"MjxUnitreeH1"``) to the
            environment class.

        """
        return LocoEnv.registered_envs

    def get_variable(key):
        """
        Read a variable from the ``LOCOMUJOCO_VARIABLES.yaml`` file.

        Args:
            key (str): Name of the variable, e.g. ``"LOCOMUJOCO_AMASS_PATH"``.

        Returns:
            Union[str, list, None]: Value of the variable, or ``None`` if the variable is not set.

        """
        try:
            return yaml.load(open(PATH_TO_VARIABLES), Loader=yaml.FullLoader)[key]
        except KeyError:
            return None

except ImportError as e:
    print(e)
