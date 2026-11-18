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
        return LocoEnv.registered_envs

    def get_variable(key):
        try:
            return yaml.load(open(PATH_TO_VARIABLES), Loader=yaml.FullLoader)[key]
        except KeyError:
            return None

except ImportError as e:
    print(e)
