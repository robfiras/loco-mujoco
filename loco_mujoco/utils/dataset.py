import os
import argparse
from typing import Union, Type

import yaml

import loco_mujoco


VARIABLE_TYPES = {
    "LOCOMUJOCO_AMASS_PATH": "str",
    "LOCOMUJOCO_CONVERTED_AMASS_PATH": "str",
    "LOCOMUJOCO_CONVERTED_DEFAULT_PATH": "str",
    "LOCOMUJOCO_CONVERTED_LAFAN1_PATH": "str",
    "LOCOMUJOCO_CUSTOM_MODELS_PATH": "list",
    "LOCOMUJOCO_SMPL_MODEL_PATH": "str",
    "LOCOMUJOCO_CUSTOM_ROBOT_CONF_PATH": "list"
}


def set_amass_path():
    """
    Set the path to the AMASS dataset.
    """
    parser = argparse.ArgumentParser(description="Set the AMASS dataset path.")
    parser.add_argument("--path", type=str, help="Path to the AMASS dataset.")
    args = parser.parse_args()
    _set_path_in_yaml_conf(args.path, "LOCOMUJOCO_AMASS_PATH", path_to_conf=loco_mujoco.PATH_TO_VARIABLES)


def set_smpl_model_path():
    """
    Set the path to the SMPL model.
    """
    parser = argparse.ArgumentParser(description="Set the SMPL model path.")
    parser.add_argument("--path", type=str, help="Path to the SMPL model.")
    args = parser.parse_args()
    _set_path_in_yaml_conf(args.path, "LOCOMUJOCO_SMPL_MODEL_PATH", path_to_conf=loco_mujoco.PATH_TO_VARIABLES)


def set_all_caches():
    """
    Set the path to which all converted datasets will be stored. This sets the following Variables:
    - LOCOMUJOCO_CONVERTED_AMASS_PATH
    - LOCOMUJOCO_CONVERTED_LAFAN1_PATH
    - LOCOMUJOCO_CONVERTED_DEFAULT_PATH

    Returns:

    """
    parser = argparse.ArgumentParser(description="Set the path to which all converted datasets will be stored.")
    parser.add_argument("--path", type=str, help="Path to which all converted datasets will be stored.")
    args = parser.parse_args()
    amass_path = os.path.join(args.path, "AMASS")
    _set_path_in_yaml_conf(amass_path, "LOCOMUJOCO_CONVERTED_AMASS_PATH",
                           path_to_conf=loco_mujoco.PATH_TO_VARIABLES)
    lafan1_path = os.path.join(args.path, "LAFAN1")
    _set_path_in_yaml_conf(lafan1_path, "LOCOMUJOCO_CONVERTED_LAFAN1_PATH",
                           path_to_conf=loco_mujoco.PATH_TO_VARIABLES)
    default_path = os.path.join(args.path, "DEFAULT")
    _set_path_in_yaml_conf(default_path, "LOCOMUJOCO_CONVERTED_DEFAULT_PATH",
                           path_to_conf=loco_mujoco.PATH_TO_VARIABLES)


def set_converted_amass_path():
    """
    Set the path to which the converted AMASS dataset is stored.
    """
    parser = argparse.ArgumentParser(description="Set the path to which the converted AMASS dataset is stored.")
    parser.add_argument("--path", type=str, help="Path to which the converted AMASS dataset is stored.")
    args = parser.parse_args()
    _set_path_in_yaml_conf(args.path, "LOCOMUJOCO_CONVERTED_AMASS_PATH",
                           path_to_conf=loco_mujoco.PATH_TO_VARIABLES)


def set_lafan1_path():
    """
    Set the path to the LAFAN1 dataset.
    """
    parser = argparse.ArgumentParser(description="Set the LAFAN1 dataset path.")
    parser.add_argument("--path", type=str, help="Path to the LAFAN1 dataset.")
    args = parser.parse_args()
    _set_path_in_yaml_conf(args.path, "LOCOMUJOCO_LAFAN1_PATH", path_to_conf=loco_mujoco.PATH_TO_VARIABLES)


def set_converted_lafan1_path():
    """
    Set the path to which the converted LAFAN1 dataset is stored.
    """
    parser = argparse.ArgumentParser(description="Set the path to which the converted LAFAN1 dataset is stored.")
    parser.add_argument("--path", type=str, help="Path to which the converted LAFAN1 dataset is stored.")
    args = parser.parse_args()
    _set_path_in_yaml_conf(args.path, "LOCOMUJOCO_CONVERTED_LAFAN1_PATH",
                           path_to_conf=loco_mujoco.PATH_TO_VARIABLES)


def add_variable_cli():
    parser = argparse.ArgumentParser(description="Set a variable in LOCOMUJOCO_VARIABLES.")
    parser.add_argument("--name", type=str, help="Name of the variable.")
    parser.add_argument("--value", type=str, help="Value to which the variable is set.")
    parser.add_argument("--type", type=str, default=None,
                        help="Type of the variable, either \"str\" or \"list\".")
    args = parser.parse_args()
    add_variable(args.name, args.value, args.type)


def remove_variable_cli():
    parser = argparse.ArgumentParser(description="Remove a variable in LOCOMUJOCO_VARIABLES.")
    parser.add_argument("--name", type=str, help="Name of the variable.")
    parser.add_argument("--value", type=str, default=None,
                        help="Value to be deleted [Only needed for list type Variables].")
    args = parser.parse_args()
    remove_variable(args.name, args.value)


def show_variable_cli():
    parser = argparse.ArgumentParser(description="Show a variable from LOCOMUJOCO_VARIABLES.")
    parser.add_argument("--name", type=str, default=None,
                        help="Name of the variable (omit and pass --all to show all).")
    parser.add_argument("--all", action="store_true",
                        help="Show all variables currently set.")
    args = parser.parse_args()

    if not args.all and args.name is None:
        parser.error("either --name <VAR> or --all must be provided")

    path_to_conf = loco_mujoco.PATH_TO_VARIABLES
    if not os.path.exists(path_to_conf):
        print(f"No variables file found at {path_to_conf}.")
        return

    with open(path_to_conf, "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader) or {}

    def _print_one(name, value):
        if isinstance(value, list):
            print(f"{name}:")
            for i, v in enumerate(value):
                print(f"  [{i}] {v}")
        else:
            print(f"{name}: {value}")

    if args.all:
        if not data:
            print("(no variables set)")
            return
        for name in sorted(data):
            _print_one(name, data[name])
        return

    if args.name not in data:
        print(f"{args.name} is not set.")
        return
    _print_one(args.name, data[args.name])


def add_variable(
    variable_name: str,
    value: str,
    variable_type: str = None,
    quiet: bool = False,
):
    """
    Adds a variable to LOCOMUJOCO_VARIABLES.yaml.

    Args:
        variable_name (str): Name of the variable.
        value (str): Value of the variable.
        variable_type (str): Type of the variable.
        quiet (bool): If True, no prints.

    """

    if variable_type is None:
        try:
            variable_type = VARIABLE_TYPES[variable_name]
        except KeyError:
            raise KeyError(f"Variable {variable_name} requires a variable type to be defined. Either str or list.")
    else:
        # do not allow different types for predefined variables
        if variable_name in VARIABLE_TYPES.keys():
            predefined_type = VARIABLE_TYPES[variable_name]
            assert variable_type == predefined_type, f"{variable_name} has to be of type {predefined_type}."
        if variable_type not in {"str", "list"}:
            raise ValueError(f"Invalid variable_type: {variable_type}. Must be 'str' or 'list'.")

    assert isinstance(value, str)

    if variable_type == "str":
        _set_path_in_yaml_conf(value, variable_name, quiet=quiet,
                               path_to_conf=loco_mujoco.PATH_TO_VARIABLES)
    else:
        _add_path_to_yaml_list(value, variable_name, quiet=quiet,
                               path_to_conf=loco_mujoco.PATH_TO_VARIABLES)


def remove_variable(variable_name: str,
                    value: str = None):
    """
    Removes a variable from LOCOMUJOCO_VARIABLES.yaml.

    - If the variable is a string, the entire key is removed.
    - If it's a list, a specific value must be provided and will be removed.
      If the list becomes empty, the key is removed.

    Args:
        variable_name (str): Name of the variable to remove.
        value (str, optional): Value to remove from a list. Required if the variable is a list.
    """

    path_to_conf = loco_mujoco.PATH_TO_VARIABLES

    if not os.path.exists(path_to_conf):
        print(f"YAML file {path_to_conf} does not exist.")
        return

    with open(path_to_conf, "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader) or {}

    if variable_name not in data:
        print(f"{variable_name} does not exist in {path_to_conf}.")
        return

    current_value = data[variable_name]

    if isinstance(current_value, str):
        _delete_path_in_yaml_conf(variable_name, path_to_conf)
    elif isinstance(current_value, list):
        if value is None:
            raise ValueError(f"Must provide a value to remove from the list variable '{variable_name}'.")
        _remove_path_from_yaml_list(value, variable_name, path_to_conf)
    else:
        raise TypeError(f"Unsupported type for variable '{variable_name}': {type(current_value).__name__}")


def _set_path_in_yaml_conf(path: str, attr: str, path_to_conf: str, quiet: bool = False):
    """
    Set the path in the yaml configuration file.
    """

    # create an empty yaml file if it does not exist
    if not os.path.exists(path_to_conf):
        with open(path_to_conf, "w") as file:
            yaml.dump({}, file)

    # load yaml file
    with open(path_to_conf, "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    # set the path
    data[attr] = path

    # save the yaml file
    with open(path_to_conf, "w") as file:
        yaml.dump(data, file)

    if not quiet:
        print(f"Set {attr} to {path} in file {path_to_conf}.")


def _delete_path_in_yaml_conf(attr: str, path_to_conf: str):
    """
    Delete the attribute from the yaml configuration file.
    """

    # if file doesn't exist, nothing to delete
    if not os.path.exists(path_to_conf):
        print(f"File {path_to_conf} does not exist. Nothing to delete.")
        return

    # load yaml file
    with open(path_to_conf, "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader) or {}

    # remove attribute if it exists
    if attr in data:
        del data[attr]
        with open(path_to_conf, "w") as file:
            yaml.dump(data, file)
        print(f"Deleted {attr} from file {path_to_conf}.")
    else:
        print(f"{attr} not found in file {path_to_conf}. Nothing to delete.")


def _add_path_to_yaml_list(path: str, attr: str, path_to_conf: str, quiet: bool = False):
    """
    Add a path to a list in the YAML configuration file.
    If the list does not exist, it is created.
    If the path already exists in the list, nothing happens.
    """

    # create an empty yaml file if it does not exist
    if not os.path.exists(path_to_conf):
        with open(path_to_conf, "w") as file:
            yaml.dump({}, file)

    # load yaml file
    with open(path_to_conf, "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader) or {}

    # ensure the attribute is a list (auto-migrate from str if needed)
    if attr not in data:
        data[attr] = []
    elif isinstance(data[attr], str):
        data[attr] = [data[attr]]
    elif not isinstance(data[attr], list):
        raise ValueError(f"The attribute '{attr}' must be a list in the YAML file.")

    # add path if not already present
    if path not in data[attr]:
        data[attr].append(path)
        if not quiet:
            print(f"Added {path} to {attr} in file {path_to_conf}.")
    else:
        if not quiet:
            print(f"{path} already exists under {attr} in file {path_to_conf}.")

    # save the yaml file
    with open(path_to_conf, "w") as file:
        yaml.dump(data, file)


def _remove_path_from_yaml_list(path: str, attr: str, path_to_conf: str):
    """
    Remove a path from a list in the YAML configuration file.
    If the list becomes empty, the attribute is removed.
    """

    if not os.path.exists(path_to_conf):
        print(f"YAML file {path_to_conf} does not exist.")
        return

    with open(path_to_conf, "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader) or {}

    if attr not in data:
        print(f"{attr} does not exist in file {path_to_conf}.")
        return

    if not isinstance(data[attr], list):
        raise ValueError(f"The attribute '{attr}' must be a list in the YAML file.")

    if path not in data[attr]:
        print(f"{path} not found under {attr} in file {path_to_conf}.")
        return

    data[attr].remove(path)

    # remove key entirely if the list becomes empty
    if not data[attr]:
        del data[attr]

    with open(path_to_conf, "w") as file:
        yaml.dump(data, file)

    print(f"Removed {path} from {attr} in file {path_to_conf}.")
