

def check_validity_task_mode_dataset(env_name, task=None, mode=None, dataset_type=None,
                                     valid_tasks=None, valid_modes=None, valid_dataset_types=None, non_combineable=None):
    """
    Checks if the chosen environment configuration is valid for the environment generator.
    If some of the args are None, they won't be checked.

    Args:
        env_name (str): Name of the environment.
        task (str): String describing the chosen task.
        mode (str): String describing the chosen mode.
        dataset_type (str): String describing the chosen dataset type.
        valid_tasks (list): List containing all valid task names.
        valid_modes (list): List containing all valid mode names.
        valid_dataset_types (list): List containing all valid dataset type names.
        non_combinable (list): List of tuples ("task", "mode", "dataset_type"),
                which are NOT allowed to be combined. If one of them is None, it is neglected.

    """

    existing_conf = []
    if task is not None:
        existing_conf.append("<task>")
    if mode is not None:
        existing_conf.append("<mode>")
    if dataset_type is not None:
        existing_conf.append("<dataset_type>")
    example_msg = f"\n\nThe general structure for calling the environment {env_name} is:\n{env_name}."
    if existing_conf:
        for c in existing_conf:
            example_msg += c
            if c != existing_conf[-1]:
                example_msg += "."

        example_msg += "\n\n"
        if task is not None:
            example_msg += f"Valid tasks are {valid_tasks}.\n"
        if mode is not None:
            example_msg += f"Valid modes are {valid_modes}.\n"
        if dataset_type is not None:
            example_msg += f"Valid dataset types are {valid_dataset_types}."

    if task is not None and task not in valid_tasks:
        raise ValueError(f"Task \"{task}\" does not exit in the environment {env_name}. Please, choose from "
                         f"{valid_tasks}. {example_msg}")
    elif mode is not None and mode not in valid_modes:
        raise ValueError(f"Mode \"{mode}\" does not exit in the environment {env_name}. Please, choose from "
                         f"{valid_modes}. {example_msg}")
    elif dataset_type is not None and dataset_type not in valid_dataset_types:
        raise ValueError(f"Dataset type \"{dataset_type}\" does not exit in the environment {env_name}. "
                         f"Please, choose from {valid_dataset_types}. {example_msg}")
    elif non_combineable is not None:
        for nc in non_combineable:
            bad_t, bad_m, bad_dt = nc
            if (task == bad_t or bad_t is None) and (mode == bad_m or bad_m is None)\
                    and (dataset_type == bad_dt or bad_dt is None):
                msg = ""
                if task is not None:
                    msg += f"Task \"{task}\""
                if mode is not None:
                    msg += f", mode \"{mode}\", "
                if dataset_type is not None:
                    msg += f" and dataset type \"{dataset_type}\""
                example_msg += "\n\nHere are the combinations that are NOT allowed:"
                for nc_msg in non_combineable:
                    example_msg += f"\n{env_name}"
                    if nc_msg[0] is not None:
                        example_msg += f".{nc_msg[0]}"
                    if nc_msg[1] is not None:
                        example_msg += f".{nc_msg[1]}"
                    if nc_msg[2] is not None:
                        example_msg += f".{nc_msg[2]}"

                raise ValueError(msg + f" are not combineable for "
                                 f"the environment {env_name}. {example_msg}")
