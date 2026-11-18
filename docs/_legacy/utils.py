import inspect
import loco_mujoco
from loco_mujoco.environments import HumanoidTorque, HumanoidMuscle, UnitreeA1, UnitreeG1, MyoSkeleton


def make_table(grid):
    max_cols = [max(out) for out in map(list, zip(*[[len(item) for item in row] for row in grid]))]
    rst = table_div(max_cols, 1)

    for i, row in enumerate(grid):
        header_flag = False
        if i == 0 or i == len(grid)-1: header_flag = True
        rst += normalize_row(row,max_cols)
        rst += table_div(max_cols, header_flag )
    return rst


def table_div(max_cols, header_flag=1):
    out = ""
    if header_flag == 1:
        style = "="
    else:
        style = "-"

    for max_col in max_cols:
        out += max_col * style + " "

    out += "\n"
    return out


def normalize_row(row, max_cols):
    r = ""
    for i, max_col in enumerate(max_cols):
        r += row[i] + (max_col  - len(row[i]) + 1) * " "

    return r + "\n"


def get_obs_space_table_docs(env, additional_info, remove=None):

    if remove is None:
        remove = []

    obs_spec = env.obs_helper.observation_spec[2:] # remove the first two
    low = env.info.observation_space.low
    high = env.info.observation_space.high
    if type(env) == UnitreeA1:
        low, high = low[:-2], high[:-2]
    assert len(obs_spec) == len(low) == len(high)

    linear_joints = ["pelvis_tx", "pelvis_tz", "pelvis_ty"]

    # Get the constructor signature
    signature = inspect.signature(env.__init__)

    # Extract default values from the signature
    default_args = {
        param.name: param.default
        for param in signature.parameters.values()
        if param.default is not inspect.Parameter.empty
    }

    if "disable_arms" in default_args.keys():
        env._disable_arms = default_args["disable_arms"]
    elif type(env) == HumanoidTorque or type(env) == HumanoidMuscle:
        env._disable_arms = True
        env._use_box_feet = True
    if "disable_back_joint" in default_args.keys():
        env._disable_back_joint = default_args["disable_back_joint"]

    try:
        joints_to_remove, _, _ = env._get_spec_modifications()
    except:
        if type(env) != UnitreeA1:
            joints_to_remove, _, _, _ = env._get_spec_modifications()
        else:
            joints_to_remove = []

    header = ["Index", "Description", "Min", "Max", "Disabled", "Dim", "Units"]
    grid = [header]
    n_on_by_default = 0
    for i, d in enumerate(zip(obs_spec, low, high)):
        obs, l, h = d
        if obs[1] not in remove:
            is_linear = True if obs[1] in linear_joints else False
            is_joint_vel = True if obs[2] == ObservationType.JOINT_VEL else False
            desc = "Velocity of Joint " + obs[1] if is_joint_vel else "Position of Joint " + obs[1]
            if is_linear and not is_joint_vel:
                unit = "Position [m]"
            elif is_linear and is_joint_vel:
                unit = "Velocity [m/s]"
            elif not is_linear and not is_joint_vel:
                unit = "Angle [rad]"
            else:
                unit = "Angular Velocity [rad/s]"
            row = []
            row.append(str(i))
            row.append(desc)
            row.append(str(l))  # min default for all envs
            row.append(str(h))  # max default for all envs
            if obs[1] in joints_to_remove:
                row.append("True")
            else:
                row.append("False")
                n_on_by_default += 1
            row.append("1") # dim
            row.append(unit)
            grid.append(row)

    curr_len = len(grid)-1
    i = 0
    # append additional info
    for info in additional_info:
        row = []
        row.append(str(i+curr_len))
        row.append(info[0])
        row.append(info[1])  # min default for all envs
        row.append(info[2])  # max default for all envs
        row.append(info[3])
        row.append(info[4])
        i += int(info[4])
        row.append(info[5])
        grid.append(row)

    return make_table(grid), n_on_by_default



def get_obs_space_table_docsv2(env_name):

    env_cls = loco_mujoco.get_registered_envs()[env_name]
    env = env_cls()

    header = ["Index in Obs", "Name", "ObservationType",  "Min", "Max", "Dim"]
    grid = [header]
    for obs_name, obs_info in env.obs_container.items():
        row = []
        obs_dim = obs_info.obs_ind
        if obs_dim.size > 0:
            obs_dims = str(obs_dim[0]) + " - " + str(obs_dim[-1]) if len(obs_dim) > 1 else str(obs_dim[0])
            row.append(obs_dims)
            row.append(obs_info.name)
            row.append(type(obs_info).__name__)

            # limit min and max to 7 values, if more than 7 values, show the first 3 and last 3 values and ... in between
            if len(obs_info.min) > 7:
                row.append(str(obs_info.min[:3])[:-1] + " ... " + str(obs_info.min[-3:])[1:])
                row.append(str(obs_info.max[:3])[:-1] + " ... " + str(obs_info.max[-3:])[1:])
            else:
                row.append(str(obs_info.min))
                row.append(str(obs_info.max))

            row.append(str(obs_info.dim))
            grid.append(row)

    return make_table(grid)

def get_action_space_table_docsv2(env_name):

    env_cls = loco_mujoco.get_registered_envs()[env_name]
    env = env_cls()

    ctrl_func_type = env._control_func.__class__.__name__
    print(f"Control function type: **{env._control_func.__class__.__name__}**")
    print("See control function interface for more details.")

    header = ["Index in Action", "Min", "Max"]
    grid = [header]
    h, l = env.info.action_space.high, env.info.action_space.low
    for i in range(len(h)):
        row = []
        row.append(str(i))
        row.append(str(l[i]))
        row.append(str(h[i]))
        grid.append(row)

    return make_table(grid)


def get_action_space_table_docs(env, use_muscles=False):

    try:
        action_spec = env._get_action_specification()
        if type(env) != UnitreeA1:
            _, motors_to_remove, _ = env._get_spec_modifications()
        else:
            motors_to_remove = []
    except :
        if type(env) == UnitreeG1 or type(env) == MyoSkeleton:
            action_spec = env._get_action_specification()
        else:
            action_spec = env._get_action_specification(use_muscles)
        _, motors_to_remove, _, _ = env._get_spec_modifications()

    header = ["Index", "Name in XML", "Control Min", "Control Max", "Disabled"]
    grid = [header]
    n_on_by_default = 0
    for i, a in enumerate(action_spec):
        row = []
        row.append(str(i))
        row.append(a)   # action name in xml
        row.append("-1.0")  # control min default for all envs
        row.append("1.0")  # control max default for all envs
        if a in motors_to_remove:
            row.append("True")
        else:
            row.append("False")
            n_on_by_default += 1
        grid.append(row)
    return make_table(grid), n_on_by_default


if __name__ == "__main__":

    """
    This file is used to auto-generate the observation space and action space tables used in the documentation.
    """

    print(get_obs_space_table_docsv2("Atlas"))
    print(get_action_space_table_docsv2("Atlas"))


