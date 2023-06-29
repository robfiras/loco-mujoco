from pathlib import Path
from copy import deepcopy
import numpy as np
from dm_control import mjcf

from mushroom_rl.utils.running_stats import *
from mushroom_rl.utils.mujoco import *

from loco_mujoco.environments import BaseEnv


class Atlas(BaseEnv):
    """
    Mujoco simulation of the Atlas robot.

    """
    def __init__(self, hold_weight=False, weight_mass=None, tmp_dir_name=None, **kwargs):
        """
        Constructor.

        """
        if hold_weight:
            assert tmp_dir_name is not None, "If you want to carry weights, you have to specify a" \
                                             "directory name for the temporary xml-files to be saved."

        xml_path = (Path(__file__).resolve().parent.parent / "data" / "atlas" / "model.xml").as_posix()

        action_spec = ["hip_flexion_r_actuator", "hip_adduction_r_actuator", "hip_rotation_r_actuator",
                       "knee_angle_r_actuator", "ankle_angle_r_actuator", "hip_flexion_l_actuator",
                       "hip_adduction_l_actuator", "hip_rotation_l_actuator", "knee_angle_l_actuator",
                       "ankle_angle_l_actuator"]

        observation_spec = [# ------------- JOINT POS -------------
                            ("q_pelvis_tx", "pelvis_tx", ObservationType.JOINT_POS),
                            ("q_pelvis_tz", "pelvis_tz", ObservationType.JOINT_POS),
                            ("q_pelvis_ty", "pelvis_ty", ObservationType.JOINT_POS),
                            ("q_pelvis_tilt", "pelvis_tilt", ObservationType.JOINT_POS),
                            ("q_pelvis_list", "pelvis_list", ObservationType.JOINT_POS),
                            ("q_pelvis_rotation", "pelvis_rotation", ObservationType.JOINT_POS),
                            ("q_hip_flexion_r", "hip_flexion_r", ObservationType.JOINT_POS),
                            ("q_hip_adduction_r", "hip_adduction_r", ObservationType.JOINT_POS),
                            ("q_hip_rotation_r", "hip_rotation_r", ObservationType.JOINT_POS),
                            ("q_knee_angle_r", "knee_angle_r", ObservationType.JOINT_POS),
                            ("q_ankle_angle_r", "ankle_angle_r", ObservationType.JOINT_POS),
                            ("q_hip_flexion_l", "hip_flexion_l", ObservationType.JOINT_POS),
                            ("q_hip_adduction_l", "hip_adduction_l", ObservationType.JOINT_POS),
                            ("q_hip_rotation_l", "hip_rotation_l", ObservationType.JOINT_POS),
                            ("q_knee_angle_l", "knee_angle_l", ObservationType.JOINT_POS),
                            ("q_ankle_angle_l", "ankle_angle_l", ObservationType.JOINT_POS),

                            # ------------- JOINT VEL -------------
                            ("dq_pelvis_tx", "pelvis_tx", ObservationType.JOINT_VEL),
                            ("dq_pelvis_tz", "pelvis_tz", ObservationType.JOINT_VEL),
                            ("dq_pelvis_ty", "pelvis_ty", ObservationType.JOINT_VEL),
                            ("dq_pelvis_tilt", "pelvis_tilt", ObservationType.JOINT_VEL),
                            ("dq_pelvis_list", "pelvis_list", ObservationType.JOINT_VEL),
                            ("dq_pelvis_rotation", "pelvis_rotation", ObservationType.JOINT_VEL),
                            ("dq_hip_flexion_r", "hip_flexion_r", ObservationType.JOINT_VEL),
                            ("dq_hip_adduction_r", "hip_adduction_r", ObservationType.JOINT_VEL),
                            ("dq_hip_rotation_r", "hip_rotation_r", ObservationType.JOINT_VEL),
                            ("dq_knee_angle_r", "knee_angle_r", ObservationType.JOINT_VEL),
                            ("dq_ankle_angle_r", "ankle_angle_r", ObservationType.JOINT_VEL),
                            ("dq_hip_flexion_l", "hip_flexion_l", ObservationType.JOINT_VEL),
                            ("dq_hip_adduction_l", "hip_adduction_l", ObservationType.JOINT_VEL),
                            ("dq_hip_rotation_l", "hip_rotation_l", ObservationType.JOINT_VEL),
                            ("dq_knee_angle_l", "knee_angle_l", ObservationType.JOINT_VEL),
                            ("dq_ankle_angle_l", "ankle_angle_l", ObservationType.JOINT_VEL)]

        collision_groups = [("floor", ["ground"]),
                            ("foot_r", ["right_foot_back"]),
                            ("front_foot_r", ["right_foot_front"]),
                            ("foot_l", ["left_foot_back"]),
                            ("front_foot_l", ["left_foot_front"])]

        self._hidable_obs = ("positions", "velocities", "foot_forces", "weight")

        self._hold_weight = hold_weight
        self._weight_mass = weight_mass
        self._valid_weights = [0.1, 1.0, 5.0, 10.0]
        if hold_weight:
            xml_handle = mjcf.from_path(xml_path)
            self.add_weight(xml_handle)
            xml_path = self.save_xml_handle(xml_handle, "atas")

        super().__init__(xml_path, action_spec, observation_spec, collision_groups, **kwargs)

    def setup(self, substep_no=None):
        super().setup(substep_no)
        if self._hold_weight:
            if self._weight_mass is None:
                ind = np.random.randint(0, len(self._valid_weights))
                new_weight_mass = self._valid_weights[ind]
                self._model.body("weight").mass = new_weight_mass

                # modify the color of the mass according to the mass
                red_rgba = np.array([[1.0, 0.0, 0.0, 1.0]])
                blue_rgba = np.array([[0.2, 0.0, 1.0, 1.0]])
                interpolation_var = ind / (len(self._valid_weights)-1)
                color = blue_rgba + ((red_rgba - blue_rgba) * interpolation_var)
                geom_id = self._model.body("weight").geomadr[0]
                self._model.geom_rgba[geom_id] = color
            else:
                self._model.body("weight").mass = self._weight_mass

    def _get_observation_space(self):
        low, high = super(Atlas, self)._get_observation_space()
        if self._hold_weight:
            low = np.concatenate([low, [self._valid_weights[0]]])
            high = np.concatenate([high, [self._valid_weights[-1]]])
        return low, high

    def _create_observation(self, obs):
        obs = super(Atlas, self)._create_observation(obs)
        if self._hold_weight:
            weight_mass = deepcopy(self._model.body("weight").mass)
            obs = np.concatenate([obs, weight_mass])
        return obs

    @staticmethod
    def has_fallen(state):
        pelvis_euler = state[1:4]
        pelvis_y_cond = (state[0] < -0.3) or (state[0] > 0.1)
        pelvis_tilt_cond = (pelvis_euler[0] < (-np.pi / 4.5)) or (pelvis_euler[0] > (np.pi / 12))
        pelvis_list_cond = (pelvis_euler[1] < -np.pi / 12) or (pelvis_euler[1] > np.pi / 8)
        pelvis_rot_cond = (pelvis_euler[2] < (-np.pi / 10)) or (pelvis_euler[2] > (np.pi / 10))
        pelvis_condition = (pelvis_y_cond or pelvis_tilt_cond or pelvis_list_cond or pelvis_rot_cond)

        return pelvis_condition

    def get_mask(self, obs_to_hide):
        """ This function returns a boolean mask to hide observations from a fully observable state. """

        if type(obs_to_hide) == str:
            obs_to_hide = (obs_to_hide,)
        assert all(x in self._hidable_obs for x in obs_to_hide), "Some of the observations you want to hide are not" \
                                                                 "supported. Valid observations to hide are %s." \
                                                                 % (self._hidable_obs,)

        pos_dim = 14
        vel_dim = 16
        force_dim = 12  # 3*4

        mask = []
        if "positions" not in obs_to_hide:
            mask += [np.ones(pos_dim, dtype=np.bool)]
        else:
            mask += [np.zeros(pos_dim, dtype=np.bool)]

        if "velocities" not in obs_to_hide:
            mask += [np.ones(vel_dim, dtype=np.bool)]
        else:
            mask += [np.zeros(vel_dim, dtype=np.bool)]

        if self._use_foot_forces:
            if "foot_forces" not in obs_to_hide:
                mask += [np.ones(force_dim, dtype=np.bool)]
            else:
                mask += [np.zeros(force_dim, dtype=np.bool)]
        else:
            assert "foot_forces" not in obs_to_hide, "Creating a mask to hide foot forces without activating " \
                                                     "the latter is not allowed."

        if self._hold_weight:
            if "weight" not in obs_to_hide:
                mask += [np.ones(1, dtype=np.bool)]
            else:
                mask += [np.zeros(1, dtype=np.bool)]
        else:
            assert "weight" not in obs_to_hide, "Creating a mask to hide the carried weight without activating " \
                                                "the latter is not allowed."

        return np.concatenate(mask).ravel()

    def add_weight(self, xml_handle):

        # find pelvis handle
        pelvis = xml_handle.find("body", "pelvis")
        pelvis.add("body", name="weight")
        weight = xml_handle.find("body", "weight")
        weight.add("geom", type="box", size="0.1 0.27 0.1", pos="0.75 0 -0.02", rgba="1.0 0.0 0.0 1.0", mass="0.1")

        # modify the arm orientation
        r_clav = xml_handle.find("body", "r_clav")
        r_clav.quat = [1.0,  0.0, -0.35, 0.0]
        l_clav = xml_handle.find("body", "l_clav")
        l_clav.quat = [0.0, -0.35, 0.0,  1.0]

    def save_xml_handle(self, xml_handle, tmp_dir_name):

        # save new model and return new xml path
        new_model_dir_name = 'new_atlas/' + tmp_dir_name + "/"
        cwd = Path.cwd()
        new_model_dir_path = Path.joinpath(cwd, new_model_dir_name)
        xml_file_name = "modified_atlas.xml"
        mjcf.export_with_assets(xml_handle, new_model_dir_path, xml_file_name)
        new_xml_path = Path.joinpath(new_model_dir_path, xml_file_name)
        return new_xml_path.as_posix()


if __name__ == '__main__':

    env = Atlas(random_start=False, hold_weight=True, tmp_dir_name="new_atlas")
    action_dim = env.info.action_space.shape[0]

    env.reset()
    env.render()

    absorbing = False
    for i in range(100):
        env.reset()
        env.render()
        for j in range(100):
            action = np.random.randn(action_dim)
            nstate, _, absorbing, _ = env.step(action)
            if absorbing:
                break
            env.render()
