from loco_mujoco.environments.humanoids.base_humanoid import BaseHumanoid
from loco_mujoco.environments.humanoids.base_humanoid_4_ages import BaseHumanoid4Ages
from loco_mujoco.environments import ValidTaskConf
from loco_mujoco.utils import check_validity_task_mode_dataset


class HumanoidTorque(BaseHumanoid):
    """
    MuJoCo simulation of a humanoid model with one torque actuator per joint.

    """

    valid_task_confs = ValidTaskConf(tasks=["walk", "run"],
                                     data_types=["real", "perfect"])

    def __init__(self, **kwargs):
        """
        Constructor.

        """

        if "use_muscles" in kwargs.keys():
            assert kwargs["use_muscles"] is False, "Activating muscles in this environment not allowed. "
            del kwargs["use_muscles"]

        super(HumanoidTorque, self).__init__(use_muscles=False, **kwargs)

    @staticmethod
    def generate(task="walk", dataset_type="real", **kwargs):

        check_validity_task_mode_dataset(HumanoidTorque.__name__, task, None, dataset_type,
                                         *HumanoidTorque.valid_task_confs.get_all())

        if dataset_type == "real":
            if task == "walk":
                path = "datasets/humanoids/real/02-constspeed_reduced_humanoid.npz"
            elif task == "run":
                path = "datasets/humanoids/real/05-run_reduced_humanoid.npz"
        elif dataset_type == "perfect":
            if "use_foot_forces" in kwargs.keys():
                assert kwargs["use_foot_forces"] is False
            if "disable_arms" in kwargs.keys():
                assert kwargs["disable_arms"] is True
            if "use_box_feet" in kwargs.keys():
                assert kwargs["use_box_feet"] is True

            if task == "walk":
                path = "datasets/humanoids/perfect/humanoid_torque_walk/perfect_expert_dataset_det.npz"
            elif task == "run":
                path = "datasets/humanoids/perfect/humanoid_torque_run/perfect_expert_dataset_det.npz"

        return BaseHumanoid.generate(HumanoidTorque, path, task, dataset_type, **kwargs)


class HumanoidMuscle(BaseHumanoid):
    """
    MuJoCo simulation of a humanoid model with muscle actuation.

    """

    valid_task_confs = ValidTaskConf(tasks=["walk", "run"],
                                     data_types=["real", "perfect"],
                                     non_combinable=[("run", None, "perfect")])

    def __init__(self, **kwargs):
        """
        Constructor.

        """

        if "use_muscles" in kwargs.keys():
            assert kwargs["use_muscles"] is True, "Activating torque actuators in this environment not allowed. "
            del kwargs["use_muscles"]

        super(HumanoidMuscle, self).__init__(use_muscles=True, **kwargs)

    @staticmethod
    def generate(task="walk", dataset_type="real", **kwargs):

        check_validity_task_mode_dataset(HumanoidMuscle.__name__, task, None, dataset_type,
                                         *HumanoidMuscle.valid_task_confs.get_all())

        if dataset_type == "real":
            if task == "walk":
                path = "datasets/humanoids/real/02-constspeed_reduced_humanoid.npz"
            elif task == "run":
                path = "datasets/humanoids/real/05-run_reduced_humanoid.npz"
        elif dataset_type == "perfect":
            if "use_foot_forces" in kwargs.keys():
                assert kwargs["use_foot_forces"] is False
            if "disable_arms" in kwargs.keys():
                assert kwargs["disable_arms"] is True
            if "use_box_feet" in kwargs.keys():
                assert kwargs["use_box_feet"] is True

            if task == "walk":
                path = "datasets/humanoids/perfect/humanoid_muscle_walk/perfect_expert_dataset_det.npz"

        return BaseHumanoid.generate(HumanoidMuscle, path, task, dataset_type, **kwargs)


class HumanoidTorque4Ages(BaseHumanoid4Ages):
    """
    MuJoCo simulation of 4 simplified humanoid models with one torque actuator per joint.
    At the beginning of each episode, one of the four humanoid models are
    sampled and used to simulate a trajectory. The different humanoids should
    resemble an adult, a teenager (∼12 years), a child (∼5 years), and a
    toddler (∼1-2 years). This environment can be partially observable by
    using state masks to hide the humanoid type indicator from the policy.

    """

    def __init__(self, **kwargs):
        """
        Constructor.

        """

        if "use_muscles" in kwargs.keys():
            assert kwargs["use_muscles"] is False, "Activating muscles in this environment not allowed. "
            del kwargs["use_muscles"]

        super(HumanoidTorque4Ages, self).__init__(use_muscles=False, **kwargs)

    @staticmethod
    def generate(*args, **kwargs):
        return BaseHumanoid4Ages.generate(HumanoidTorque4Ages, *args, **kwargs)


class HumanoidMuscle4Ages(BaseHumanoid4Ages):
    """
    MuJoCo simulation of 4 simplified humanoid models with muscle actuation.
    At the beginning of each episode, one of the four humanoid models are
    sampled and used to simulate a trajectory. The different humanoids should
    resemble an adult, a teenager (∼12 years), a child (∼5 years), and a
    toddler (∼1-2 years). This environment can be partially observable by
    using state masks to hide the humanoid type indicator from the policy.

    """

    def __init__(self, **kwargs):
        """
                Constructor.

        """

        if "use_muscles" in kwargs.keys():
            assert kwargs["use_muscles"] is True, "Activating torque actuators in this environment not allowed. "
            del kwargs["use_muscles"]

        super(HumanoidMuscle4Ages, self).__init__(use_muscles=True, **kwargs)

    @staticmethod
    def generate(*args, **kwargs):
        return BaseHumanoid4Ages.generate(HumanoidMuscle4Ages, *args, **kwargs)
