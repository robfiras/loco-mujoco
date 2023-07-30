from loco_mujoco.environments.humanoids.base_humanoid import BaseHumanoid
from loco_mujoco.environments.humanoids.base_humanoid_4_ages import BaseHumanoid4Ages


class HumanoidTorque(BaseHumanoid):
    """
    MuJoCo simulation of a humanoid model with one torque actuator per joint.

    """

    def __init__(self, **kwargs):
        """
        Constructor.

        """

        if "use_muscles" in kwargs.keys():
            assert kwargs["use_muscles"] is False, "Activating muscles in this environment not allowed. "
            del kwargs["use_muscles"]

        super(HumanoidTorque, self).__init__(use_muscles=False, **kwargs)

    @staticmethod
    def generate(**kwargs):
        return BaseHumanoid.generate(HumanoidTorque, **kwargs)


class HumanoidMuscle(BaseHumanoid):
    """
    MuJoCo simulation of a humanoid model with muscle actuation.

    """

    def __init__(self, **kwargs):
        """
        Constructor.

        """

        if "use_muscles" in kwargs.keys():
            assert kwargs["use_muscles"] is True, "Activating torque actuators in this environment not allowed. "
            del kwargs["use_muscles"]

        super(HumanoidMuscle, self).__init__(use_muscles=True, **kwargs)

    @staticmethod
    def generate(**kwargs):
        return BaseHumanoid.generate(HumanoidMuscle, **kwargs)


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
    def generate(**kwargs):
        return BaseHumanoid4Ages.generate(HumanoidTorque4Ages, **kwargs)


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
    def generate(**kwargs):
        return BaseHumanoid4Ages.generate(HumanoidMuscle4Ages, **kwargs)
