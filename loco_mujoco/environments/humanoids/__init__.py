from .atlas import Atlas
from .humanoid_muscle import HumanoidMuscle
from .humanoid_torque import HumanoidTorque
from .humanoid_torque_4_ages import HumanoidTorque4Ages

# register environments in mushroom
Atlas.register()
HumanoidTorque.register()
HumanoidTorque4Ages.register()

from gymnasium import register

# register gymnasium wrapper environment
register("LocoMujoco",
         entry_point="loco_mujoco.environments.gymnasium:GymnasiumWrapper"
         )
