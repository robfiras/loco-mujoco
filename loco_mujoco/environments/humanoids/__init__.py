from .atlas import Atlas
from .talos import Talos
from .unitreeH1 import UnitreeH1
from .humanoids import HumanoidTorque, HumanoidMuscle, HumanoidTorque4Ages, HumanoidMuscle4Ages
from .myosuite_humanoid import MyoSuiteHumanoid


# register environments in mushroom
Atlas.register()
Talos.register()
UnitreeH1.register()
HumanoidTorque.register()
HumanoidMuscle.register()
HumanoidTorque4Ages.register()
HumanoidMuscle4Ages.register()
MyoSuiteHumanoid.register()

from gymnasium import register

# register gymnasium wrapper environment
register("LocoMujoco",
         entry_point="loco_mujoco.environments.gymnasium:GymnasiumWrapper"
         )
