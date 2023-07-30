from .atlas import Atlas
from .humanoids import HumanoidTorque, HumanoidMuscle, HumanoidTorque4Ages, HumanoidMuscle4Ages


# register environments in mushroom
Atlas.register()
HumanoidTorque.register()
HumanoidMuscle.register()
HumanoidTorque4Ages.register()
HumanoidMuscle4Ages.register()


from gymnasium import register

# register gymnasium wrapper environment
register("LocoMujoco",
         entry_point="loco_mujoco.environments.gymnasium:GymnasiumWrapper"
         )
