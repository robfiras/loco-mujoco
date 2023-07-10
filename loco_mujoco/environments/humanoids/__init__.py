from .atlas import Atlas
from .full_humanoid_muscle import FullHumanoid
from .reduced_humanoid_torque import ReducedHumanoidTorque
from .reduced_humanoid_torque_4_ages import ReducedHumanoidTorque4Ages

# register environments
Atlas.register()
ReducedHumanoidTorque.register()
ReducedHumanoidTorque4Ages.register()
