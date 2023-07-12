from .atlas import Atlas
from .humanoid_muscle import HumanoidMuscle
from .humanoid_torque import HumanoidTorque
from .humanoid_torque_4_ages import HumanoidTorque4Ages

# register environments
Atlas.register()
HumanoidTorque.register()
HumanoidTorque4Ages.register()
