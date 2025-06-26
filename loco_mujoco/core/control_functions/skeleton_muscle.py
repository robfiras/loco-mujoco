import jax 
from loco_mujoco.core.control_functions.default import DefaultControl
import mujoco
# from loco_mujoco.environments.base import LocoEnv

class SkeletonMuscleControlFunction(DefaultControl):
    """
    Control function for skeleton muscle control. This controller normalizes the action space to [-1, 1] for the agent
    but uses the original action space for the environment. Before it applies a tanh activation function for torque control
    and a sigmoid activation function for muscle control, depending on the actuator limits.
    """

    def generate_action(self, env, action, model, data, carry, backend):  #env:LocoEnv
        """
        Calculates the action. This function scales the action from [-1, 1] to the original action space.
        """

        # if self._actuator_low = -1 and self._actuator_high = 1, then tanh is applied to the action
        # if self._actuator_low = 0 and self._actuator_high = 1, then sigmoid is applied to the action
        # actuator.dyntype

        # jax.debug.print("Original action: {action}", action=action)
        # jax.debug.print("Actuator Dyntype: {actuator_dyntype}", actuator_dyntype=model.actuator_dyntype)
        
        for i in range(model.nu):
            if model.actuator_dyntype[i] == mujoco.mjtDyn.mjDYN_MUSCLE: #Muscle
                # jax.debug.print("Actuator {i} is a Muscle", i=i)
                # apply sigmoid activation function for muscle control
                # action = action.at[i].set(jax.nn.sigmoid(action[i]))
                action = action.at[i].set(self.adapted_sigmoid(action[i]))
            # else: # Motor 
            #     # jax.debug.print("Actuator {i} is a Motor", i=i)
            #     # apply tanh activation function for motor control
            #     action = action.at[i].set(jax.nn.tanh(action[i]))

        # jax.debug.print("Normalized action after applying fnc: {action}", action=action)
        
        # # unnormalize the action
        # unnormalized_action = self._unnormalize_action(action)

        # jax.debug.print("Unnormalized action: {unnormalized_action}", unnormalized_action=unnormalized_action)

        # # check if normalized action is within the limits
        # action_within_limits = jax.numpy.all(jax.numpy.logical_and(
        #     unnormalized_action >= self._actuator_low,
        #     unnormalized_action <= self._actuator_high
        # ))
        
        # print action within limit or not 
        # jax.debug.print("Action within limits: {action_within_limits}", action_within_limits=action_within_limits)

        # if not action_within_limits:
        #     raise ValueError("Action is out of bounds of the actuator limits.")

        return action, carry  #unnormalized_action
    

    def adapted_sigmoid(self, action):
        """
        Applies a sigmoid activation function to the action, adapted to the actuator limits.
        The sigmoid has a constant for a steeper slope 
        """
        q = 5
        return 1 / (1 + jax.numpy.exp(-q * action))
        