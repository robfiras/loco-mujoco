import jax 
import jax.numpy as jnp
import numpy as np
from loco_mujoco.core.domain_randomizer import DomainRandomizer
from flax import struct

from loco_mujoco.core.utils.backend import assert_backend_is_supported
from mujoco.mjx import Data, Model
from mujoco import MjData, MjModel
import mujoco

from typing import Any, Union, Tuple
from types import ModuleType


from scipy.spatial.transform import Rotation as R
from jax.scipy.spatial.transform import Rotation as jaxR



## ! Update_observation and update_action only take input and pass output without any changes! (Can be used to add noise in the future) 
@struct.dataclass
class ProsthesisRandomizerState:
    """
    Represents the state of the default randomizer.

    """
    prosthesis_joint_stiffness: Union[np.ndarray, jax.Array]
    prosthesis_dof_damping: Union[np.ndarray, jax.Array]
    prosthesis_body_position: Union[np.ndarray, jax.Array]
    prosthesis_body_orientation: Union[np.ndarray, jax.Array]

    # geom_friction: Union[np.ndarray, jax.Array]
    # geom_stiffness: Union[np.ndarray, jax.Array]
    # geom_damping: Union[np.ndarray, jax.Array]
    # base_mass_to_add: float
    # com_displacement: Union[np.ndarray, jax.Array]
    # link_mass_multipliers: Union[np.ndarray, jax.Array]
    # joint_friction_loss: Union[np.ndarray, jax.Array]
    # dof_damping: Union[np.ndarray, jax.Array]
    # joint_armature: Union[np.ndarray, jax.Array]



class ProsthesisRandomizer(DomainRandomizer):
    """
    Randomize prosthesis parameters in environment 

    Gives options to randomize 
    - Stiffness and damping of amputated limb prosthesis
    - #########Mass of amputated limb prosthesis? Intertia? ######
    - Orientation of amputated limb prosthesis
    - Position of amputated limb prosthesis
     
    
    """
    def __init__(self, env, **kwargs): 
        self._init_prosthesis_joint_stiffness = None 
        self._init_prosthesis_dof_damping = None
        self._init_prosthesis_body_position = None
        self._init_prosthesis_body_orientation = None
        
        super().__init__(env, **kwargs)


    # def init_state(self,
    #                env: Any,
    #                key: Any,
    #                model: Union[MjModel, Model],
    #                data: Union[MjData, Data],
    #                backend: ModuleType) -> ProsthesisRandomizerState:
    #     """
    #     Initialize the randomizer state.

    #     Args:
    #         env (Any): The environment instance.
    #         key (Any): Random seed key.
    #         model (Union[MjModel, Model]): The simulation model.
    #         data (Union[MjData, Data]): The simulation data.
    #         backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

    #     Returns:
    #         DefaultRandomizerState: The initialized randomizer state.

    #     """
    #     self._body_indices = {}
    #     self._joint_indices = {}
    #     self._dof_indices = {}

    #     # get stiffness and damping for randomization_joints 
    #     prosthesis_side = self.rand_conf["prosthesis_side"]
    #     joint_names = self.rand_conf["randomization_joints"]

    #     assert prosthesis_side in ["left_side", "right_side"], f"Invalid prosthesis side: {prosthesis_side}. Expected 'left' or 'right'."

    #     if prosthesis_side == "left_side":
    #         joint_names = [name + '_l' for name in joint_names]
    #     elif prosthesis_side == "right_side":
    #         joint_names = [name + '_r' for name in joint_names]

    #     self.joint_names_side = joint_names

    #     # get joint stiffness and damping
    #     # Get mujoco joint indices for the prosthesis joints
    #     for joint_name in joint_names:
    #         # if joint_name in model.joint_names:
    #         idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    #         if idx != -1:
    #             self._joint_indices[joint_name] = idx
        
    #     # Initialize dof_indices
    #     for joint_name, idx in self._joint_indices.items():
    #         dof_adr = model.jnt_dofadr[idx]
    #         self._dof_indices[joint_name] = dof_adr

    #     # For bodies add the correct side and then get positon and orientation 
    #     if prosthesis_side == "left_side":
    #         body_names = [name + '_l' for name in self.rand_conf["randomization_bodies"]]
    #     elif prosthesis_side == "right_side":
    #         body_names = [name + '_r' for name in self.rand_conf["randomization_bodies"]]

    #     for body_name in body_names:
    #         idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    #         self._body_indices[body_name] = idx

    #     return ProsthesisRandomizerState(prosthesis_joint_stiffness=backend.array([10.0] * len(self._joint_indices.values())),
    #                                   prosthesis_dof_damping=backend.array([1.0] * len(self._dof_indices.values())), 
    #                                     # prosthesis_body_position=backend.array([[0.0, 0.0, 0.0]] * len(self._body_indices.values())),
    #                                     # prosthesis_body_orientation=backend.array([[1.0, 0.0, 0.0, 0.0]] * len(self._body_indices.values())
    #                                   )
    
    # def init_state(self,
    #                env: Any,
    #                key: Any,
    #                model: Union[MjModel, Model],
    #                data: Union[MjData, Data],
    #                backend: ModuleType) -> ProsthesisRandomizerState:
    #     """
    #     Initialize the randomizer state.

    #     Args:
    #         env (Any): The environment instance.
    #         key (Any): Random seed key.
    #         model (Union[MjModel, Model]): The simulation model.
    #         data (Union[MjData, Data]): The simulation data.
    #         backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

    #     Returns:
    #         DefaultRandomizerState: The initialized randomizer state.

    #     """
    #     self._body_indices = {}
    #     self._joint_indices = {}
    #     self._dof_indices = {}

    #     # get stiffness and damping for randomization_joints 
    #     prosthesis_side = self.rand_conf["prosthesis_side"]
    #     joint_names = self.rand_conf["randomization_joints"]

    #     assert prosthesis_side in ["left_side", "right_side"], f"Invalid prosthesis side: {prosthesis_side}. Expected 'left' or 'right'."

    #     if prosthesis_side == "left_side":
    #         joint_names = [name + '_l' for name in joint_names]
    #     elif prosthesis_side == "right_side":
    #         joint_names = [name + '_r' for name in joint_names]

    #     self.joint_names_side = joint_names

    #     # get joint stiffness and damping
    #     # Get mujoco joint indices for the prosthesis joints
    #     for joint_name in joint_names:
    #         # if joint_name in model.joint_names:
    #         idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    #         if idx != -1:
    #             self._joint_indices[joint_name] = idx
        
    #     # Initialize dof_indices
    #     for joint_name, idx in self._joint_indices.items():
    #         dof_adr = model.jnt_dofadr[idx]
    #         self._dof_indices[joint_name] = dof_adr

    #     # Get stiffness and damping for those joints
    #     prosthesis_joint_stiffness = backend.array([model.jnt_stiffness[idx] for idx in self._joint_indices.values()])
    #     prosthesis_dof_damping = backend.array([model.dof_damping[idx] for idx in self._dof_indices.values()])


    #     # For bodies add the correct side and then get positon and orientation 
    #     if prosthesis_side == "left_side":
    #         body_names = [name + '_l' for name in self.rand_conf["randomization_bodies"]]
    #     elif prosthesis_side == "right_side":
    #         body_names = [name + '_r' for name in self.rand_conf["randomization_bodies"]]

    #     self.body_names_side = body_names
    #     # Get body indices for the prosthesis bodies
    #     # self.body_indices = []
    #     for body_name in body_names:
    #         idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    #         self._body_indices[body_name] = idx
    #         # else:
    #         #     raise ValueError(f"Body '{body_name}' not found in model.body_names.")
    #     # # Get position and orientation for those bodies
    #     prosthesis_body_position = backend.array([model.body_pos[idx] for idx in self._body_indices.values()])
    #     prosthesis_body_orientation = backend.array([model.body_quat[idx] for idx in self._body_indices.values()])

    #     # assert_backend_is_supported(backend)

            
    #     return ProsthesisRandomizerState(prosthesis_joint_stiffness=prosthesis_joint_stiffness,
    #                                   prosthesis_dof_damping=prosthesis_dof_damping,
    #                                   prosthesis_body_position=prosthesis_body_position,
    #                                   prosthesis_body_orientation=prosthesis_body_orientation,


    #                                 #   base_mass_to_add=0.0,
    #                                 #   com_displacement=backend.array([0.0, 0.0, 0.0]),
    #                                 #   link_mass_multipliers=backend.array([1.0] * (model.nbody-1)), #exclude worldbody
    #                                 #   joint_friction_loss=backend.array([0.0] * (model.nv-6)), #exclude freejoint 6 dofs
    #                                 #   dof_damping=backend.array([0.0] * (model.nv-6)), #exclude freejoint 6 dofs
    #                                 #   joint_armature=backend.array([0.0] * (model.nv-6)), #exclude freejoint 6 dofs
    #                                   )

    def init_state(self,
                   env: Any,
                   key: Any,
                   model: Union[MjModel, Model],
                   data: Union[MjData, Data],
                   backend: ModuleType) -> ProsthesisRandomizerState:
        """
        Initialize the randomizer state.

        Args:
            env (Any): The environment instance.
            key (Any): Random seed key.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            DefaultRandomizerState: The initialized randomizer state.

        """
        self._body_pos_indices = {}
        self._body_quat_indices = {}
        self._joint_indices = {}
        self._dof_indices = {}
        # self._feet_geom_solref_indices = {}

        # get stiffness and damping for randomization_joints 
        prosthesis_side = self.rand_conf["prosthesis_side"]
        joint_names = self.rand_conf["randomization_joint_names"]
        dof_names = self.rand_conf["randomization_dof_names"]
        body_pos_names = self.rand_conf["randomization_body_position_names"]
        body_quat_names = self.rand_conf["randomization_body_orientation_names"]
        # foot_geom_names = self.rand_conf["randomization_foot_geom_solref_names"]
        

        assert prosthesis_side in ["left_side", "right_side"], f"Invalid prosthesis side: {prosthesis_side}. Expected 'left' or 'right'."

        if prosthesis_side == "left_side":
            joint_names = [name + '_l' for name in joint_names]
            dof_names = [name + '_l' for name in dof_names]
            body_pos_names = [name + '_l' for name in body_pos_names]
            body_quat_names = [name + '_l' for name in body_quat_names]
        elif prosthesis_side == "right_side":
            joint_names = [name + '_r' for name in joint_names]
            dof_names = [name + '_r' for name in dof_names]
            body_pos_names = [name + '_r' for name in body_pos_names]
            body_quat_names = [name + '_r' for name in body_quat_names]
        
        # foot_geom_names = [name + '_l' for name in foot_geom_names] + [name + '_r' for name in foot_geom_names]


        # get joint stiffness and damping
        # Get mujoco joint indices for the prosthesis joints
        for joint_name in joint_names:
            # if joint_name in model.joint_names:
            idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if idx != -1:
                self._joint_indices[joint_name] = idx
        
        # Initialize dof_indices
        for dof_name in joint_names:
            idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, dof_name)
            if idx != -1:
                dof_adr = model.jnt_dofadr[idx]
                self._dof_indices[joint_name] = dof_adr

        # Get stiffness and damping for those joints
        prosthesis_joint_stiffness = backend.array([model.jnt_stiffness[idx] for idx in self._joint_indices.values()])
        prosthesis_dof_damping = backend.array([model.dof_damping[idx] for idx in self._dof_indices.values()])


        # Get body indices for the prosthesis bodies
        # self.body_indices = []
        for body_pos_name in body_pos_names:
            idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_pos_name)
            self._body_pos_indices[body_pos_name] = idx


        for body_quat_name in body_quat_names:
            idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_quat_name)
            self._body_quat_indices[body_quat_name] = idx
            
        # # Get position and orientation for those bodies
        prosthesis_body_position = backend.array([model.body_pos[idx] for idx in self._body_pos_indices.values()])
        prosthesis_body_orientation = backend.array([model.body_quat[idx] for idx in self._body_quat_indices.values()])

        # assert_backend_is_supported(backend)

        # # Foot geom indices
        # for foot_geom_name in foot_geom_names:
        #     idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, foot_geom_name)
        #     if idx != -1:
        #         if foot_geom_name.endswith("_l") and prosthesis_side == "left_side":
        #             self._feet_geom_solref_indices["prosthesis_side"] = model.geom_solref[idx]
        #         elif foot_geom_name.endswith("_r") and prosthesis_side == "right_side":
        #             self._feet_geom_solref_indices["prosthesis_side"] = model.geom_solref[idx]
        #         else:
        #             self._feet_geom_solref_indices["other_side"] = model.geom_solref[idx]
        #         # self._feet_geom_solref_indices[foot_geom_name] = model.geom_solref[idx]
        
        # feet_geom_solref = backend.array([model.geom_solref[idx] for idx in self._feet_geom_solref_indices.values()])

        # Map feet geom names to indices, and match side for solref assignment
        # Example: feet_geom_solref_range: {'prosthesis_side': [0.01, 0.1], 'other_side': [0.01, 0.1]}
        # randomization_feet_geom_names: ['foot_box']
        # foot_geom_names will be ['foot_box_l', 'foot_box_r']
        # We want to know which index is prosthesis_side and which is other_side

        # # Build a mapping from geom name to side
        # self._feet_geom_side_map = {}
        # for name in foot_geom_names:
        #     if prosthesis_side == "left_side" and name.endswith("_l"):
        #         self._feet_geom_side_map[name] = "prosthesis_side"
        #     elif prosthesis_side == "right_side" and name.endswith("_r"):
        #         self._feet_geom_side_map[name] = "prosthesis_side"
        #     else:
        #         self._feet_geom_side_map[name] = "other_side"

            
        return ProsthesisRandomizerState(prosthesis_joint_stiffness=prosthesis_joint_stiffness,
                                      prosthesis_dof_damping=prosthesis_dof_damping,
                                      prosthesis_body_position=prosthesis_body_position,
                                      prosthesis_body_orientation=prosthesis_body_orientation,
                                    #   feet_geom_solref= feet_geom_solref


                                    #   base_mass_to_add=0.0,
                                    #   com_displacement=backend.array([0.0, 0.0, 0.0]),
                                    #   link_mass_multipliers=backend.array([1.0] * (model.nbody-1)), #exclude worldbody
                                    #   joint_friction_loss=backend.array([0.0] * (model.nv-6)), #exclude freejoint 6 dofs
                                    #   dof_damping=backend.array([0.0] * (model.nv-6)), #exclude freejoint 6 dofs
                                    #   joint_armature=backend.array([0.0] * (model.nv-6)), #exclude freejoint 6 dofs
                                      )



    def reset(self,
              env: Any,
              model: Union[MjModel, Model],
              data: Union[MjData, Data],
              carry: Any,
              backend: ModuleType) -> Tuple[Union[MjData, Data], Any]:
        """
        Reset the randomizer, applying domain randomization.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[MjData, Data], Any]: The updated simulation data and carry.

        """
        assert_backend_is_supported(backend)
        domain_randomizer_state = carry.domain_randomizer_state

        # jax.debug.print("carry.domain_randomizer_state RESET")

        # Working print 
        # jax.debug.print("domain_randomizer_state org reset: {domain_randomizer_state}", domain_randomizer_state=domain_randomizer_state)

        if backend == np and self._init_prosthesis_joint_stiffness is None:
            self._init_prosthesis_joint_stiffness = model.jnt_stiffness.copy()
            self._init_prosthesis_dof_damping = model.dof_damping.copy()
            self._init_prosthesis_body_position = model.body_pos.copy()
            self._init_prosthesis_body_orientation = model.body_quat.copy()
            # self._init_feet_geom_solref = model.geom_solref.copy()
        # elif backend == jnp:
        #     self._init_prosthesis_joint_stiffness = jnp.array(model.jnt_stiffness)
        #     self._init_prosthesis_dof_damping = jnp.array(model.dof_damping)
        #     self._init_prosthesis_body_position = jnp.array(model.body_pos)
        #     self._init_prosthesis_body_orientation = jnp.array(model.body_quat)

        prosthesis_joint_stiffness, carry = self._sample_joint_stiffness(model, carry, backend)
        prosthesis_dof_damping, carry = self._sample_dof_damping(model, carry, backend)
        prosthesis_body_position, carry = self._sample_geom_position(model, carry, backend)
        prosthesis_body_orientation, carry = self._sample_joint_orientation(model, carry, backend)
        # feet_geom_solref, carry = self._sample_feet_geom_solref(model, carry, backend)


        # print("prosthesis_joint_stiffness: ", prosthesis_joint_stiffness)
        # print("prosthesis_dof_damping: ", prosthesis_dof_damping)

        # jax.debug.print("prosthesis_joint_stiffness: {prosthesis_joint_stiffness}", prosthesis_joint_stiffness=prosthesis_joint_stiffness)
        # jax.debug.print("prosthesis_dof_damping: {prosthesis_dof_damping}", prosthesis_dof_damping=prosthesis_dof_damping)
        # jax.debug.print("prosthesis_body_position: {prosthesis_body_position}", prosthesis_body_position=prosthesis_body_position)
        # jax.debug.print("prosthesis_body_orientation: {prosthesis_body_orientation}", prosthesis_body_orientation=prosthesis_body_orientation)

        carry = carry.replace(domain_randomizer_state=domain_randomizer_state.replace(
                prosthesis_joint_stiffness=prosthesis_joint_stiffness,
                prosthesis_dof_damping=prosthesis_dof_damping,
                prosthesis_body_position=prosthesis_body_position,
                prosthesis_body_orientation=prosthesis_body_orientation,
                # feet_geom_solref = feet_geom_solref
                ))
        
        # jax.debug.print("carry.domain_randomizer_state end of reset: {carry}", carry=carry)

        return data, carry


    def update(self,
               env: Any,
               model: Union[MjModel, Model],
               data: Union[MjData, Data],
               carry: Any,
               backend: ModuleType) -> Tuple[Union[MjModel, Model], Union[MjData, Data], Any]:
        """
        Update the randomizer by applying the state changes to the model.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[MjModel, Model], Union[MjData, Data], Any]: The updated simulation model, data, and carry.

        """

        assert_backend_is_supported(backend)

        domrand_state = carry.domain_randomizer_state
        # jax.debug.print("domrand_state in update: {domrand_state}", domrand_state=domrand_state)
        # print("domrand_state in update: ", domrand_state)

        dof_indices = list(self._dof_indices.values())
        jnt_indices = list(self._joint_indices.values())
        body_pos_indices = list(self._body_pos_indices.values())
        body_quat_indices = list(self._body_quat_indices.values())
        # feet_geom_solref_indices = list(self._feet_geom_solref_indices.values())


        if backend == jnp:
            # Use JAX for randomization
            jnt_stiffness = model.jnt_stiffness.at[jnp.array(jnt_indices)].set(domrand_state.prosthesis_joint_stiffness)
            dof_damping = model.dof_damping.at[jnp.array(dof_indices)].set(domrand_state.prosthesis_dof_damping)
            body_pos = model.body_pos.at[jnp.array(body_pos_indices)].set(domrand_state.prosthesis_body_position) #model.body_pos[jnp.array(body_pos_indices)])# + domrand_state.prosthesis_body_position)
            body_quat = model.body_quat.at[jnp.array(body_quat_indices)].set(domrand_state.prosthesis_body_orientation) #model.body_quat[jnp.array(body_quat_indices)])# + domrand_state.prosthesis_body_orientation)
            # feet_geom_solref = model.geom_solref.at[jnp.array(feet_geom_solref_indices)].set(domrand_state.feet_geom_solref)
        else:
            dof_damping = self._init_prosthesis_dof_damping.copy()
            dof_damping[dof_indices] = domrand_state.prosthesis_dof_damping
            jnt_stiffness = self._init_prosthesis_joint_stiffness.copy()
            jnt_stiffness[jnt_indices] = domrand_state.prosthesis_joint_stiffness
            body_pos = self._init_prosthesis_body_position.copy()
            body_pos[body_pos_indices] = domrand_state.prosthesis_body_position
            body_quat = self._init_prosthesis_body_orientation.copy()
            body_quat[body_quat_indices] = domrand_state.prosthesis_body_orientation
            # feet_geom_solref = self._init_feet_geom_solref.copy()
            # feet_geom_solref[feet_geom_solref_indices] = domrand_state.feet_geom_solref


        # jax.debug.print("model.jnt_stiffness before: {jnt_stiffness}", jnt_stiffness=model.jnt_stiffness)
        # jax.debug.print("model.dof_damping before: {dof_damping}", dof_damping=model.dof_damping)
        # jax.debug.print("model.body_pos before: {body_pos}", body_pos=model.body_pos)
        # # jax.debug.print("model.body_quat before: {body_quat}", body_quat=model.body_quat)

        if self.rand_conf["randomize_prosthesis_dof_damping"]:
            model = self._set_attribute_in_model(model, "dof_damping", dof_damping, backend)
        if self.rand_conf["randomize_prosthesis_joint_stiffness"]:
            model = self._set_attribute_in_model(model, "jnt_stiffness", jnt_stiffness, backend)
        if self.rand_conf["randomize_prosthesis_body_position"]:
            model = self._set_attribute_in_model(model, "body_pos", body_pos, backend)
        if self.rand_conf["randomize_prosthesis_body_orientation"]:
            model = self._set_attribute_in_model(model, "body_quat", body_quat, backend)
        # if self.rand_conf["randomize_prosthesis_foot_geom_solref"]:
        #     model = self._set_attribute_in_model(model, "geom_solref", feet_geom_solref, backend)


        # jax.debug.print("model.jnt_stiffness: {jnt_stiffness}", jnt_stiffness=model.jnt_stiffness)
        # jax.debug.print("model.dof_damping: {dof_damping}", dof_damping=model.dof_damping)
        # jax.debug.print("model.body_pos after: {body_pos}", body_pos=model.body_pos)
        # jax.debug.print("model.body_quat: {body_quat}", body_quat=model.body_quat)
        # print("model.jnt_stiffness: ", model.jnt_stiffness)
        # print("model.dof_damping: ", model.dof_damping)
        # print("model.body_pos: ", model.body_pos)
        # print("model.body_quat: ", model.body_quat)

        return model, data, carry




    def _sample_joint_stiffness(self, model: Union[MjModel, Model],
                              carry: Any,
                              backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """ Samples the joint stiffness parameters.

        Args:
            model (Union[MjModel, Model]): The simulation model.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: The randomized joint stiffness and carry.

        """
        assert_backend_is_supported(backend)
        if self.rand_conf["randomize_prosthesis_joint_stiffness"]:
            stiffness_min, stiffness_max = self.rand_conf["prosthesis_joint_stiffness_range"]
            n_dofs = len(self._joint_indices.values()) #model.nv - 6 #exclude freejoint 6 degrees of freedom

            if backend == jnp:
                key = carry.key
                key, _k = jax.random.split(key)
                interpolation = jax.random.uniform(_k, shape=(n_dofs,))
                carry = carry.replace(key=key)
            else:
                interpolation = np.random.uniform(size=(n_dofs,))

        # if self.rand_conf["randomize_prosthesis_joint_stiffness"]:
            sampled_stiffness = stiffness_min + (stiffness_max - stiffness_min) * interpolation
        else: 
            if backend == np: 
                sampled_stiffness = model.jnt_stiffness[list(self._joint_indices.values())].copy()
            elif backend == jnp:
                indices = jnp.array(list(self._joint_indices.values()))
                sampled_stiffness = model.jnt_stiffness.at[indices].get()

        return sampled_stiffness, carry

    def _sample_dof_damping(self, model: Union[MjModel, Model],
                              carry: Any,
                              backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Samples the joint damping parameters.

        Args:
            model (Union[MjModel, Model]): The simulation model.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: The randomized joint damping and carry.

        """

        assert_backend_is_supported(backend)
        if self.rand_conf["randomize_prosthesis_dof_damping"]:
            damping_min, damping_max = self.rand_conf["prosthesis_dof_damping_range"]
            n_dofs = len(self._dof_indices.values()) #model.nv - 6 #exclude freejoint 6 degrees of freedom

            if backend == jnp:
                key = carry.key
                key, _k = jax.random.split(key)
                interpolation = jax.random.uniform(_k, shape=(n_dofs,))
                # jax.debug.print("interpolation: {}", interpolation)
                carry = carry.replace(key=key)
            else:
                interpolation = np.random.uniform(size=(n_dofs,))

        # sampled_damping = (
        #     damping_min + (damping_max - damping_min) * interpolation
        #     if self.rand_conf["randomize_prosthesis_dof_damping"]
        #     else model.dof_damping[self._dof_indices.values()]
        # )


            sampled_damping = damping_min + (damping_max - damping_min) * interpolation
        else: 
            if backend == np: 
                sampled_damping = model.jnt_stiffness[list(self._dof_indices.values())].copy()
            elif backend == jnp:
                indices = jnp.array(list(self._dof_indices.values()))
                sampled_damping = model.dof_damping.at[indices].get()


        return sampled_damping, carry


    def _sample_geom_position(self, model: Union[MjModel, Model],
                              carry: Any,
                              backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Sample random position for the amputated limb prosthesis.
        
        Args:
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).
        
        Returns:
            ndarray: Randomly sampled position.
        """
        assert_backend_is_supported(backend)

        # Adapt prosthesis_body_position_range: {'x': [-0.1, 0.1], 'y': [-0.1, 0.1], 'z': [-0.001, 0.001]}
        position_range = self.rand_conf["prosthesis_body_position_range"]
        directions = list(position_range.keys())

        # position_range = self.rand_conf["prosthesis_body_position_range"]
        # directions = self.rand_conf["prosthesis_body_position_direction"]
        # Build min/max arrays in the order of directions
        position_min = np.array([position_range[axis][0] for axis in directions])
        position_max = np.array([position_range[axis][1] for axis in directions])
        n_bodies = len(self._body_pos_indices.values())

        ncoordinates = len(directions) #self.rand_conf["prosthesis_body_position_direction"])

        
        if backend == jnp:
            key = carry.key
            key, _k = jax.random.split(key)
            interpolation = jax.random.uniform(_k, shape=(n_bodies, ncoordinates))
            carry = carry.replace(key=key)
        else:
            interpolation = np.random.uniform(size=(n_bodies, ncoordinates))

        sampled_position_interp = position_min + (position_max - position_min) * interpolation
        if self.rand_conf["randomize_prosthesis_body_position"]:
            if backend == np:
                sampled_position = self._init_prosthesis_body_position[list(self._body_pos_indices.values())].copy()
                for i in range(ncoordinates):
                    if directions[i] == 'x':
                        sampled_position[:, 0] = sampled_position_interp[:, i] + sampled_position[:,0]
                    elif directions[i] == 'y':
                        sampled_position[:, 1] = sampled_position_interp[:, i] + sampled_position[:,1]
                    elif directions[i] == 'z':
                        sampled_position[:, 2] = sampled_position_interp[:, i] + sampled_position[:,2]
                # if 'x' in self.rand_conf["prosthesis_body_position_direction"]:
                #     sampled_position[:, 0] = sampled_position_interp[:, i]
                #     i += 1
                # if 'y' in self.rand_conf["prosthesis_body_position_direction"]:
                #     sampled_position[:, 1] = sampled_position_interp[:, i]
                #     i += 1
                # if 'z' in self.rand_conf["prosthesis_body_position_direction"]:
                #     sampled_position[:, 2] = sampled_position_interp[:, i]
                #     i += 1
            elif backend == jnp:
                # JAX does not support list indexing with .at[].get(), so use jnp.array for indices
                indices = jnp.array(list(self._body_pos_indices.values()))
                sampled_position = model.body_pos.at[indices].get()
                def update_position(pos, interp):
                    for i in range(ncoordinates):
                        if directions[i] == 'x':
                            pos = pos.at[..., 0].set(interp[..., i] + pos[..., 0])
                        elif directions[i] == 'y':
                            pos = pos.at[..., 1].set(interp[..., i] + pos[..., 1])
                        elif directions[i] == 'z':
                            pos = pos.at[..., 2].set(interp[..., i] + pos[..., 2])
                    return pos

                sampled_position = jax.vmap(update_position)(sampled_position, sampled_position_interp)
        else:
            # No randomization, keep initial position
            if backend == np:
                sampled_position = model.body_pos[list(self._body_pos_indices.values())].copy()
            elif backend == jnp:
                # JAX does not support list indexing with .at[].get(), so use jnp.array for indices
                indices = jnp.array(list(self._body_pos_indices.values()))
                sampled_position = model.body_pos.at[indices].get()
            # sampled_position = self._init_prosthesis_body_position[list(self._body_indices.values())].copy()
        return sampled_position, carry


    def _sample_joint_orientation(self, model: Union[MjModel, Model],
                              carry: Any,
                              backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Sample random orientation for the amputated limb prosthesis.
        
        Args:
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).
        
        Returns:
            ndarray: Randomly sampled orientation.
        """
        assert_backend_is_supported(backend)

        orientation_range = self.rand_conf["prosthesis_body_orientation_range"]

        directions= list(orientation_range.keys())
        # directions = self.rand_conf["prosthesis_body_orientation_direction"]
        # Build min/max arrays in the order of directions
        orientation_min = np.array([orientation_range[axis][0] for axis in directions])
        orientation_max = np.array([orientation_range[axis][1] for axis in directions])
        n_bodies = len(self._body_quat_indices.values())
        ndirections = len(directions) #self.rand_conf["prosthesis_body_orientation_direction"]) 
        
        if backend == jnp:
            key = carry.key
            key, _k = jax.random.split(key)
            interpolation = jax.random.uniform(_k, shape=(n_bodies, ndirections))
            carry = carry.replace(key=key)
        else:
            interpolation = np.random.uniform(size=(n_bodies, ndirections))
        sampled_orientation_euler = orientation_min + (orientation_max - orientation_min) * interpolation
        
        if backend == np:
            init_orientation = self._init_prosthesis_body_orientation[list(self._body_quat_indices.values())].copy() # Format: (w,x,y,z)
            # Convert (w, x, y, z) to (x, y, z, w) for rotation calculations
            init_orientation_sorted = np.concatenate(
                [init_orientation[:, 1:4], init_orientation[:, 0:1]], axis=1
            ) #Format: (x,y,z,w)
            init_rotation = R.from_quat(init_orientation_sorted)
        elif backend == jnp:
            # JAX does not support list indexing with .at[].get(), so use jnp.array for indices
            indices = jnp.array(list(self._body_quat_indices.values()))
            init_orientation = model.body_quat.at[indices].get()
            # Convert (w, x, y, z) to (x, y, z, w) for rotation calculations
            init_orientation_sorted = jnp.concatenate(
                [init_orientation[:, 1:4], init_orientation[:, 0:1]], axis=1
            ) #Format: (x,y,z,w)
            init_rotation = jaxR.from_quat(init_orientation_sorted)
        
        
        orientation_euler = init_rotation.as_euler('xyz', degrees=False)

        if self.rand_conf["randomize_prosthesis_body_orientation"]:
            for i in range(len(directions)):
                if backend == np:
                    if directions[i] == 'x':
                        orientation_euler[:, 0] = sampled_orientation_euler[:, i] + orientation_euler[:, 0]
                    elif directions[i] == 'y':
                        orientation_euler[:, 1] = sampled_orientation_euler[:, i] + orientation_euler[:, 1]
                    elif directions[i] == 'z':
                        orientation_euler[:, 2] = sampled_orientation_euler[:, i] + orientation_euler[:, 2]
                elif backend == jnp:
                    def update_orientation(euler, interp):
                        for i in range(ndirections):
                            if directions[i] == 'x':
                                euler = euler.at[..., 0].set(interp[..., i] + euler[..., 0])
                                # euler = euler.at[..., 0].set(interp[i] + orientation_euler[0])
                            elif directions[i] == 'y':
                                euler = euler.at[..., 1].set(interp[..., i] + euler[..., 1])
                                # euler = euler.at[..., 1].set(interp[i] + orientation_euler[1])
                            elif directions[i] == 'z':
                                euler = euler.at[..., 2].set(interp[..., i] + euler[..., 2])
                                # euler = euler.at[..., 2].set(interp[i] + orientation_euler[2])
                        return euler
                    

                    orientation_euler = jax.vmap(update_orientation)(orientation_euler, sampled_orientation_euler)


        # sampled_orientation = sampled_rotation.as_quat()  # Convert back to (w, x, y, z) format
        if self.rand_conf["randomize_prosthesis_body_orientation"]:
            if backend == np:
                sampled_rotation = R.from_euler('xyz', orientation_euler, degrees=False)
                sampled_orientation_unsorted = sampled_rotation.as_quat() 
                sampled_orientation = np.concatenate(
                    [sampled_orientation_unsorted[:, 3:4], sampled_orientation_unsorted[:, 0:3]], axis=1
                )
            elif backend == jnp:
                sampled_rotation = jaxR.from_euler('xyz', orientation_euler, degrees=False)
                sampled_orientation_unsorted = sampled_rotation.as_quat()
                sampled_orientation = jnp.concatenate(
                    [sampled_orientation_unsorted[..., 3:4], sampled_orientation_unsorted[..., 0:3]], axis=1
                )
                # Convert back to jnp array for consistency
                # jax.debug.print("sampled_orientation: {sampled_orientation}", sampled_orientation = sampled_orientation)
                # jax.debug.print("sampled_orientation_euler: {orientation_euler}", orientation_euler = orientation_euler)
                # jax.debug.print("init orientation: {init_orientation}", init_orientation = init_orientation)
            # if backend == np:
            #     # Loop over each body and convert euler to quat in-place
            #     for i in range(sampled_orientation.shape[0]):
            #         mujoco.mju_euler2Quat(sampled_orientation[i], orientation_euler[i], 'xyz')
            #         sampled_orientation[i] = sampled_orientation[i]
            # else:
            #     # For JAX, use vmap with a function that returns a new quaternion
            #     def euler2quat(eul):
            #         quat = jnp.zeros(4)
            #         quat = quat.at[:].set(jnp.array(mujoco.mju_euler2Quat(np.zeros(4), np.array(eul), 'xyz')))
            #         return quat
            #     sampled_orientation = jax.vmap(euler2quat)(orientation_euler)
        else:
            # No randomization, keep initial orientation
            if backend == jnp: 
                indices = jnp.array(list(self._body_quat_indices.values()))
                sampled_orientation = model.body_quat.at[indices].get()
            elif backend == np:
                sampled_orientation = model.body_quat[list(self._body_quat_indices.values())].copy() #self._init_prosthesis_body_orientation[list(self._body_indices.values())].copy()

        return sampled_orientation, carry
    

    # def _sample_feet_geom_solref(self, model: Union[MjModel, Model],
    #                           carry: Any,
    #                           backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
    #     """ Sample random solref for the feet geoms of the amputated limb prosthesis.
    #     Args:
    #         model (Union[MjModel, Model]): The simulation model.
    #         carry (Any): Carry instance with additional state information.
    #         backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).
    #     Returns:
    #         Tuple[Union[np.ndarray, jnp.ndarray], Any]: The randomized feet geom solref and carry.
    #     """
    #     assert_backend_is_supported(backend)
    #     if self.rand_conf["randomize_feet_geom_solref"]:
    #         solref_range = self.rand_conf["feet_geom_solref_range"]
    #         geom_sides = list(solref_range.keys())  # ['prosthesis_side', 'other_side']

    #         # Build solref_min and solref_max arrays in the order of indices in self._feet_geom_solref_indices
    #         solref_min = []
    #         solref_max = []
    #         for side_key in self._feet_geom_solref_indices.keys():
    #             solref_min.append(solref_range[side_key][0])
    #             solref_max.append(solref_range[side_key][1])
    #         solref_min = np.array(solref_min)
    #         solref_max = np.array(solref_max)

    #         n_geoms = len(self._feet_geom_solref_indices.values())
    #         if backend == jnp:
    #             key = carry.key
    #             key, _k = jax.random.split(key)
    #             interpolation = jax.random.uniform(_k, shape=(n_geoms, 2))
    #             carry = carry.replace(key=key)
    #         else:
    #             interpolation = np.random.uniform(size=(n_geoms, 2))
    #     # if self.rand_conf["randomize_prosthesis_foot_geom_solref"]:
    #         sampled_solref = solref_min + (solref_max - solref_min) * interpolation
    #     else:
    #         if backend == np: 
    #             sampled_solref = model.geom_solref[list(self._feet_geom_solref_indices.values())].copy()
    #         elif backend == jnp:
    #             indices = jnp.array(list(self._feet_geom_solref_indices.values()))
    #             sampled_solref = model.geom_solref.at[indices].get()


    #     return sampled_solref, carry
    


    def update_action(self,
                      env: Any,
                      action: Union[np.ndarray, jnp.ndarray],
                      model: Union[MjModel, Model],
                      data: Union[MjData, Data],
                      carry: Any,
                      backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Update the action with randomization effects.

        Args:
            env (Any): The environment instance.
            action (Union[np.ndarray, jnp.ndarray]): The action to be updated.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: The updated action and carry.

        """
        assert_backend_is_supported(backend)
        return action, carry
    
    

    def update_observation(self,
                           env: Any,
                           obs: Union[np.ndarray, jnp.ndarray],
                           model: Union[MjModel, Model],
                           data: Union[MjData, Data],
                           carry: Any,
                           backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:

        """
            Update the observation with randomization effects.

            Args:
                env (Any): The environment instance.
                obs (Union[np.ndarray, jnp.ndarray]): The observation to be updated.
                model (Union[MjModel, Model]): The simulation model.
                data (Union[MjData, Data]): The simulation data.
                carry (Any): Carry instance with additional state information.
                backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

            Returns:
                Tuple[Union[np.ndarray, jnp.ndarray], Any]: The updated observation and carry.

            """
        
        assert_backend_is_supported(backend)
        return obs, carry
    