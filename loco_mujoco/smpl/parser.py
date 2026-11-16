# This script is borrowed and extended from https://github.com/nkolot/SPIN/blob/master/models/hmr.py
# Adhere to their licence to use this script

import os
import pickle

import numpy as np
from loco_mujoco.smpl import SMPL_BONE_ORDER_NAMES, SMPLH_BONE_ORDER_NAMES

try:
    import torch
    from smplx import SMPL as _SMPL
    from smplx import SMPLH as _SMPLH

    from smplx.utils import to_np, to_tensor
    from smplx.lbs import blend_shapes, vertices2joints, batch_rodrigues, batch_rigid_transform, transform_mat

except ImportError:
    # what can i do here?
    _SMPL = None
    _SMPLH = None
    torch = None


# The two helpers below give the parsers the behaviour they need on the smplx release from PyPI.
# Earlier versions required a fork of smplx that (1) broadcasts the default pose and shape to the
# batch size and (2) uses every shape coefficient stored in the model file instead of the first 10.


def match_dim(x, batch_size):
    """Repeats a single-row tensor along the batch dimension."""
    if x.shape[0] == batch_size:
        return x
    if x.shape[0] == 1:
        return x.repeat((batch_size, 1))
    raise ValueError(f"cannot broadcast a batch of {x.shape[0]} to {batch_size}")


def _use_all_shape_coefficients(model, model_name, args, kwargs):
    """Makes ``model`` use every shape coefficient of its model file (the PyPI smplx keeps 10)."""
    data_struct = kwargs.get("data_struct")
    if data_struct is not None:
        shapedirs = data_struct.shapedirs
    else:
        model_path = kwargs.get("model_path", args[0] if args else None)
        if os.path.isdir(model_path):
            model_path = os.path.join(
                model_path, f"{model_name}_{kwargs.get('gender', 'neutral').upper()}.{kwargs.get('ext', 'pkl')}")
        if model_path.endswith(".npz"):
            shapedirs = np.load(model_path, allow_pickle=True)["shapedirs"]
        else:
            with open(model_path, "rb") as f:
                shapedirs = pickle.load(f, encoding="latin1")["shapedirs"]

    shapedirs = to_np(shapedirs)
    n_coefficients = shapedirs.shape[-1]
    if n_coefficients == model.shapedirs.shape[-1]:
        return
    dtype = model.shapedirs.dtype
    model.shapedirs = to_tensor(shapedirs, dtype=dtype)
    model._num_betas = n_coefficients
    if hasattr(model, "betas"):
        betas = torch.zeros([model.betas.shape[0], n_coefficients], dtype=dtype)
        model.betas = torch.nn.Parameter(betas, requires_grad=model.betas.requires_grad)


def _batch_defaults(model, names, given):
    """Fills the arguments that were not passed with the model's defaults, repeated to the batch size."""
    batch_size = max([given[n].shape[0] for n in ("betas", "global_orient", "body_pose") if given[n] is not None],
                     default=1)
    return {n: given[n] if given[n] is not None else match_dim(getattr(model, n), batch_size) for n in names}


class SMPL_Parser(_SMPL):

    def __init__(self, create_transl=False, *args, **kwargs):
        """SMPL model constructor
        Parameters
        ----------
        model_path: str
            The path to the folder or to the file where the model
            parameters are stored
        data_struct: Strct
            A struct object. If given, then the parameters of the model are
            read from the object. Otherwise, the model tries to read the
            parameters from the given `model_path`. (default = None)
        create_global_orient: bool, optional
            Flag for creating a member variable for the global orientation
            of the body. (default = True)
        global_orient: torch.tensor, optional, Bx3
            The default value for the global orientation variable.
            (default = None)
        create_body_pose: bool, optional
            Flag for creating a member variable for the pose of the body.
            (default = True)
        body_pose: torch.tensor, optional, Bx(Body Joints * 3)
            The default value for the body pose variable.
            (default = None)
        create_betas: bool, optional
            Flag for creating a member variable for the shape space
            (default = True).
        betas: torch.tensor, optional, Bx10
            The default value for the shape member variable.
            (default = None)
        create_transl: bool, optional
            Flag for creating a member variable for the translation
            of the body. (default = True)
        transl: torch.tensor, optional, Bx3
            The default value for the transl variable.
            (default = None)
        dtype: torch.dtype, optional
            The data type for the created variables
        batch_size: int, optional
            The batch size used for creating the member variables
        joint_mapper: object, optional
            An object that re-maps the joints. Useful if one wants to
            re-order the SMPL joints to some other convention (e.g. MSCOCO)
            (default = None)
        gender: str, optional
            Which gender to load
        vertex_ids: dict, optional
            A dictionary containing the indices of the extra vertices that
            will be selected
        """
        super(SMPL_Parser, self).__init__(*args, **kwargs)
        _use_all_shape_coefficients(self, "SMPL", args, kwargs)
        self.device = next(self.parameters()).device
        self.joint_names = SMPL_BONE_ORDER_NAMES

        self.joint_axes = {x: np.identity(3) for x in self.joint_names}
        self.joint_dofs = {x: ["x", "y", "z"] for x in self.joint_names}
        self.joint_range = {x: np.hstack([np.ones([3, 1]) * -np.pi, np.ones([3, 1]) * np.pi]) for x in self.joint_names}
        self.joint_range["L_Elbow"] *= 4
        self.joint_range["R_Elbow"] *= 4
        self.joint_range["L_Shoulder"] *= 4
        self.joint_range["R_Shoulder"] *= 4

        self.contype = {1: self.joint_names}
        self.conaffinity = {1: self.joint_names}

        self.zero_pose = torch.zeros(1, 72).float()

    def forward(self, betas=None, body_pose=None, global_orient=None, **kwargs):
        defaults = _batch_defaults(self, ("betas", "body_pose", "global_orient"),
                                   dict(betas=betas, body_pose=body_pose, global_orient=global_orient))
        smpl_output = super(SMPL_Parser, self).forward(**defaults, **kwargs)
        return smpl_output

    def get_joints_verts(self, pose, th_betas=None, th_trans=None):
        """
        Pose should be batch_size x 72
        """
        if pose.shape[1] != 72:
            pose = pose.reshape(-1, 72)

        pose = pose.float()
        if th_betas is not None:
            th_betas = th_betas.float()

            if th_betas.shape[-1] == 16:
                th_betas = th_betas[:, :10]

        batch_size = pose.shape[0]

        smpl_output = self.forward(
            betas=th_betas,
            transl=th_trans,
            body_pose=pose[:, 3:],
            global_orient=pose[:, :3],
        )
        vertices = smpl_output.vertices
        joints = smpl_output.joints[:, :24]
        return vertices, joints

    def get_offsets(self, v_template=None, zero_pose=None, betas=None):

        if betas is None:
            betas = torch.zeros(1, 10).float()

        with torch.no_grad():
            if zero_pose is None:
                verts, Jtr = self.get_joints_verts(self.zero_pose, th_betas=betas)
            else:
                verts, Jtr = self.get_joints_verts(zero_pose, th_betas=betas)
            verts_np = verts.detach().cpu().numpy()
            jts_np = Jtr.detach().cpu().numpy()
            parents = self.parents.cpu().numpy()
            offsets_smpl = [np.array([0, 0, 0])]
            for i in range(1, len(parents)):
                p_id = parents[i]
                p3d = jts_np[0, p_id]
                curr_3d = jts_np[0, i]
                offset_curr = curr_3d - p3d
                offsets_smpl.append(offset_curr)
            offsets_smpl = np.array(offsets_smpl)
            joint_names = self.joint_names
            joint_pos = Jtr[0].numpy()
            smpl_joint_parents = self.parents.cpu().numpy()
            joint_offsets = {joint_names[c]: (joint_pos[c] - joint_pos[p]) if c > 0 else joint_pos[c] for c, p in
                             enumerate(smpl_joint_parents)}
            parents_dict = {joint_names[i]: joint_names[parents[i]] for i in range(len(joint_names))}
            channels = ["z", "y", "x"]
            skin_weights = self.lbs_weights.numpy()
            return (verts[0], jts_np[0], skin_weights, self.joint_names, joint_offsets, parents_dict, channels,
                    self.joint_range)

    def get_mesh_offsets(self, zero_pose=None, betas=None, flatfoot=False):
        if betas is None:
            betas = torch.zeros(1, 10)

        with torch.no_grad():
            joint_names = self.joint_names
            if zero_pose is None:
                verts, Jtr = self.get_joints_verts(self.zero_pose, th_betas=betas)
            else:
                verts, Jtr = self.get_joints_verts(zero_pose, th_betas=betas)

            verts_np = verts.detach().cpu().numpy()
            verts = verts_np[0]

            if flatfoot:
                feet_subset = verts[:, 1] < np.min(verts[:, 1]) + 0.01
                verts[feet_subset, 1] = np.mean(verts[feet_subset][:, 1])

            smpl_joint_parents = self.parents.cpu().numpy()

            joint_pos = Jtr[0].numpy()
            joint_offsets = {joint_names[c]: (joint_pos[c] - joint_pos[p]) if c > 0 else joint_pos[c] for c, p in
                             enumerate(smpl_joint_parents)}
            joint_parents = {x: joint_names[i] if i >= 0 else None for x, i in zip(joint_names, smpl_joint_parents)}

            # skin_weights = smpl_layer.th_weights.numpy()
            skin_weights = self.lbs_weights.numpy()
            return (
                verts,
                joint_pos,
                skin_weights,
                joint_names,
                joint_offsets,
                joint_parents,
                self.joint_axes,
                self.joint_dofs,
                self.joint_range,
                self.contype,
                self.conaffinity,
            )

    def get_mesh_offsets_batch(self, betas=None, flatfoot=False):
        if betas is None:
            betas = torch.zeros(1, 10)

        with torch.no_grad():
            joint_names = self.joint_names
            verts, Jtr = self.get_joints_verts(self.zero_pose.repeat(betas.shape[0], 1), th_betas=betas)
            verts_np = verts.detach().cpu().numpy()
            verts = verts_np[0]

            if flatfoot:
                feet_subset = verts[:, 1] < np.min(verts[:, 1]) + 0.01
                verts[feet_subset, 1] = np.mean(verts[feet_subset][:, 1])

            smpl_joint_parents = self.parents.cpu().numpy()

            joint_pos = Jtr
            joint_offsets = {joint_names[c]: (joint_pos[:, c] - joint_pos[:, p]) if c > 0 else joint_pos[:, c] for c, p
                             in enumerate(smpl_joint_parents)}
            joint_parents = {x: joint_names[i] if i >= 0 else None for x, i in zip(joint_names, smpl_joint_parents)}

            skin_weights = self.lbs_weights
            return (
                verts,
                joint_pos,
                skin_weights,
                joint_names,
                joint_offsets,
                joint_parents,
                self.joint_axes,
                self.joint_dofs,
                self.joint_range,
                self.contype,
                self.conaffinity,
            )


class SMPLH_Parser(_SMPLH):

    def __init__(self, *args, **kwargs):
        super(SMPLH_Parser, self).__init__(*args, **kwargs)
        _use_all_shape_coefficients(self, "SMPLH", args, kwargs)
        self.device = next(self.parameters()).device
        self.joint_names = SMPLH_BONE_ORDER_NAMES
        self.joint_axes = {x: np.identity(3) for x in self.joint_names}
        self.joint_dofs = {x: ["z", "y", "x"] for x in self.joint_names}
        self.joint_range = {x: np.hstack([np.ones([3, 1]) * -np.pi, np.ones([3, 1]) * np.pi]) for x in self.joint_names}
        self.joint_range["L_Elbow"] *= 4
        self.joint_range["R_Elbow"] *= 4

        self.contype = {1: self.joint_names}
        self.conaffinity = {1: self.joint_names}
        self.zero_pose = torch.zeros(1, 156).float()

    def forward(self, betas=None, body_pose=None, global_orient=None, left_hand_pose=None, right_hand_pose=None,
                **kwargs):
        names = ("betas", "body_pose", "global_orient", "left_hand_pose", "right_hand_pose")
        defaults = _batch_defaults(self, names, dict(betas=betas, body_pose=body_pose, global_orient=global_orient,
                                                     left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose))
        smpl_output = super(SMPLH_Parser, self).forward(**defaults, **kwargs)
        return smpl_output

    def get_joints_verts(self, pose, th_betas=None, th_trans=None):
        """
        Pose should be batch_size x 156
        """

        if pose.shape[1] != 156:
            pose = pose.reshape(-1, 156)
        pose = pose.float()
        if th_betas is not None:
            th_betas = th_betas.float()

        batch_size = pose.shape[0]
        smpl_output = self.forward(
            body_pose=pose[:, 3:66],
            global_orient=pose[:, :3],
            L_hand_pose=pose[:, 66:111],
            R_hand_pose=pose[:, 111:156],
            betas=th_betas,
            transl=th_trans,
        )
        vertices = smpl_output.vertices
        joints = smpl_output.joints
        return vertices, joints

    def get_joint_transformations(self, pose, th_betas=None, th_trans=None):

        if pose.shape[1] != 156:
            pose = pose.reshape(-1, 156)
        pose = pose.float()
        if th_betas is not None:
            th_betas = th_betas.float()

        batch_size = pose.shape[0]
        T = self._transforms(
            body_pose=pose[:, 3:66],
            global_orient=pose[:, :3],
            L_hand_pose=pose[:, 66:111],
            R_hand_pose=pose[:, 111:156],
            betas=th_betas,
            transl=th_trans,
        )

        if th_trans is not None:
            T[..., :3, 3] += th_trans.unsqueeze(dim=1)

        return T

    def _transforms(self,  #sim_to_forward
                    betas = None,
                    global_orient = None,
                    body_pose = None,
                    left_hand_pose = None,
                    right_hand_pose = None,
                    transl = None,
                    return_verts: bool = True,
                    return_full_pose: bool = False,
                    pose2rot: bool = True,
                    **kwargs):

            # If no shape and pose parameters are passed along, then use the
            # ones from the module

            batch_size = max(
                betas.shape[0] if not betas is None else 1,
                global_orient.shape[0] if not global_orient is None else 1,
                body_pose.shape[0] if not body_pose is None else 1,
            )

            global_orient = (global_orient if global_orient is not None else
                             match_dim(self.global_orient, batch_size))
            body_pose = (body_pose if body_pose is not None else match_dim(
                self.body_pose, batch_size))
            betas = betas if betas is not None else match_dim(
                self.betas, batch_size)

            left_hand_pose = (left_hand_pose if left_hand_pose is not None else
                              match_dim(self.left_hand_pose, batch_size))
            right_hand_pose = (right_hand_pose if right_hand_pose is not None else
                               match_dim(self.right_hand_pose, batch_size))

            apply_trans = transl is not None or hasattr(self, "transl")
            if transl is None:
                if hasattr(self, "transl"):
                    transl = self.transl

            if self.use_pca:
                left_hand_pose = torch.einsum(
                    "bi,ij->bj", [left_hand_pose, self.left_hand_components])
                right_hand_pose = torch.einsum(
                    "bi,ij->bj", [right_hand_pose, self.right_hand_components])
            full_pose = torch.cat(
                [global_orient, body_pose, left_hand_pose, right_hand_pose], dim=1)
            full_pose += self.pose_mean

            batch_size = max(betas.shape[0], full_pose.shape[0])
            device, dtype = betas.device, betas.dtype

            # Add shape contribution
            v_shaped = self.v_template + blend_shapes(betas, self.shapedirs)

            # Get the joints
            # NxJx3 array
            J = vertices2joints(self.J_regressor, v_shaped)

            # 3. Add pose blend shapes
            # N x J x 3 x 3
            if pose2rot:
                rot_mats = batch_rodrigues(full_pose.view(-1, 3)).view(
                    [batch_size, -1, 3, 3])

            else:
                #pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
                rot_mats = full_pose.view(batch_size, -1, 3, 3)

            # 4. Get the global joint location
            T = self.batch_rigid_transform_global(rot_mats, J, self.parents, dtype=dtype)

            return T

    @staticmethod
    def batch_rigid_transform_global(
            rot_mats,
            joints,
            parents,
            dtype=torch.float32
    ):
        """
        Applies a batch of rigid transformations to the joints

        """

        joints = torch.unsqueeze(joints, dim=-1)

        rel_joints = joints.clone()
        rel_joints[:, 1:] -= joints[:, parents[1:]]

        transforms_mat = transform_mat(
            rot_mats.reshape(-1, 3, 3),
            rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

        transform_chain = [transforms_mat[:, 0]]
        for i in range(1, parents.shape[0]):
            # Subtract the joint location at the rest pose
            # No need for rotation, since it's identity when at rest
            curr_res = torch.matmul(transform_chain[parents[i]],
                                    transforms_mat[:, i])
            transform_chain.append(curr_res)

        transforms = torch.stack(transform_chain, dim=1)

        return transforms

    def get_offsets(self, v_template=None, zero_pose=None, betas=torch.zeros(1, 10), flatfoot=False):
        with torch.no_grad():
            joint_names = self.joint_names
            if zero_pose is None:
                verts, Jtr = self.get_joints_verts(self.zero_pose, th_betas=betas)
            else:
                verts, Jtr = self.get_joints_verts(zero_pose, th_betas=betas)

            jts_np = Jtr.detach().cpu().numpy()

            smpl_joint_parents = self.parents.cpu().numpy()
            joint_pos = Jtr[0].numpy()
            joint_offsets = {joint_names[c]: (joint_pos[c] - joint_pos[p]) if c > 0 else joint_pos[c] for c, p in
                             enumerate(smpl_joint_parents) if joint_names[c] in self.joint_names}
            parents_dict = {x: joint_names[i] if i >= 0 else None for x, i in zip(joint_names, smpl_joint_parents) if
                            joint_names[i] in self.joint_names and x in self.joint_names}

            #  (SMPLX_BONE_ORDER_NAMES[:22] + SMPLX_BONE_ORDER_NAMES[25:55]) == SMPLH_BONE_ORDER_NAMES # ZL Hack: only use the weights we need.
            skin_weights = self.lbs_weights.numpy()
            skin_weights.argmax(axis=1)

            channels = ["z", "y", "x"]
            return (verts[0], jts_np[0], skin_weights, self.joint_names, joint_offsets, parents_dict, channels,
                    self.joint_range)

    def get_mesh_offsets(self, zero_pose=None, betas=torch.zeros(1, 10), flatfoot=False):
        with torch.no_grad():
            joint_names = self.joint_names
            if zero_pose is None:
                verts, Jtr = self.get_joints_verts(self.zero_pose, th_betas=betas)
            else:
                verts, Jtr = self.get_joints_verts(zero_pose, th_betas=betas)

            verts_np = verts.detach().cpu().numpy()
            verts = verts_np[0]

            if flatfoot:
                feet_subset = verts[:, 1] < np.min(verts[:, 1]) + 0.01
                verts[feet_subset, 1] = np.mean(verts[feet_subset][:, 1])

            smpl_joint_parents = self.parents.cpu().numpy()

            joint_pos = Jtr[0].numpy()
            joint_offsets = {joint_names[c]: (joint_pos[c] - joint_pos[p]) if c > 0 else joint_pos[c] for c, p in
                             enumerate(smpl_joint_parents)}
            joint_parents = {x: joint_names[i] if i >= 0 else None for x, i in zip(joint_names, smpl_joint_parents)}

            skin_weights = self.lbs_weights.numpy()
            return (
                verts,
                joint_pos,
                skin_weights,
                joint_names,
                joint_offsets,
                joint_parents,
                self.joint_axes,
                self.joint_dofs,
                self.joint_range,
                self.contype,
                self.conaffinity,
            )

    def get_mesh_offsets_batch(self, betas=torch.zeros(1, 10), flatfoot=False):
        with torch.no_grad():
            joint_names = self.joint_names
            verts, Jtr = self.get_joints_verts(self.zero_pose.repeat(betas.shape[0], 1), th_betas=betas)
            verts_np = verts.detach().cpu().numpy()
            verts = verts_np[0]

            if flatfoot:
                feet_subset = verts[:, 1] < np.min(verts[:, 1]) + 0.01
                verts[feet_subset, 1] = np.mean(verts[feet_subset][:, 1])

            smpl_joint_parents = self.parents.cpu().numpy()

            joint_pos = Jtr
            joint_offsets = {joint_names[c]: (joint_pos[:, c] - joint_pos[:, p]) if c > 0 else joint_pos[:, c] for c, p
                             in enumerate(smpl_joint_parents)}
            joint_parents = {x: joint_names[i] if i >= 0 else None for x, i in zip(joint_names, smpl_joint_parents)}

            skin_weights = self.lbs_weights
            return (
                verts,
                joint_pos,
                skin_weights,
                joint_names,
                joint_offsets,
                joint_parents,
                self.joint_axes,
                self.joint_dofs,
                self.joint_range,
                self.contype,
                self.conaffinity,
            )
