"""
Parsers for the SMPL and SMPL-H body models.

The classes extend the models of the ``smplx`` package by the joint names used in LocoMuJoCo and by methods
returning joint positions, joint offsets and global joint transformations, which the retargeting pipeline builds
on. The module requires the optional ``smpl`` dependencies (``torch`` and ``smplx``).

Borrowed and extended from https://github.com/nkolot/SPIN/blob/master/models/hmr.py.
"""

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
    """
    Repeat a single-row tensor along the batch dimension.

    Args:
        x (torch.Tensor): Tensor of shape ``(1, dim)`` or ``(batch_size, dim)``.
        batch_size (int): Desired batch size.

    Returns:
        torch.Tensor: Tensor of shape ``(batch_size, dim)``.

    Raises:
        ValueError: If the first dimension of ``x`` is neither 1 nor ``batch_size``.

    """
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
    """
    SMPL body model with the joint names and helper methods used in LocoMuJoCo.

    Extends ``smplx.SMPL``. All shape coefficients stored in the model file are used (``smplx`` itself keeps the
    first 10), and pose and shape arguments that are not passed to :meth:`forward` are filled with the model's
    defaults, repeated to the batch size. A pose is an axis-angle vector of 72 entries: 3 for the global orientation
    followed by 3 for each of the 23 body joints, in the order of ``SMPL_BONE_ORDER_NAMES``.

    Attributes:
        joint_names (List[str]): Names of the 24 SMPL joints.
        joint_axes (dict): Joint axes per joint name, a 3x3 identity matrix each.
        joint_dofs (dict): Names of the degrees of freedom per joint name.
        joint_range (dict): Joint range per joint name, shape ``(3, 2)``.
        contype (dict): Contact type mapped to joint names.
        conaffinity (dict): Contact affinity mapped to joint names.
        zero_pose (torch.Tensor): Pose with all joint rotations set to zero, shape ``(1, 72)``.

    """

    def __init__(self, create_transl=False, *args, **kwargs):
        """
        SMPL model constructor.

        Args:
            create_transl (bool): Not used.
            *args: Positional arguments passed to ``smplx.SMPL``.
            **kwargs: Keyword arguments passed to ``smplx.SMPL``. The most relevant ones are ``model_path`` (str, path
                to the folder or to the file where the model parameters are stored), ``gender`` (str, which gender to
                load), ``data_struct`` (a struct object to read the model parameters from instead of ``model_path``),
                ``batch_size`` (int, batch size used for creating the member variables) and ``dtype`` (data type of
                the created variables). See the documentation of ``smplx.SMPL`` for the complete list.

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
        """
        Run the SMPL model.

        Arguments that are not given are replaced by the model's default values, repeated to the batch size of the
        given arguments.

        Args:
            betas (torch.Tensor): Shape coefficients, shape ``(batch_size, n_betas)``.
            body_pose (torch.Tensor): Axis-angle rotations of the body joints, shape ``(batch_size, 69)``.
            global_orient (torch.Tensor): Axis-angle global orientation, shape ``(batch_size, 3)``.
            **kwargs: Further keyword arguments of ``smplx.SMPL.forward``, e.g. ``transl``.

        Returns:
            The output of ``smplx.SMPL.forward``, with the fields ``vertices`` and ``joints``.

        """
        defaults = _batch_defaults(self, ("betas", "body_pose", "global_orient"),
                                   dict(betas=betas, body_pose=body_pose, global_orient=global_orient))
        smpl_output = super(SMPL_Parser, self).forward(**defaults, **kwargs)
        return smpl_output

    def get_joints_verts(self, pose, th_betas=None, th_trans=None):
        """
        Compute the mesh vertices and the joint positions for a batch of poses.

        Args:
            pose (torch.Tensor): Axis-angle poses, shape ``(batch_size, 72)`` or reshapeable to it.
            th_betas (torch.Tensor): Shape coefficients, shape ``(batch_size, n_betas)``. If 16 coefficients are
                given, only the first 10 are used. Defaults to the model's shape.
            th_trans (torch.Tensor): Root translations, shape ``(batch_size, 3)``.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The vertices, shape ``(batch_size, n_vertices, 3)``, and the positions
            of the 24 joints, shape ``(batch_size, 24, 3)``.

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
        """
        Get the rest-pose skeleton of the model for a body shape.

        Args:
            v_template: Not used.
            zero_pose (torch.Tensor): Pose in which the skeleton is evaluated, shape ``(1, 72)``. Defaults to the zero
                pose.
            betas (torch.Tensor): Shape coefficients, shape ``(1, n_betas)``. Defaults to zeros (the mean shape).

        Returns:
            tuple: A tuple of eight elements: the vertices of the mesh, shape ``(n_vertices, 3)``; the joint
            positions, shape ``(n_joints, 3)``; the linear blend skinning weights, shape ``(n_vertices, n_joints)``; the
            joint names; a dictionary mapping each joint name to its position relative to its parent (the absolute
            position for the root); a dictionary mapping each joint name to the name of its parent; the rotation
            channel order ``["z", "y", "x"]``; and the dictionary of joint ranges.

        """

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
        """
        Get the mesh, the skeleton and the joint properties of the model for a body shape.

        Args:
            zero_pose (torch.Tensor): Pose in which the model is evaluated, shape ``(1, 72)``. Defaults to the zero
                pose.
            betas (torch.Tensor): Shape coefficients, shape ``(1, n_betas)``. Defaults to zeros (the mean shape).
            flatfoot (bool): If True, the lowest vertices of the mesh (within 1 cm of the minimum along the y-axis) are
                moved to a common height, which flattens the soles.

        Returns:
            tuple: A tuple of eleven elements:

            - ``verts``: Vertices of the mesh, shape ``(n_vertices, 3)``.
            - ``joint_pos``: Joint positions, shape ``(n_joints, 3)``.
            - ``skin_weights``: Linear blend skinning weights, shape ``(n_vertices, n_joints)``.
            - ``joint_names``: Names of the joints.
            - ``joint_offsets``: Dictionary mapping each joint name to its position relative to its parent (the
              absolute position for the root).
            - ``joint_parents``: Dictionary mapping each joint name to the name of its parent (``None`` for the root).
            - ``joint_axes``: Dictionary mapping each joint name to its joint axes, a 3x3 identity matrix.
            - ``joint_dofs``: Dictionary mapping each joint name to the names of its degrees of freedom.
            - ``joint_range``: Dictionary mapping each joint name to its joint range, shape ``(3, 2)``.
            - ``contype``: Dictionary mapping the contact type ``1`` to the joint names.
            - ``conaffinity``: Dictionary mapping the contact affinity ``1`` to the joint names.

        """
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
        """
        Get the mesh, the skeleton and the joint properties of the model in the zero pose for a batch of body shapes.

        Args:
            betas (torch.Tensor): Shape coefficients, shape ``(batch_size, n_betas)``. Defaults to zeros (the mean
                shape).
            flatfoot (bool): If True, the lowest vertices of the mesh (within 1 cm of the minimum along the y-axis) are
                moved to a common height, which flattens the soles.

        Returns:
            tuple: A tuple of eleven elements:

            - ``verts``: Vertices of the mesh of the first body shape, shape ``(n_vertices, 3)``.
            - ``joint_pos``: Joint positions, shape ``(batch_size, n_joints, 3)``.
            - ``skin_weights``: Linear blend skinning weights, shape ``(n_vertices, n_joints)``.
            - ``joint_names``: Names of the joints.
            - ``joint_offsets``: Dictionary mapping each joint name to its position relative to its parent (the
              absolute position for the root).
            - ``joint_parents``: Dictionary mapping each joint name to the name of its parent (``None`` for the root).
            - ``joint_axes``: Dictionary mapping each joint name to its joint axes, a 3x3 identity matrix.
            - ``joint_dofs``: Dictionary mapping each joint name to the names of its degrees of freedom.
            - ``joint_range``: Dictionary mapping each joint name to its joint range, shape ``(3, 2)``.
            - ``contype``: Dictionary mapping the contact type ``1`` to the joint names.
            - ``conaffinity``: Dictionary mapping the contact affinity ``1`` to the joint names.

        Unlike in :meth:`get_mesh_offsets`, the joint positions, the joint offsets and the skinning weights are
        torch tensors, and the joint offsets have a leading batch dimension.

        """
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
    """
    SMPL-H body model (SMPL with articulated hands) with the joint names and helper methods used in LocoMuJoCo.

    Extends ``smplx.SMPLH`` and is the body model of the retargeting pipeline. All shape coefficients stored in the
    model file are used (``smplx`` itself keeps the first 10), and pose and shape arguments that are not passed to
    :meth:`forward` are filled with the model's defaults, repeated to the batch size. A pose is an axis-angle vector
    of 156 entries: 3 for the global orientation, 63 for the 21 body joints and 45 for each hand, in the order of
    ``SMPLH_BONE_ORDER_NAMES``.

    Attributes:
        joint_names (List[str]): Names of the 52 SMPL-H joints.
        joint_axes (dict): Joint axes per joint name, a 3x3 identity matrix each.
        joint_dofs (dict): Names of the degrees of freedom per joint name.
        joint_range (dict): Joint range per joint name, shape ``(3, 2)``.
        contype (dict): Contact type mapped to joint names.
        conaffinity (dict): Contact affinity mapped to joint names.
        zero_pose (torch.Tensor): Pose with all joint rotations set to zero, shape ``(1, 156)``.

    """

    def __init__(self, *args, **kwargs):
        """
        SMPL-H model constructor.

        Args:
            *args: Positional arguments passed to ``smplx.SMPLH``.
            **kwargs: Keyword arguments passed to ``smplx.SMPLH``, e.g. ``model_path`` (str, path to the folder or to
                the file where the model parameters are stored) and ``gender`` (str, which gender to load).

        """
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
        """
        Run the SMPL-H model.

        Arguments that are not given are replaced by the model's default values, repeated to the batch size of the
        given arguments.

        Args:
            betas (torch.Tensor): Shape coefficients, shape ``(batch_size, n_betas)``.
            body_pose (torch.Tensor): Axis-angle rotations of the body joints, shape ``(batch_size, 63)``.
            global_orient (torch.Tensor): Axis-angle global orientation, shape ``(batch_size, 3)``.
            left_hand_pose (torch.Tensor): Pose of the left hand in the representation expected by ``smplx.SMPLH``.
            right_hand_pose (torch.Tensor): Pose of the right hand in the representation expected by ``smplx.SMPLH``.
            **kwargs: Further keyword arguments of ``smplx.SMPLH.forward``, e.g. ``transl``.

        Returns:
            The output of ``smplx.SMPLH.forward``, with the fields ``vertices`` and ``joints``.

        """
        names = ("betas", "body_pose", "global_orient", "left_hand_pose", "right_hand_pose")
        defaults = _batch_defaults(self, names, dict(betas=betas, body_pose=body_pose, global_orient=global_orient,
                                                     left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose))
        smpl_output = super(SMPLH_Parser, self).forward(**defaults, **kwargs)
        return smpl_output

    def get_joints_verts(self, pose, th_betas=None, th_trans=None):
        """
        Compute the mesh vertices and the joint positions for a batch of poses.

        The global orientation and the body pose are taken from the first 66 entries of ``pose``.

        Args:
            pose (torch.Tensor): Axis-angle poses, shape ``(batch_size, 156)`` or reshapeable to it.
            th_betas (torch.Tensor): Shape coefficients, shape ``(batch_size, n_betas)``. Defaults to the model's
                shape.
            th_trans (torch.Tensor): Root translations, shape ``(batch_size, 3)``.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The vertices, shape ``(batch_size, n_vertices, 3)``, and the joint
            positions as returned by ``smplx.SMPLH``, shape ``(batch_size, n_joints, 3)``.

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
        """
        Compute the global rigid transformations of all joints for a batch of poses.

        This is the forward kinematics of the SMPL-H skeleton without skinning the mesh. The retargeting pipeline uses
        the resulting joint positions and orientations as targets for the mimic sites of a robot. The global orientation
        and the body pose are taken from the first 66 entries of ``pose``.

        Args:
            pose (torch.Tensor): Axis-angle poses, shape ``(batch_size, 156)`` or reshapeable to it, e.g.
                ``(batch_size, 52, 3)``.
            th_betas (torch.Tensor): Shape coefficients, shape ``(batch_size, n_betas)``. Defaults to the model's
                shape.
            th_trans (torch.Tensor): Root translations, shape ``(batch_size, 3)``, added to the position of every
                joint.

        Returns:
            torch.Tensor: Homogeneous transformations from each joint frame to the world frame, shape
            ``(batch_size, 52, 4, 4)``. The upper left 3x3 block is the global rotation and the last column the global
            position of the joint.

        """

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
            """
            Compute the global joint transformations from shape and pose parameters.

            Follows the first steps of ``smplx.SMPLH.forward`` (shape blend shapes, joint regression, conversion of the pose
            to rotation matrices) and then chains the transformations along the kinematic tree instead of skinning the mesh.
            Arguments that are not given are replaced by the model's default values. The translation is not applied here.

            Args:
                betas (torch.Tensor): Shape coefficients, shape ``(batch_size, n_betas)``.
                global_orient (torch.Tensor): Axis-angle global orientation, shape ``(batch_size, 3)``.
                body_pose (torch.Tensor): Axis-angle rotations of the body joints, shape ``(batch_size, 63)``.
                left_hand_pose (torch.Tensor): Pose of the left hand in the representation expected by ``smplx.SMPLH``.
                right_hand_pose (torch.Tensor): Pose of the right hand in the representation expected by ``smplx.SMPLH``.
                transl (torch.Tensor): Not applied.
                return_verts (bool): Not used.
                return_full_pose (bool): Not used.
                pose2rot (bool): If True, the pose is given as axis-angle vectors; otherwise as rotation matrices.
                **kwargs: Ignored.

            Returns:
                torch.Tensor: Global joint transformations, shape ``(batch_size, n_joints, 4, 4)``.

            """

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
        Compute the global rigid transformations of the joints of a kinematic tree.

        Args:
            rot_mats (torch.Tensor): Local rotation matrices of the joints, shape ``(batch_size, n_joints, 3, 3)``.
            joints (torch.Tensor): Joint positions in the rest pose, shape ``(batch_size, n_joints, 3)``.
            parents (torch.Tensor): Index of the parent of each joint, shape ``(n_joints,)``.
            dtype (torch.dtype): Not used.

        Returns:
            torch.Tensor: Homogeneous transformations from each joint frame to the world frame, shape
            ``(batch_size, n_joints, 4, 4)``.

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
        """
        Get the rest-pose skeleton of the model for a body shape.

        Args:
            v_template: Not used.
            zero_pose (torch.Tensor): Pose in which the skeleton is evaluated, shape ``(1, 156)``. Defaults to the
                zero pose.
            betas (torch.Tensor): Shape coefficients, shape ``(1, n_betas)``. Defaults to zeros (the mean shape).
            flatfoot (bool): Not used.

        Returns:
            tuple: A tuple of eight elements: the vertices of the mesh, shape ``(n_vertices, 3)``; the joint
            positions, shape ``(n_joints, 3)``; the linear blend skinning weights, shape ``(n_vertices, n_joints)``; the
            joint names; a dictionary mapping each joint name to its position relative to its parent (the absolute
            position for the root); a dictionary mapping each joint name to the name of its parent; the rotation
            channel order ``["z", "y", "x"]``; and the dictionary of joint ranges.

        """
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
        """
        Get the mesh, the skeleton and the joint properties of the model for a body shape.

        Args:
            zero_pose (torch.Tensor): Pose in which the model is evaluated, shape ``(1, 156)``. Defaults to the zero
                pose.
            betas (torch.Tensor): Shape coefficients, shape ``(1, n_betas)``. Defaults to zeros (the mean shape).
            flatfoot (bool): If True, the lowest vertices of the mesh (within 1 cm of the minimum along the y-axis) are
                moved to a common height, which flattens the soles.

        Returns:
            tuple: A tuple of eleven elements:

            - ``verts``: Vertices of the mesh, shape ``(n_vertices, 3)``.
            - ``joint_pos``: Joint positions, shape ``(n_joints, 3)``.
            - ``skin_weights``: Linear blend skinning weights, shape ``(n_vertices, n_joints)``.
            - ``joint_names``: Names of the joints.
            - ``joint_offsets``: Dictionary mapping each joint name to its position relative to its parent (the
              absolute position for the root).
            - ``joint_parents``: Dictionary mapping each joint name to the name of its parent (``None`` for the root).
            - ``joint_axes``: Dictionary mapping each joint name to its joint axes, a 3x3 identity matrix.
            - ``joint_dofs``: Dictionary mapping each joint name to the names of its degrees of freedom.
            - ``joint_range``: Dictionary mapping each joint name to its joint range, shape ``(3, 2)``.
            - ``contype``: Dictionary mapping the contact type ``1`` to the joint names.
            - ``conaffinity``: Dictionary mapping the contact affinity ``1`` to the joint names.

        """
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
        """
        Get the mesh, the skeleton and the joint properties of the model in the zero pose for a batch of body shapes.

        Args:
            betas (torch.Tensor): Shape coefficients, shape ``(batch_size, n_betas)``. Defaults to zeros (the mean
                shape).
            flatfoot (bool): If True, the lowest vertices of the mesh (within 1 cm of the minimum along the y-axis) are
                moved to a common height, which flattens the soles.

        Returns:
            tuple: A tuple of eleven elements:

            - ``verts``: Vertices of the mesh of the first body shape, shape ``(n_vertices, 3)``.
            - ``joint_pos``: Joint positions, shape ``(batch_size, n_joints, 3)``.
            - ``skin_weights``: Linear blend skinning weights, shape ``(n_vertices, n_joints)``.
            - ``joint_names``: Names of the joints.
            - ``joint_offsets``: Dictionary mapping each joint name to its position relative to its parent (the
              absolute position for the root).
            - ``joint_parents``: Dictionary mapping each joint name to the name of its parent (``None`` for the root).
            - ``joint_axes``: Dictionary mapping each joint name to its joint axes, a 3x3 identity matrix.
            - ``joint_dofs``: Dictionary mapping each joint name to the names of its degrees of freedom.
            - ``joint_range``: Dictionary mapping each joint name to its joint range, shape ``(3, 2)``.
            - ``contype``: Dictionary mapping the contact type ``1`` to the joint names.
            - ``conaffinity``: Dictionary mapping the contact affinity ``1`` to the joint names.

        Unlike in :meth:`get_mesh_offsets`, the joint positions, the joint offsets and the skinning weights are
        torch tensors, and the joint offsets have a leading batch dimension.

        """
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
