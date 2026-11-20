"""
Trajectory-similarity metrics for validating imitation policies.

During validation, a policy is rolled out in parallel environments and every rollout is compared with the reference
motion it tracks, as two time series of one kinematic quantity (joint positions, body positions, site velocities,
...). Three distance measures from the JAX-native ``metrx`` library are supported:

- ``"EuclideanDistance"``: mean Euclidean distance between rollout and reference at the same time step. It also
  penalizes a motion that is correct but early or late.
- ``"DynamicTimeWarping"``: aligns the two series with a monotone time warp and sums the distances along the best
  alignment, which makes it insensitive to timing offsets.
- ``"DiscreteFrechetDistance"``: uses the same kind of alignment, but takes the largest distance along it, i.e. it
  reports the worst deviation rather than the accumulated one.

:class:`MetricsHandler` extracts the quantities from a batch of environment states and from the reference trajectory
and evaluates the measures; the result is a :class:`ValidationSummary`. The names accepted in the configuration are
listed in ``SUPPORTED_QUANTITIES`` and ``SUPPORTED_MEASURES``.
"""

from omegaconf import OmegaConf, DictConfig

import jax
import jax.numpy as jnp
from flax import struct
from jax.scipy.spatial.transform import Rotation as R

from metrx import DistanceMeasures
import mujoco
from loco_mujoco.core.wrappers import SummaryMetrics
from loco_mujoco.core.utils.math import calc_site_velocities, calculate_relative_site_quatities, quat_scalarfirst2scalarlast
from loco_mujoco.core.utils.mujoco import mj_jntid2qposid, mj_jntid2qvelid, mj_jntname2qposid, mj_jntname2qvelid


SUPPORTED_QUANTITIES = ["JointPosition", "JointVelocity", "BodyPosition", "BodyVelocity", "BodyOrientation",
                        "SitePosition", "SiteVelocity",
                        "SiteOrientation", "RelSitePosition", "RelSiteVelocity", "RelSiteOrientation"]

SUPPORTED_MEASURES = ["EuclideanDistance", "DynamicTimeWarping", "DiscreteFrechetDistance"]


@struct.dataclass
class QuantityContainer:
    """
    Container holding one array per supported kinematic quantity.

    The same structure is used for three purposes: the quantities of the rollout, the corresponding quantities of the
    reference trajectory, and the value of one distance measure per quantity (see :class:`ValidationSummary`). A
    quantity that was not requested is an empty array.

    Attributes:
        qpos (jnp.ndarray): Joint positions (``"JointPosition"``).
        qvel (jnp.ndarray): Joint velocities (``"JointVelocity"``).
        xpos (jnp.ndarray): Body positions in the world frame (``"BodyPosition"``).
        xrotvec (jnp.ndarray): Body orientations in the world frame as rotation vectors (``"BodyOrientation"``).
        cvel (jnp.ndarray): 6D body velocities, MuJoCo's ``cvel`` (``"BodyVelocity"``).
        site_xpos (jnp.ndarray): Site positions in the world frame (``"SitePosition"``).
        site_xrotvec (jnp.ndarray): Site orientations in the world frame as rotation vectors
            (``"SiteOrientation"``).
        site_xvel (jnp.ndarray): 6D site velocities (``"SiteVelocity"``).
        site_rpos (jnp.ndarray): Site positions relative to the main site (``"RelSitePosition"``).
        site_rrotvec (jnp.ndarray): Site orientations relative to the main site as rotation vectors
            (``"RelSiteOrientation"``).
        site_rvel (jnp.ndarray): 6D site velocities relative to the main site (``"RelSiteVelocity"``).

    """
    qpos: jnp.ndarray = struct.field(default_factory=lambda: jnp.array([]))
    qvel: jnp.ndarray = struct.field(default_factory=lambda: jnp.array([]))
    xpos: jnp.ndarray = struct.field(default_factory=lambda: jnp.array([]))
    xrotvec: jnp.ndarray = struct.field(default_factory=lambda: jnp.array([]))
    cvel: jnp.ndarray = struct.field(default_factory=lambda: jnp.array([]))
    site_xpos: jnp.ndarray = struct.field(default_factory=lambda: jnp.array([]))
    site_xrotvec: jnp.ndarray = struct.field(default_factory=lambda: jnp.array([]))
    site_xvel: jnp.ndarray = struct.field(default_factory=lambda: jnp.array([]))
    site_rpos: jnp.ndarray = struct.field(default_factory=lambda: jnp.array([]))
    site_rrotvec: jnp.ndarray = struct.field(default_factory=lambda: jnp.array([]))
    site_rvel: jnp.ndarray = struct.field(default_factory=lambda: jnp.array([]))


@struct.dataclass
class ValidationSummary(SummaryMetrics):
    """
    Result of a validation run: the default summary metrics plus the trajectory-similarity measures.

    Extends :class:`~loco_mujoco.core.wrappers.mjx.SummaryMetrics` (mean episode return, mean episode length and
    maximum time step) by one :class:`QuantityContainer` per distance measure. Each field of such a container is the
    scalar value of the measure for that quantity, averaged over the elements of the quantity (joints, bodies or sites)
    and over the validation environments, or an empty array if the quantity was not requested. A measure that was not
    requested is reported as zero for the requested quantities.

    Attributes:
        euclidean_distance (QuantityContainer): Mean Euclidean distance between rollout and reference at the same
            time step, per quantity.
        dynamic_time_warping (QuantityContainer): Dynamic-time-warping distance between rollout and reference, per
            quantity.
        discrete_frechet_distance (QuantityContainer): Discrete Fréchet distance between rollout and reference, per
            quantity.

    """
    euclidean_distance: QuantityContainer = struct.field(default_factory=QuantityContainer)
    dynamic_time_warping: QuantityContainer = struct.field(default_factory=QuantityContainer)
    discrete_frechet_distance: QuantityContainer = struct.field(default_factory=QuantityContainer)


class MetricsHandler:
    """
    Computes validation metrics from a batch of rollouts and the reference trajectories they track.

    The handler is configured through the ``experiment.validation`` node of the training configuration and is passed
    to the training and evaluation functions of an algorithm (e.g.
    :meth:`~loco_mujoco.algorithms.ppo_jax.PPOJax.build_train_fn`), which call it on the environment states collected
    during a validation rollout. The call returns a :class:`ValidationSummary` and is compatible with ``jax.jit``.

    The following entries of ``experiment.validation`` are read:

    - ``quantities`` (list of str): Quantities to compare; names from ``SUPPORTED_QUANTITIES``.
    - ``measures`` (list of str): Distance measures to evaluate; names from ``SUPPORTED_MEASURES``. Requires an
      environment with a loaded trajectory.
    - ``rel_joint_names`` (list of str, optional): Joints used for the joint quantities. Defaults to all joints.
    - ``joints_to_ignore`` (list of str, optional): Joints excluded from the joint quantities.
    - ``rel_body_names`` (list of str, optional): Bodies used for the body quantities. Defaults to all bodies.
    - ``rel_site_names`` (list of str, optional): Sites used for the site quantities. Defaults to all sites. For
      the relative site quantities, the first site is the main site with respect to which the others are expressed.

    Attributes:
        supported_measures (List[str]): Names of the supported distance measures.
        supported_quantities (List[str]): Names of the supported quantities.
        quantaties (List[str]): Quantities selected in the configuration (the attribute name is spelled this way).
        measures (List[str]): Distance measures selected in the configuration.
        rel_qpos_ids (jnp.ndarray): Indices into ``qpos`` of the relevant joints.
        rel_qvel_ids (jnp.ndarray): Indices into ``qvel`` of the relevant joints.
        rel_body_ids (jnp.ndarray): Ids of the relevant bodies.
        rel_site_ids (jnp.ndarray): Ids of the relevant sites.

    """

    supported_measures = SUPPORTED_MEASURES
    supported_quantities = SUPPORTED_QUANTITIES

    def __init__(self, config: DictConfig, env):
        """
        Initialize the handler from the training configuration and the environment.

        Args:
            config (DictConfig): Training configuration. Its ``experiment`` node must provide ``num_envs`` and may
                provide the ``validation`` entries described in the class docstring.
            env: Environment the policy is validated in, a :class:`~loco_mujoco.environments.base.LocoEnv`. Its model
                is used to resolve joint, body and site names, and its trajectory data serves as the reference. A
                trajectory must be loaded if any distance measure is requested.

        Raises:
            AssertionError: If a requested measure or quantity is not supported, a body or site name is not found in
                the model, or measures are requested although the environment has no trajectory.

        """

        self._config = config.experiment

        if env.th is not None:
            self._traj_data = env._traj.data
        else:
            self._traj_data = None

        self.quantaties = OmegaConf.select(self._config, "validation.quantities")
        self.measures = OmegaConf.select(self._config, "validation.measures")

        rel_joint_names = OmegaConf.select(self._config, "validation.rel_joint_names")
        joints_to_ignore = OmegaConf.select(self._config, "validation.joints_to_ignore")
        rel_body_names = OmegaConf.select(self._config, "validation.rel_body_names")
        rel_site_names = OmegaConf.select(self._config, "validation.rel_site_names")

        if joints_to_ignore is None:
            joints_to_ignore = []

        model = env.get_model()
        if rel_joint_names is not None:
            self.rel_qpos_ids = [jnp.array(mj_jntid2qposid(name, model)) for name in rel_joint_names
                                 if name not in joints_to_ignore]
            self.rel_qvel_ids = [jnp.array(mj_jntid2qvelid(name, model)) for name in rel_joint_names
                                 if name not in joints_to_ignore]
        else:
            self.rel_qpos_ids = [jnp.array(mj_jntid2qposid(i, model)) for i in range(model.njnt)
                                 if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) not in joints_to_ignore]
            self.rel_qvel_ids = [jnp.array(mj_jntid2qvelid(i, model)) for i in range(model.njnt)
                                 if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) not in joints_to_ignore]

        if rel_body_names is not None:
            self.rel_body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                                 for name in rel_body_names]
            assert -1 not in self.rel_body_ids, f"Body {rel_body_names[self.rel_body_ids.index(-1)]} not found."
        else:
            self.rel_body_ids = [i for i in range(model.nbody)]

        if rel_site_names is not None:
            self.rel_site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
                                 for name in rel_site_names]
            assert -1 not in self.rel_site_ids, f"Site {rel_site_names[self.rel_site_ids.index(-1)]} not found."
        else:
            self.rel_site_ids = [i for i in range(model.nsite)]

        self._site_bodyid = jnp.array([model.site_bodyid[i] for i in range(model.nsite)]) # get the body id of all sites
        self._body_rootid = jnp.array(model.body_rootid) # get the root body id for all bodies

        if self.measures is not None:
            assert self._traj_data is not None, "Trajectory data is required for calculating measures."
            for m in self.measures:
                assert m in SUPPORTED_MEASURES, f"{m} is not a supported measure."

            dummy_func = lambda x, y: 0.0
            self._euclidean_distance = jax.vmap(jax.vmap(DistanceMeasures.create_instance("EuclideanDistance", mean=True),
                                                in_axes=(0, 0)), in_axes=(0, 0)) \
                if "EuclideanDistance" in self.measures else dummy_func
            self._dynamic_time_warping = jax.vmap(jax.vmap(DistanceMeasures.create_instance("DynamicTimeWarping"),
                                                  in_axes=(0, 0)), in_axes=(0, 0)) \
                if "DynamicTimeWarping" in self.measures else dummy_func
            self._discrete_frechet_distance = jax.vmap(jax.vmap(DistanceMeasures.create_instance("DiscreteFrechetDistance"),
                                                       in_axes=(0, 0)), in_axes=(0, 0))\
                if "DiscreteFrechetDistance" in self.measures else dummy_func

        if self.quantaties is not None:
            for q in self.quantaties:
                assert q in SUPPORTED_QUANTITIES, f"{q} is not a supported quantity."

                if "Rel" in self.quantaties:
                    assert self.rel_site_ids is not None, ("Relative site quantities requires relative site ids with "
                                                           "the first site being the site used to calculate the "
                                                           "relative quantities.")

        self._vec_calc_site_velocities = jax.vmap(jax.vmap(calc_site_velocities, in_axes=(None, 0, None, None, None, None)), in_axes=(None, 0, None, None, None, None))
        self._vec_calc_rel_site_quantities = jax.vmap(jax.vmap(calculate_relative_site_quatities, in_axes=(0, None, None, None, None)), in_axes=(0, None, None, None, None))

        # determine the quaternions in qpos
        self._quat_in_qpos = jnp.concatenate([jnp.array([False]*3+[True]*4) if len(j) == 7 else jnp.array([False])
                                              for j in self.rel_qpos_ids])
        self._not_quat_in_qpos = jnp.invert(self._quat_in_qpos)
        self.rel_qpos_ids = jnp.concatenate(self.rel_qpos_ids)
        self.rel_qvel_ids = jnp.concatenate(self.rel_qvel_ids)
        self.rel_body_ids = jnp.array(self.rel_body_ids)
        self.rel_site_ids = jnp.array(self.rel_site_ids)

    def __call__(self, env_states):
        """
        Compute the validation summary for a batch of rollouts.

        All requested quantities are extracted from the simulation states and from the reference trajectory at the
        trajectory indices stored in the environment states. Every element of a quantity (e.g. one joint or one body) is
        then treated as a time series over the rollout steps, each requested measure is evaluated between rollout and
        reference per element and per environment, and the result is averaged over elements and environments.

        Args:
            env_states: Batched environment states of a validation rollout, as stacked by ``jax.lax.scan`` over the
                environment steps. All leaves have the leading dimensions ``(num_steps, num_envs)``. The states must
                carry the logging metrics (``env_states.metrics``), the simulation data (``env_states.data``) and the
                trajectory state (``env_states.additional_carry.traj_state``).

        Returns:
            ValidationSummary: Mean return and length of the episodes that finished during the rollout, the maximum
            time step (scaled by ``num_envs``), and one scalar per requested quantity for each distance measure.

        """

        # calculate default metrics
        logged_metrics = env_states.metrics
        mean_episode_return = jnp.sum(
            jnp.where(logged_metrics.done, logged_metrics.returned_episode_returns, 0.0)) / jnp.sum(
            logged_metrics.done)
        mean_episode_length = jnp.sum(
            jnp.where(logged_metrics.done, logged_metrics.returned_episode_lengths, 0.0)) / jnp.sum(
            logged_metrics.done)
        max_timestep = jnp.max(logged_metrics.timestep * self._config.num_envs)

        # get all quantities
        if "JointPosition" in self.quantaties:
            qpos, traj_qpos = self.get_joint_positions(env_states)
            # extend last dim
            qpos = jnp.expand_dims(qpos, axis=-1)
            traj_qpos = jnp.expand_dims(traj_qpos, axis=-1)
        else:
            qpos = traj_qpos = jnp.empty(0)
        if "JointVelocity" in self.quantaties:
            qvel, traj_qvel = self.get_joint_velocities(env_states)
            # extend last dim
            qvel = jnp.expand_dims(qvel, axis=-1)
            traj_qvel = jnp.expand_dims(traj_qvel, axis=-1)
        else:
            qvel = traj_qvel = jnp.empty(0)
        if "BodyPosition" in self.quantaties:
            xpos, traj_xpos = self.get_body_positions(env_states)
        else:
            xpos = traj_xpos = jnp.empty(0)
        if "BodyOrientation" in self.quantaties:
            xrotvec, traj_xrotvec = self.get_body_orientations(env_states)
        else:
            xrotvec = traj_xrotvec = jnp.empty(0)
        if "BodyVelocity" in self.quantaties:
            cvel, traj_cvel = self.get_body_velocities(env_states)
        else:
            cvel = traj_cvel = jnp.empty(0)
        if "SitePosition" in self.quantaties:
            site_xpos, traj_site_xpos = self.get_site_positions(env_states)
        else:
            site_xpos = traj_site_xpos = jnp.empty(0)
        if "SiteOrientation" in self.quantaties:
            site_xrotvec, traj_site_xrotvec = self.get_site_orientations(env_states)
        else:
            site_xrotvec = traj_site_xrotvec = jnp.empty(0)
        if "SiteVelocity" in self.quantaties:
            site_xvel, traj_site_xvel = self.get_site_velocities(env_states)
        else:
            site_xvel = traj_site_xvel = jnp.empty(0)
        if ("RelSitePosition" in self.quantaties or "RelSiteOrientation" in self.quantaties
                or "RelSiteVelocity" in self.quantaties):
            (rel_site_pos, rel_site_rotvec, rel_site_vel,
             traj_rel_site_pos, traj_rel_site_rotvec, traj_rel_site_vel) = self.get_relative_site_quantities(env_states)
        else:
            rel_site_pos = rel_site_rotvec = rel_site_vel = traj_rel_site_pos =\
                traj_rel_site_rotvec = traj_rel_site_vel = jnp.empty(0)

        # create containers
        container = QuantityContainer(qpos=qpos, qvel=qvel, xpos=xpos, xrotvec=xrotvec, cvel=cvel,
                                      site_xpos=site_xpos, site_xrotvec=site_xrotvec, site_xvel=site_xvel,
                                      site_rpos=rel_site_pos, site_rrotvec=rel_site_rotvec, site_rvel=rel_site_vel)
        container_traj = QuantityContainer(qpos=traj_qpos, qvel=traj_qvel, xpos=traj_xpos, xrotvec=traj_xrotvec,
                                           cvel=traj_cvel, site_xpos=traj_site_xpos, site_xrotvec=traj_site_xrotvec,
                                           site_xvel=traj_site_xvel, site_rpos=traj_rel_site_pos,
                                           site_rrotvec=traj_rel_site_rotvec, site_rvel=traj_rel_site_vel)

        # the dimensions for each quantity is (S, N, D) where S is the number of samples, N is the number of elements
        # (e.g., joints, bodies, site) and D is the dimension of the quantity (e.g., position, velocity, orientation).
        # We want to switch the dimensions to (N, S, D) to calculate the distance measures on trajectories.

        container = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 2) if x.size > 0 else x, container)
        container_traj = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 2) if x.size > 0 else x, container_traj)

        return ValidationSummary(
            mean_episode_return=mean_episode_return,
            mean_episode_length=mean_episode_length,
            max_timestep=max_timestep,
            euclidean_distance=jax.tree.map(lambda x, y: jnp.mean(self._euclidean_distance(x, y)) if x.size > 0 else x, container, container_traj),
            dynamic_time_warping=jax.tree.map(lambda x, y: jnp.mean(self._dynamic_time_warping(x, y)) if x.size > 0 else x, container, container_traj),
            discrete_frechet_distance=jax.tree.map(lambda x, y: jnp.mean(self._discrete_frechet_distance(x, y)) if x.size > 0 else x, container, container_traj))

    def get_joint_positions(self, env_states):
        """
        Get the positions of the relevant joints from the rollout and from the reference trajectory.

        The orientation quaternion of a free joint is converted to a rotation vector, so that all entries live in a
        Euclidean space. The three rotation-vector entries are placed after all other entries.

        Args:
            env_states: Batched environment states with leading dimensions ``(num_steps, num_envs)``.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Joint positions of the rollout and of the reference, each of shape
            ``(num_steps, num_envs, n)``, where ``n`` is the number of relevant ``qpos`` entries (minus one if a free
            joint is included, since its quaternion becomes a rotation vector).

        """
        # get from data
        qpos = env_states.data.qpos

        # get from trajectory
        traj_qpos = self._traj_data.qpos[self.get_traj_indices(env_states)]
        # filter for relevant joints
        qpos, traj_qpos = qpos[..., self.rel_qpos_ids], traj_qpos[..., self.rel_qpos_ids]

        # there might be quaternions due to free joints, so we need to convert them
        # to rotation vector to use metrics in the Euclidean space
        quat, quat_traj = qpos[..., self._quat_in_qpos], traj_qpos[..., self._quat_in_qpos]
        quat, quat_traj = quat.reshape(-1, 4), quat_traj.reshape(-1, 4)
        quat, quat_traj = quat_scalarfirst2scalarlast(quat), quat_scalarfirst2scalarlast(quat_traj)
        rot_vec, rot_vec_traj = R.from_quat(quat).as_rotvec(), R.from_quat(quat_traj).as_rotvec()
        qpos = jnp.concatenate([qpos[..., self._not_quat_in_qpos],
                                rot_vec.reshape((*qpos.shape[:-1], 3))], axis=-1)
        traj_qpos = jnp.concatenate([traj_qpos[..., self._not_quat_in_qpos],
                                     rot_vec_traj.reshape((*traj_qpos.shape[:-1], 3))], axis=-1)

        return qpos, traj_qpos

    def get_joint_velocities(self, env_states):
        """
        Get the velocities of the relevant joints from the rollout and from the reference trajectory.

        Args:
            env_states: Batched environment states with leading dimensions ``(num_steps, num_envs)``.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Joint velocities of the rollout and of the reference, each of shape
            ``(num_steps, num_envs, n_rel_qvel)``.

        """
        # get from data
        qvel = env_states.data.qvel

        # get from trajectory
        traj_qvel = self._traj_data.qvel[self.get_traj_indices(env_states)]

        return qvel[..., self.rel_qvel_ids], traj_qvel[..., self.rel_qvel_ids]

    def get_body_positions(self, env_states):
        """
        Get the world-frame positions of the relevant bodies from the rollout and from the reference trajectory.

        Args:
            env_states: Batched environment states with leading dimensions ``(num_steps, num_envs)``.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Body positions of the rollout and of the reference, each of shape
            ``(num_steps, num_envs, n_rel_bodies, 3)``.

        """

        # get from data
        body_pos = env_states.data.xpos

        # get from trajectory
        traj_body_pos = self._traj_data.xpos[self.get_traj_indices(env_states)]

        return body_pos[..., self.rel_body_ids], traj_body_pos[..., self.rel_body_ids]

    def get_body_orientations(self, env_states):
        """
        Get the world-frame orientations of the relevant bodies from the rollout and from the reference trajectory.

        The body quaternions are converted to rotation vectors.

        Args:
            env_states: Batched environment states with leading dimensions ``(num_steps, num_envs)``.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Body orientations of the rollout and of the reference as rotation
            vectors, each of shape ``(num_steps, num_envs, n_rel_bodies, 3)``.

        """
        # get from data
        xquat_env = quat_scalarfirst2scalarlast(env_states.data.xquat)
        body_rotvec = R.from_quat(xquat_env).as_rotvec()

        # get from trajectory
        xquat_traj = quat_scalarfirst2scalarlast(self._traj_data.xquat[self.get_traj_indices(env_states)])
        traj_body_rotvec = R.from_quat(xquat_traj).as_rotvec()

        return body_rotvec[..., self.rel_body_ids], traj_body_rotvec[..., self.rel_body_ids]

    def get_body_velocities(self, env_states):
        """
        Get the 6D velocities of the relevant bodies from the rollout and from the reference trajectory.

        The velocities are MuJoCo's ``cvel`` (rotational part first, then translational part).

        Args:
            env_states: Batched environment states with leading dimensions ``(num_steps, num_envs)``.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Body velocities of the rollout and of the reference, each of shape
            ``(num_steps, num_envs, n_rel_bodies, 6)``.

        """
        # get from data
        body_vel = env_states.data.cvel

        # get from trajectory
        traj_body_vel = self._traj_data.cvel[self.get_traj_indices(env_states)]

        return body_vel[..., self.rel_body_ids], traj_body_vel[..., self.rel_body_ids]

    def get_site_positions(self, env_states):
        """
        Get the world-frame positions of the relevant sites from the rollout and from the reference trajectory.

        Args:
            env_states: Batched environment states with leading dimensions ``(num_steps, num_envs)``.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Site positions of the rollout and of the reference, each of shape
            ``(num_steps, num_envs, n_rel_sites, 3)``.

        """
        # get from data
        site_pos = env_states.data.site_xpos

        # get from trajectory
        traj_site_pos = self._traj_data.site_xpos[self.get_traj_indices(env_states)]

        return site_pos[..., self.rel_site_ids], traj_site_pos[..., self.rel_site_ids]

    def get_site_orientations(self, env_states):
        """
        Get the world-frame orientations of the relevant sites from the rollout and from the reference trajectory.

        The site rotation matrices are converted to rotation vectors.

        Args:
            env_states: Batched environment states with leading dimensions ``(num_steps, num_envs)``.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Site orientations of the rollout and of the reference as rotation
            vectors, each of shape ``(num_steps, num_envs, n_rel_sites, 3)``.

        """
        # get from data
        site_rotvec = R.from_matrix(env_states.data.site_xmat).as_rotvec()

        # get from trajectory
        site_xmat = self._traj_data.site_xmat
        assert len(site_xmat.shape) == 3
        site_xmat = site_xmat.reshape(site_xmat.shape[0], site_xmat.shape[1], 3, 3)
        traj_site_rotvec = R.from_matrix(site_xmat[self.get_traj_indices(env_states)]).as_rotvec()

        return site_rotvec[..., self.rel_site_ids], traj_site_rotvec[..., self.rel_site_ids]

    def get_site_velocities(self, env_states):
        """
        Get the 6D world-frame velocities of the relevant sites from the rollout and from the reference trajectory.

        The velocities are computed with :func:`~loco_mujoco.core.utils.math.calc_site_velocities` from the velocity of
        the body each site is attached to.

        Args:
            env_states: Batched environment states with leading dimensions ``(num_steps, num_envs)``.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Site velocities of the rollout and of the reference, each of shape
            ``(num_steps, num_envs, n_rel_sites, 6)``.

        """

        site_xvel = self._vec_calc_site_velocities(self.rel_site_ids, env_states.data,
                                                   self._site_bodyid[self.rel_site_ids],
                                                   self._body_rootid[self.rel_site_ids], jnp, False)

        traj_indices = self.get_traj_indices(env_states)
        traj_data = jax.tree.map(lambda x: x[traj_indices], self._traj_data)
        traj_site_xvel = self._vec_calc_site_velocities(self.rel_site_ids, traj_data,
                                                        self._site_bodyid[self.rel_site_ids],
                                                        self._body_rootid[self.rel_site_ids], jnp, False)

        return site_xvel[..., self.rel_site_ids], traj_site_xvel[..., self.rel_site_ids]

    def get_relative_site_quantities(self, env_states):
        """
        Get the positions, orientations and velocities of the relevant sites relative to the main site.

        The main site is the first of the relevant sites. The quantities are computed with
        :func:`~loco_mujoco.core.utils.math.calculate_relative_site_quatities` for the rollout and for the reference
        trajectory; the main site itself is not part of the result.

        Args:
            env_states: Batched environment states with leading dimensions ``(num_steps, num_envs)``.

        Returns:
            Tuple[jnp.ndarray, ...]: Six arrays: relative site positions, relative site orientations (rotation vectors)
            and relative 6D site velocities of the rollout, followed by the same three quantities of the reference
            trajectory. The leading dimensions are ``(num_steps, num_envs, n_rel_sites - 1)``.

        """

        rel_site_pos, rel_site_rotvec, rel_site_vel = self._vec_calc_rel_site_quantities(
            env_states.data, self.rel_site_ids, self._site_bodyid[self.rel_site_ids],
            self._body_rootid[self.rel_site_ids], jnp)

        traj_states = env_states.additional_carry.traj_state
        traj_data = self._traj_data.get(traj_states.traj_no, traj_states.subtraj_step_no)
        traj_rel_site_pos, traj_rel_site_rotvec, traj_rel_site_vel = self._vec_calc_rel_site_quantities(
            traj_data, self.rel_site_ids, self._site_bodyid[self.rel_site_ids],
            self._body_rootid[self.rel_site_ids], jnp)

        return (rel_site_pos[..., self.rel_site_ids], rel_site_rotvec[..., self.rel_site_ids],
                rel_site_vel[..., self.rel_site_ids], traj_rel_site_pos[..., self.rel_site_ids],
                traj_rel_site_rotvec[..., self.rel_site_ids], traj_rel_site_vel[..., self.rel_site_ids])

    def get_traj_indices(self, env_states):
        """
        Get the indices of the current reference samples in the flat trajectory data.

        Args:
            env_states: Batched environment states carrying the trajectory state in
                ``additional_carry.traj_state``.

        Returns:
            jnp.ndarray: Index of the current sample of each environment state, i.e. the start index of its trajectory
            (``split_points[traj_no]``) plus the step within the trajectory (``subtraj_step_no``). Same batch shape as
            the environment states.

        """
        traj_states = env_states.additional_carry.traj_state
        start_idx = self._traj_data.split_points[traj_states.traj_no]
        return start_idx + traj_states.subtraj_step_no

    @property
    def requires_trajectory(self):
        """
        Whether the handler holds reference trajectory data.

        Returns:
            bool: ``True`` if the environment passed at construction has a trajectory loaded.

        """
        return self._traj_data is not None

    def get_zero_container(self):
        """
        Create a summary with the same structure as the one returned by :meth:`__call__`, filled with zeros.

        Used as the alternative branch of a ``jax.lax.cond`` in iterations in which no validation is run, so that both
        branches return the same pytree structure.

        Returns:
            ValidationSummary: Summary whose default metrics are zero and whose containers hold a scalar zero for
            every requested quantity and an empty array for all others.

        """

        def _zeros_if_exists(quantity_name):
            return jnp.array(0.0) if quantity_name in self.quantaties else jnp.empty(0)

        container = QuantityContainer(qpos=_zeros_if_exists("JointPosition"),
                                      qvel=_zeros_if_exists("JointVelocity"),
                                      xpos=_zeros_if_exists("BodyPosition"),
                                      xrotvec=_zeros_if_exists("BodyOrientation"),
                                      cvel=_zeros_if_exists("BodyVelocity"),
                                      site_xpos=_zeros_if_exists("SitePosition"),
                                      site_xrotvec=_zeros_if_exists("SiteOrientation"),
                                      site_xvel=_zeros_if_exists("SiteVelocity"),
                                      site_rpos=_zeros_if_exists("RelSitePosition"),
                                      site_rrotvec=_zeros_if_exists("RelSiteOrientation"),
                                      site_rvel=_zeros_if_exists("RelSiteVelocity"))

        return ValidationSummary(mean_episode_return=jnp.array(0.0), mean_episode_length=jnp.array(0.0),
                                 max_timestep=jnp.array(0), euclidean_distance=container,
                                 dynamic_time_warping=container, discrete_frechet_distance=container)
