"""Tests for the torch-free helper functions in loco_mujoco.smpl.retargeting.

These cover the config/path resolution, AMASS loading, robot-conf loading, and
hashing helpers -- none of which require the real SMPL model, AMASS datasets, or
a torch optimization pass. The heavy torch retargeting pipeline
(``fit_smpl_motion`` / ``fit_smpl_shape``) is covered separately as an
integration test driven against the synthetic SMPL model.
"""
import os
import pickle

import numpy as np
import pytest

# retargeting pulls in mujoco + LocoEnv; skip cleanly if that stack is absent.
retargeting = pytest.importorskip("loco_mujoco.smpl.retargeting")
import loco_mujoco  # noqa: E402


# --------------------------- create_multi_trajectory_hash ---------------------------

def test_hash_is_order_invariant():
    a = retargeting.create_multi_trajectory_hash(["walk", "run", "jump"])
    b = retargeting.create_multi_trajectory_hash(["jump", "walk", "run"])
    assert a == b


def test_hash_is_deterministic_and_sensitive():
    h1 = retargeting.create_multi_trajectory_hash(["walk"])
    h2 = retargeting.create_multi_trajectory_hash(["walk"])
    h3 = retargeting.create_multi_trajectory_hash(["walk2"])
    assert h1 == h2 and h1 != h3
    assert len(h1) == 64  # sha256 hex digest


# --------------------------- check_optional_imports ---------------------------

def test_check_optional_imports_raises_when_missing(monkeypatch):
    monkeypatch.setattr(retargeting, "_OPTIONAL_IMPORT_INSTALLED", False)
    monkeypatch.setattr(retargeting, "_OPTIONAL_IMPORT_EXCEPTION",
                        ImportError("boom"), raising=False)
    with pytest.raises(ImportError, match="Optional smpl"):
        retargeting.check_optional_imports()


def test_check_optional_imports_passes_when_present(monkeypatch):
    monkeypatch.setattr(retargeting, "_OPTIONAL_IMPORT_INSTALLED", True)
    retargeting.check_optional_imports()  # should not raise


# --------------------------- path getters ---------------------------

def _write_variables(tmp_path, **kv):
    import yaml
    p = tmp_path / "VARIABLES.yaml"
    p.write_text(yaml.safe_dump(kv))
    return p


@pytest.mark.parametrize("getter,key", [
    ("get_amass_dataset_path", "LOCOMUJOCO_AMASS_PATH"),
    ("get_converted_amass_dataset_path", "LOCOMUJOCO_CONVERTED_AMASS_PATH"),
    ("get_smpl_model_path", "LOCOMUJOCO_SMPL_MODEL_PATH"),
])
def test_path_getters_return_configured_value(tmp_path, monkeypatch, getter, key):
    p = _write_variables(tmp_path, **{key: "/some/configured/path"})
    monkeypatch.setattr(loco_mujoco, "PATH_TO_VARIABLES", str(p))
    assert getattr(retargeting, getter)() == "/some/configured/path"


@pytest.mark.parametrize("getter,key", [
    ("get_amass_dataset_path", "LOCOMUJOCO_AMASS_PATH"),
    ("get_converted_amass_dataset_path", "LOCOMUJOCO_CONVERTED_AMASS_PATH"),
    ("get_smpl_model_path", "LOCOMUJOCO_SMPL_MODEL_PATH"),
])
def test_path_getters_assert_on_empty(tmp_path, monkeypatch, getter, key):
    p = _write_variables(tmp_path, **{key: ""})
    monkeypatch.setattr(loco_mujoco, "PATH_TO_VARIABLES", str(p))
    with pytest.raises(AssertionError):
        getattr(retargeting, getter)()


# --------------------------- load_amass_data ---------------------------

def _write_amass_npz(path, n_frames=5, framerate_key="mocap_framerate"):
    np.savez(
        path,
        poses=np.zeros((n_frames, 72), dtype=np.float64),
        trans=np.zeros((n_frames, 3), dtype=np.float64),
        betas=np.zeros(16, dtype=np.float64),
        gender="neutral",
        **{framerate_key: np.array(30.0)},
    )


@pytest.mark.parametrize("framerate_key", ["mocap_framerate", "mocap_frame_rate"])
def test_load_amass_data(tmp_path, monkeypatch, framerate_key):
    amass_dir = tmp_path / "amass"
    (amass_dir / "SubjA").mkdir(parents=True)
    npz = amass_dir / "SubjA" / "clip.npz"
    _write_amass_npz(npz, framerate_key=framerate_key)
    monkeypatch.setattr(retargeting, "get_amass_dataset_path", lambda: str(amass_dir))

    out = retargeting.load_amass_data("SubjA/clip")
    assert set(out.keys()) == {"pose_aa", "gender", "trans", "betas", "fps"}
    assert out["pose_aa"].shape == (5, 72)  # 66 kept + 6 zero-padded
    assert out["fps"] == 30.0


def test_load_amass_data_missing_framerate_raises(tmp_path, monkeypatch):
    amass_dir = tmp_path / "amass"
    amass_dir.mkdir()
    npz = amass_dir / "clip.npz"
    np.savez(npz, poses=np.zeros((3, 72)), trans=np.zeros((3, 3)),
             betas=np.zeros(16), gender="neutral")  # no framerate key
    monkeypatch.setattr(retargeting, "get_amass_dataset_path", lambda: str(amass_dir))
    with pytest.raises(ValueError, match="Framerate"):
        retargeting.load_amass_data("clip")


# --------------------------- load_robot_conf_file ---------------------------

def test_load_robot_conf_file_real_conf(monkeypatch):
    # no custom conf dirs -> only the in-repo default confs are searched
    monkeypatch.setattr(loco_mujoco, "get_variable", lambda *_a, **_k: None)
    conf = retargeting.load_robot_conf_file("UnitreeG1")
    assert conf is not None


def test_load_robot_conf_file_strips_mjx_prefix(monkeypatch):
    monkeypatch.setattr(loco_mujoco, "get_variable", lambda *_a, **_k: None)
    # "MjxUnitreeG1" must resolve to the same UnitreeG1.yaml conf
    conf_mjx = retargeting.load_robot_conf_file("MjxUnitreeG1")
    conf_plain = retargeting.load_robot_conf_file("UnitreeG1")
    assert conf_mjx == conf_plain


def test_load_robot_conf_file_missing_raises(monkeypatch):
    monkeypatch.setattr(loco_mujoco, "get_variable", lambda *_a, **_k: None)
    with pytest.raises(FileNotFoundError):
        retargeting.load_robot_conf_file("NoSuchRobot12345")


# --------------------------- Tier B: fit_smpl_shape integration ---------------------------
#
# This drives the *real* torch retargeting pipeline (``fit_smpl_shape``) against
# a real robot environment (UnitreeG1) and a *synthetic* SMPL-H model. It needs
# torch + smplx, so it is skipped when the optional SMPL stack is absent. The
# shape-optimization loop is capped at 2 iterations so it stays fast; numerical
# realism of the fitted shape is intentionally not asserted (the real SMPL model
# would be needed for that) -- the point is that the whole env-build /
# mocap-body / T-pose / joint-transform / Adam-optimize / save path executes.

_SMPLH_VERTEX_KEYS = ["nose", "reye", "leye", "rear", "lear",
                      "rthumb", "rindex", "rmiddle", "rring", "rpinky",
                      "lthumb", "lindex", "lmiddle", "lring", "lpinky",
                      "LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"]


def _tiny_vertex_ids():
    return {name: i for i, name in enumerate(_SMPLH_VERTEX_KEYS)}


def _write_synthetic_smplh(path, n_verts=64, n_joints=52, n_betas=16, n_pca=45, seed=1):
    rng = np.random.RandomState(seed)

    def regressor():
        r = rng.rand(n_joints, n_verts)
        return (r / r.sum(axis=1, keepdims=True)).astype(np.float64)

    def weights():
        w = rng.rand(n_verts, n_joints)
        return (w / w.sum(axis=1, keepdims=True)).astype(np.float64)

    def kintree():
        parents = np.zeros(n_joints, dtype=np.int64)
        for i in range(1, n_joints):
            parents[i] = rng.randint(0, i)
        kt = np.zeros((2, n_joints), dtype=np.int64)
        kt[0] = parents
        kt[1] = np.arange(n_joints)
        return kt

    model = {
        "v_template": (rng.randn(n_verts, 3) * 0.1).astype(np.float64),
        # 16 shape coefficients -> matches the (1, 16) betas used by fit_smpl_shape
        "shapedirs": (rng.randn(n_verts, 3, n_betas) * 0.01).astype(np.float64),
        "posedirs": (rng.randn(n_verts, 3, 9 * (n_joints - 1)) * 1e-3).astype(np.float64),
        "J_regressor": regressor(),
        "kintree_table": kintree(),
        "weights": weights(),
        "f": rng.randint(0, n_verts, size=(n_verts, 3)).astype(np.int64),
        "hands_componentsl": (rng.randn(n_pca, 45) * 0.01).astype(np.float64),
        "hands_componentsr": (rng.randn(n_pca, 45) * 0.01).astype(np.float64),
        "hands_meanl": np.zeros(45, dtype=np.float64),
        "hands_meanr": np.zeros(45, dtype=np.float64),
    }
    path.write_bytes(pickle.dumps(model))
    return str(path)


def test_fit_smpl_shape_runs_end_to_end(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")  # noqa: F841
    pytest.importorskip("smplx")
    joblib = pytest.importorskip("joblib")
    from omegaconf import OmegaConf
    import logging

    # Build the real robot conf but cap the optimization to keep it fast.
    monkeypatch.setattr(loco_mujoco, "get_variable", lambda *_a, **_k: None)
    conf = retargeting.load_robot_conf_file("UnitreeG1")
    OmegaConf.set_struct(conf, False)
    conf.optimization_params.shape_iterations = 2
    conf.optimization_params.torch_device = "cpu"

    # The real parser expects a 6890-vertex mesh with 16 betas; inject a tiny
    # vertex_ids map + num_betas so the synthetic model is accepted.
    from loco_mujoco.smpl.parser import SMPLH_Parser as _RealSMPLH

    def _patched_smplh(*args, **kwargs):
        kwargs.setdefault("vertex_ids", _tiny_vertex_ids())
        kwargs.setdefault("num_betas", 16)
        return _RealSMPLH(*args, **kwargs)

    monkeypatch.setattr(retargeting, "SMPLH_Parser", _patched_smplh)

    model_path = _write_synthetic_smplh(tmp_path / "SMPLH_NEUTRAL.pkl")
    save_path = str(tmp_path / "shape" / "out.pkl")

    retargeting.fit_smpl_shape(
        "MjxUnitreeG1", conf, model_path, save_path, logging.getLogger("test"),
        visualize=False,
    )

    assert os.path.isfile(save_path)
    shape_new, scale, smpl2robot_pos, smpl2robot_rot_mat, offset_z, height_scale = \
        joblib.load(save_path)
    assert tuple(shape_new.shape) == (1, 16)
    assert tuple(scale.shape) == (1,)
    # one (pos, rot) offset per mimic site
    n_sites = smpl2robot_rot_mat.shape[0]
    assert smpl2robot_pos.shape == (n_sites, 3)
    assert smpl2robot_rot_mat.shape == (n_sites, 3, 3)


def _build_capped_conf(monkeypatch):
    """Real UnitreeG1 conf with every optimization loop capped to a couple of
    iterations so the torch/mujoco pipeline runs fast under CI."""
    from omegaconf import OmegaConf
    monkeypatch.setattr(loco_mujoco, "get_variable", lambda *_a, **_k: None)
    conf = retargeting.load_robot_conf_file("UnitreeG1")
    OmegaConf.set_struct(conf, False)
    conf.optimization_params.shape_iterations = 2
    conf.optimization_params.motion_iterations = 1
    conf.optimization_params.init_motion_iterations = 1
    conf.optimization_params.pose_iterations = 2  # robot2robot pose opt
    conf.optimization_params.torch_device = "cpu"
    return conf


def test_fit_smpl_motion_runs_end_to_end(tmp_path, monkeypatch):
    """Drive the full motion-retargeting pipeline: optimize a (synthetic) shape,
    then fit a short synthetic AMASS-style motion to the UnitreeG1 robot. Numerical
    realism is not asserted -- the point is that the shape-load / joint-transform /
    mocap-drive / mj_step / qpos-qvel-assembly path executes and yields a valid
    Trajectory of the expected shape."""
    torch = pytest.importorskip("torch")  # noqa: F841
    pytest.importorskip("smplx")
    pytest.importorskip("joblib")
    import logging
    from loco_mujoco.core.trajectory import Trajectory
    from loco_mujoco.smpl.parser import SMPLH_Parser as _RealSMPLH

    conf = _build_capped_conf(monkeypatch)

    def _patched_smplh(*args, **kwargs):
        kwargs.setdefault("vertex_ids", _tiny_vertex_ids())
        kwargs.setdefault("num_betas", 16)
        return _RealSMPLH(*args, **kwargs)

    monkeypatch.setattr(retargeting, "SMPLH_Parser", _patched_smplh)

    model_path = _write_synthetic_smplh(tmp_path / "SMPLH_NEUTRAL.pkl")
    shape_path = str(tmp_path / "shape" / "out.pkl")
    logger = logging.getLogger("test")

    # 1) optimize the shape -> writes the shape file fit_smpl_motion consumes
    retargeting.fit_smpl_shape(
        "MjxUnitreeG1", conf, model_path, shape_path, logger, visualize=False,
    )

    # 2) synthetic AMASS-style motion (T frames): zero pose, gently drifting root
    T = 6
    trans = np.zeros((T, 3), dtype=np.float64)
    trans[:, 0] = np.linspace(0.0, 0.05, T)  # small forward drift
    motion_data = {
        "pose_aa": np.zeros((T, 72), dtype=np.float64),
        "trans": trans,
        "fps": 30,
    }

    traj = retargeting.fit_smpl_motion(
        "UnitreeG1", conf, model_path, motion_data, shape_path, logger,
        skip_steps=False, visualize=False,
    )

    assert isinstance(traj, Trajectory)
    # velocity uses a centered difference -> traj length is T-2
    qpos = np.asarray(traj.data.qpos)
    qvel = np.asarray(traj.data.qvel)
    assert qpos.shape[0] == T - 2
    assert qvel.shape[0] == T - 2
    assert qpos.shape[1] > 0 and qvel.shape[1] > 0
    assert np.all(np.isfinite(qpos)) and np.all(np.isfinite(qvel))
    assert traj.info.frequency == 30


def test_extend_motion_runs_end_to_end(tmp_path, monkeypatch):
    """`extend_motion` takes a retargeted (qpos/qvel-only) trajectory, interpolates
    it to the environment frequency and replays it under an ExtendTrajData callback
    (render=False) to add model-specific entities (body xpos, site pos, ...). Feed it
    the output of the fit_smpl pipeline and check the extended trajectory is longer
    (resampled to env.dt) and carries the added fields."""
    torch = pytest.importorskip("torch")  # noqa: F841
    pytest.importorskip("smplx")
    pytest.importorskip("joblib")
    import logging
    from loco_mujoco.core.trajectory import Trajectory
    from loco_mujoco.smpl.parser import SMPLH_Parser as _RealSMPLH

    conf = _build_capped_conf(monkeypatch)

    def _patched_smplh(*args, **kwargs):
        kwargs.setdefault("vertex_ids", _tiny_vertex_ids())
        kwargs.setdefault("num_betas", 16)
        return _RealSMPLH(*args, **kwargs)

    monkeypatch.setattr(retargeting, "SMPLH_Parser", _patched_smplh)

    model_path = _write_synthetic_smplh(tmp_path / "SMPLH_NEUTRAL.pkl")
    shape_path = str(tmp_path / "shape" / "out.pkl")
    logger = logging.getLogger("test")

    retargeting.fit_smpl_shape(
        "MjxUnitreeG1", conf, model_path, shape_path, logger, visualize=False,
    )

    T = 8
    trans = np.zeros((T, 3), dtype=np.float64)
    trans[:, 0] = np.linspace(0.0, 0.05, T)
    motion_data = {
        "pose_aa": np.zeros((T, 72), dtype=np.float64),
        "trans": trans,
        "fps": 30,
    }
    traj = retargeting.fit_smpl_motion(
        "UnitreeG1", conf, model_path, motion_data, shape_path, logger,
        skip_steps=False, visualize=False,
    )

    n_before = np.asarray(traj.data.qpos).shape[0]
    extended = retargeting.extend_motion("UnitreeG1", conf.env_params, traj, logger)

    assert isinstance(extended, Trajectory)
    # resampled from 30 Hz to the (higher) env control frequency -> more samples
    assert np.asarray(extended.data.qpos).shape[0] > n_before
    # ExtendTrajData populates model-specific fields absent from the raw traj
    assert extended.data.xpos is not None
    assert np.asarray(extended.data.xpos).shape[0] == np.asarray(extended.data.qpos).shape[0]


def test_motion_transfer_robot_to_robot_load_branch(tmp_path, monkeypatch):
    """motion_transfer_robot_to_robot's source-motion optimization hardcodes
    torch.device('cuda') (613-771) and cannot run on CPU CI. Its else-branch --
    load a precomputed fitted motion and retarget it onto the *target* robot,
    fitting the target shape on the way -- is CPU-friendly. Exercise that path."""
    torch = pytest.importorskip("torch")  # noqa: F841
    pytest.importorskip("smplx")
    pytest.importorskip("joblib")
    import logging
    from loco_mujoco.core.trajectory import Trajectory
    from loco_mujoco.smpl.parser import SMPLH_Parser as _RealSMPLH

    conf = _build_capped_conf(monkeypatch)

    def _patched_smplh(*args, **kwargs):
        kwargs.setdefault("vertex_ids", _tiny_vertex_ids())
        kwargs.setdefault("num_betas", 16)
        return _RealSMPLH(*args, **kwargs)

    monkeypatch.setattr(retargeting, "SMPLH_Parser", _patched_smplh)

    model_path = _write_synthetic_smplh(tmp_path / "SMPLH_NEUTRAL.pkl")
    logger = logging.getLogger("test")

    # precomputed fitted source motion on disk -> takes the load (else) branch,
    # skipping the cuda optimization entirely.
    T = 6
    trans = np.zeros((T, 3), dtype=np.float64)
    trans[:, 0] = np.linspace(0.0, 0.05, T)
    fitted_motion = tmp_path / "fitted" / "motion.npz"
    fitted_motion.parent.mkdir(parents=True)
    np.savez(fitted_motion, pose_aa=np.zeros((T, 72), dtype=np.float64),
             trans=trans, fps=30)

    # target data dir without a shape file -> fit_smpl_shape(target) is triggered
    target_dir = tmp_path / "target"
    target_dir.mkdir()

    traj = retargeting.motion_transfer_robot_to_robot(
        env_name_source="UnitreeG1", robot_conf_source=conf, traj_source=None,
        path_source_robot_smpl_data=str(tmp_path / "source"),
        env_name_target="UnitreeG1", robot_conf_target=conf,
        path_target_robot_smpl_data=str(target_dir),
        path_to_smpl_model=model_path, logger=logger,
        path_to_fitted_motion_source=str(fitted_motion), visualize=False,
    )

    assert isinstance(traj, Trajectory)
    assert np.asarray(traj.data.qpos).shape[0] == T - 2
    # the target shape was fitted and cached on the way
    assert (target_dir / retargeting.OPTIMIZED_SHAPE_FILE_NAME).is_file()


def test_motion_transfer_robot_to_robot_optimize_branch(tmp_path, monkeypatch):
    """With no precomputed fitted-motion file, motion_transfer_robot_to_robot runs
    its source-motion optimization (extend source traj -> fit source shape ->
    Adam-optimize an SMPL pose to match the source site poses -> Gaussian-smooth)
    and then retargets onto the target robot. This path used to hardcode
    torch.device('cuda'); it now respects optimization_params.torch_device, so it
    runs on CPU. All loops are capped to 1-2 iterations."""
    torch = pytest.importorskip("torch")  # noqa: F841
    pytest.importorskip("smplx")
    pytest.importorskip("joblib")
    import logging
    from loco_mujoco.core.trajectory import Trajectory
    from loco_mujoco.smpl.parser import SMPLH_Parser as _RealSMPLH

    conf = _build_capped_conf(monkeypatch)

    def _patched_smplh(*args, **kwargs):
        kwargs.setdefault("vertex_ids", _tiny_vertex_ids())
        kwargs.setdefault("num_betas", 16)
        return _RealSMPLH(*args, **kwargs)

    monkeypatch.setattr(retargeting, "SMPLH_Parser", _patched_smplh)

    model_path = _write_synthetic_smplh(tmp_path / "SMPLH_NEUTRAL.pkl")
    logger = logging.getLogger("test")

    # build a source trajectory to transfer (raw qpos/qvel from the fit pipeline)
    source_shape = str(tmp_path / "src_shape" / "out.pkl")
    retargeting.fit_smpl_shape(
        "MjxUnitreeG1", conf, model_path, source_shape, logger, visualize=False,
    )
    T = 6
    trans = np.zeros((T, 3), dtype=np.float64)
    trans[:, 0] = np.linspace(0.0, 0.05, T)
    src_motion = {"pose_aa": np.zeros((T, 72), dtype=np.float64),
                  "trans": trans, "fps": 30}
    traj_source = retargeting.fit_smpl_motion(
        "UnitreeG1", conf, model_path, src_motion, source_shape, logger,
        skip_steps=False, visualize=False,
    )

    # empty data dirs -> source & target shapes are fitted inside the call.
    # path_to_fitted_motion_source=None -> the optimization (if) branch runs and
    # writes the fitted motion out.
    source_dir = tmp_path / "source"; source_dir.mkdir()
    target_dir = tmp_path / "target"; target_dir.mkdir()
    fitted_out = tmp_path / "fitted" / "src_motion.npz"

    traj = retargeting.motion_transfer_robot_to_robot(
        env_name_source="UnitreeG1", robot_conf_source=conf, traj_source=traj_source,
        path_source_robot_smpl_data=str(source_dir),
        env_name_target="UnitreeG1", robot_conf_target=conf,
        path_target_robot_smpl_data=str(target_dir),
        path_to_smpl_model=model_path, logger=logger,
        path_to_fitted_motion_source=str(fitted_out), visualize=False,
    )

    assert isinstance(traj, Trajectory)
    assert np.asarray(traj.data.qpos).shape[0] > 0
    assert np.all(np.isfinite(np.asarray(traj.data.qpos)))
    # the optimization branch cached the fitted source motion + the target shape
    assert fitted_out.is_file()
    assert (target_dir / retargeting.OPTIMIZED_SHAPE_FILE_NAME).is_file()
