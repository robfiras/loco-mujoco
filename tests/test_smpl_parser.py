"""Tests for loco_mujoco.smpl.parser using *synthetic* SMPL/SMPL-H models.

The real SMPL/SMPL-H model files are license-restricted and cannot be shipped
with the repo, so historically this module was untested. These tests show the
parser code paths (blend-shape / joint-regression / rigid-transform math) can
be exercised end-to-end against a tiny procedurally-generated model that shares
the real model's *structure* (not its data). This validates shapes, code paths,
and error handling without any external model download.

Numerical fidelity vs. the real SMPL output is intentionally NOT asserted here
(that requires the real model); those checks belong in an optional, model-gated
job. See tests marked with ``requires_real_smpl`` elsewhere if/when added.
"""
import pickle

import numpy as np
import pytest

# Skip the whole module if the optional SMPL stack (torch + smplx) is absent.
torch = pytest.importorskip("torch")
pytest.importorskip("smplx")
_parser = pytest.importorskip("loco_mujoco.smpl.parser")
SMPL_Parser = getattr(_parser, "SMPL_Parser", None)
SMPLH_Parser = getattr(_parser, "SMPLH_Parser", None)
if SMPL_Parser is None or SMPLH_Parser is None:  # pragma: no cover
    pytest.skip("SMPL parsers unavailable", allow_module_level=True)


# VertexJointSelector in smplx requires these 21 named vertices; we remap them
# to small in-range indices so a tiny mesh suffices instead of a 6890-vert one.
_VERTEX_KEYS = ["nose", "reye", "leye", "rear", "lear",
                "rthumb", "rindex", "rmiddle", "rring", "rpinky",
                "lthumb", "lindex", "lmiddle", "lring", "lpinky",
                "LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"]


def _tiny_vertex_ids():
    return {name: i for i, name in enumerate(_VERTEX_KEYS)}


def _rand_regressor(rng, n_joints, n_verts):
    r = rng.rand(n_joints, n_verts)
    return (r / r.sum(axis=1, keepdims=True)).astype(np.float64)


def _rand_weights(rng, n_verts, n_joints):
    w = rng.rand(n_verts, n_joints)
    return (w / w.sum(axis=1, keepdims=True)).astype(np.float64)


def _kintree(rng, n_joints):
    parents = np.zeros(n_joints, dtype=np.int64)
    for i in range(1, n_joints):
        parents[i] = rng.randint(0, i)
    kt = np.zeros((2, n_joints), dtype=np.int64)
    kt[0] = parents
    kt[1] = np.arange(n_joints)
    return kt


def _write_synthetic_smpl(path, n_verts=64, n_joints=24, n_betas=10, seed=0):
    rng = np.random.RandomState(seed)
    model = {
        "v_template": (rng.randn(n_verts, 3) * 0.1).astype(np.float64),
        "shapedirs": (rng.randn(n_verts, 3, n_betas) * 0.01).astype(np.float64),
        "posedirs": (rng.randn(n_verts, 3, 9 * (n_joints - 1)) * 1e-3).astype(np.float64),
        "J_regressor": _rand_regressor(rng, n_joints, n_verts),
        "kintree_table": _kintree(rng, n_joints),
        "weights": _rand_weights(rng, n_verts, n_joints),
        "f": rng.randint(0, n_verts, size=(n_verts, 3)).astype(np.int64),
    }
    path.write_bytes(pickle.dumps(model))
    return str(path)


def _write_synthetic_smplh(path, n_verts=64, n_joints=52, n_betas=10, n_pca=45, seed=1):
    rng = np.random.RandomState(seed)
    model = {
        "v_template": (rng.randn(n_verts, 3) * 0.1).astype(np.float64),
        "shapedirs": (rng.randn(n_verts, 3, n_betas) * 0.01).astype(np.float64),
        "posedirs": (rng.randn(n_verts, 3, 9 * (n_joints - 1)) * 1e-3).astype(np.float64),
        "J_regressor": _rand_regressor(rng, n_joints, n_verts),
        "kintree_table": _kintree(rng, n_joints),
        "weights": _rand_weights(rng, n_verts, n_joints),
        "f": rng.randint(0, n_verts, size=(n_verts, 3)).astype(np.int64),
        "hands_componentsl": (rng.randn(n_pca, 45) * 0.01).astype(np.float64),
        "hands_componentsr": (rng.randn(n_pca, 45) * 0.01).astype(np.float64),
        "hands_meanl": np.zeros(45, dtype=np.float64),
        "hands_meanr": np.zeros(45, dtype=np.float64),
    }
    path.write_bytes(pickle.dumps(model))
    return str(path)


@pytest.fixture
def smpl_parser(tmp_path):
    pkl = _write_synthetic_smpl(tmp_path / "SMPL_NEUTRAL.pkl")
    return SMPL_Parser(model_path=pkl, gender="neutral",
                       vertex_ids=_tiny_vertex_ids())


@pytest.fixture
def smplh_parser(tmp_path):
    pkl = _write_synthetic_smplh(tmp_path / "SMPLH_NEUTRAL.pkl")
    return SMPLH_Parser(model_path=pkl, gender="neutral",
                        vertex_ids=_tiny_vertex_ids())


# --------------------------- SMPL_Parser ---------------------------

def test_smpl_construct(smpl_parser):
    assert len(smpl_parser.joint_names) == 24
    # elbow/shoulder ranges are widened x4 in the parser
    assert np.all(smpl_parser.joint_range["L_Elbow"] == smpl_parser.joint_range["R_Elbow"])


def test_smpl_get_joints_verts(smpl_parser):
    pose = torch.zeros(1, 72)
    verts, joints = smpl_parser.get_joints_verts(pose, th_betas=torch.zeros(1, 10))
    assert verts.shape == (1, 64, 3)
    assert joints.shape == (1, 24, 3)


def test_smpl_get_joints_verts_reshapes_pose(smpl_parser):
    # a 2-D pose whose dim-1 != 72 is reshaped to (-1, 72) internally
    verts, joints = smpl_parser.get_joints_verts(torch.zeros(2, 36))
    assert joints.shape == (1, 24, 3)


def test_smpl_get_offsets(smpl_parser):
    verts, jts, skin_w, names, offsets, parents, channels, jrange = smpl_parser.get_offsets()
    assert len(names) == 24
    assert set(offsets.keys()) == set(names)
    assert skin_w.shape == (64, 24)


def test_smpl_get_mesh_offsets(smpl_parser):
    out = smpl_parser.get_mesh_offsets(flatfoot=True)
    assert len(out) == 11  # verts, joints, weights, names, offsets, parents, axes, dofs, range, contype, conaffinity


# --------------------------- SMPLH_Parser ---------------------------

def test_smplh_construct(smplh_parser):
    assert len(smplh_parser.joint_names) == len(smplh_parser.parents)


def test_smplh_get_joints_verts(smplh_parser):
    pose = torch.zeros(1, 156)
    verts, joints = smplh_parser.get_joints_verts(pose, th_betas=torch.zeros(1, 10))
    assert verts.shape[-1] == 3
    assert joints.shape[0] == 1


def test_smplh_get_joint_transformations(smplh_parser):
    """Exercises _transforms + batch_rigid_transform_global (the retarget math)."""
    T = smplh_parser.get_joint_transformations(torch.zeros(1, 156),
                                               th_betas=torch.zeros(1, 10))
    assert T.shape == (1, 52, 4, 4)
    # bottom row of every homogeneous transform must be [0,0,0,1]
    bottom = T[0, :, 3, :].detach().cpu().numpy()
    expected = np.tile(np.array([0, 0, 0, 1.0]), (52, 1))
    np.testing.assert_allclose(bottom, expected, atol=1e-5)


def test_smplh_get_joint_transformations_with_translation(smplh_parser):
    trans = torch.ones(1, 3)
    T = smplh_parser.get_joint_transformations(torch.zeros(1, 156),
                                               th_betas=torch.zeros(1, 10),
                                               th_trans=trans)
    # translation is added into the last column of each transform
    assert T.shape == (1, 52, 4, 4)


def test_smplh_get_offsets_and_mesh(smplh_parser):
    off = smplh_parser.get_offsets()
    assert len(off) == 8
    mesh = smplh_parser.get_mesh_offsets()
    assert len(mesh) == 11
