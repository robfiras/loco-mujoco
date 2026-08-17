"""Unit tests for the pure helpers in algorithms/experimental/offpolicy_base.

`_normalize_dense_kernels` (XQC-style kernel normalization) and
`_c51_project_target` (distributional-critic support projection) are only
reached through specific off-policy config branches that the default SAC/TD3
training tests don't hit. Testing them directly pins their invariants without
running a full training loop.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from loco_mujoco.algorithms.experimental.offpolicy_base import (
    _normalize_dense_kernels,
    _c51_project_target,
)

jax.config.update("jax_platform_name", "cpu")

_ATOL = 1e-5


def _augmented_col_norms(kernel, bias):
    """||[kernel; bias]||_axis=-2 -- the quantity the normalizer sets to 1."""
    w = np.concatenate([np.asarray(kernel), np.asarray(bias)[None, :]], axis=0)
    return np.linalg.norm(w, axis=0)


def test_normalize_dense_kernels_unit_norm():
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    params = {
        "q1": {
            "Dense_0": {"kernel": jax.random.normal(k1, (4, 3)),
                        "bias": jax.random.normal(k2, (3,))},
        },
    }
    out = _normalize_dense_kernels(params, normalize_last_layer=True)
    dense = out["q1"]["Dense_0"]
    norms = _augmented_col_norms(dense["kernel"], dense["bias"])
    np.testing.assert_allclose(norms, np.ones_like(norms), atol=_ATOL)


def test_normalize_dense_kernels_skips_last_layer():
    key = jax.random.PRNGKey(1)
    ks = jax.random.split(key, 4)
    params = {
        "q1": {
            "Dense_0": {"kernel": jax.random.normal(ks[0], (4, 3)),
                        "bias": jax.random.normal(ks[1], (3,))},
            "Dense_1": {"kernel": jax.random.normal(ks[2], (3, 2)),
                        "bias": jax.random.normal(ks[3], (2,))},
        },
    }
    out = _normalize_dense_kernels(params, normalize_last_layer=False)

    # Dense_0 (hidden) is normalized to unit augmented-column norm.
    d0 = out["q1"]["Dense_0"]
    np.testing.assert_allclose(
        _augmented_col_norms(d0["kernel"], d0["bias"]),
        np.ones(3), atol=_ATOL,
    )
    # Dense_1 (predictor head) is left untouched.
    d1 = out["q1"]["Dense_1"]
    np.testing.assert_allclose(
        np.asarray(d1["kernel"]),
        np.asarray(params["q1"]["Dense_1"]["kernel"]), atol=_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(d1["bias"]),
        np.asarray(params["q1"]["Dense_1"]["bias"]), atol=_ATOL,
    )


def test_normalize_dense_kernels_no_bias():
    # A Dense path with only a kernel (no bias) normalizes the kernel columns.
    params = {"Dense_0": {"kernel": jnp.array([[3.0, 0.0], [4.0, 0.0]])}}
    out = _normalize_dense_kernels(params, normalize_last_layer=True)
    kernel = np.asarray(out["Dense_0"]["kernel"])
    # first column had norm 5 -> normalized to unit; degenerate second column
    # (norm 0) is guarded by the +1e-12 epsilon and stays finite.
    assert np.linalg.norm(kernel[:, 0]) == pytest.approx(1.0, abs=_ATOL)
    assert np.all(np.isfinite(kernel))


def test_c51_project_target_identity_on_support():
    """Projecting onto the exact bin centers is the identity: the target
    distribution equals the (renormalized) input probabilities."""
    num_atoms, min_v, max_v = 5, -1.0, 1.0
    support = jnp.linspace(min_v, max_v, num_atoms)  # bin centers

    logits = jnp.array([[0.1, 0.5, -0.3, 0.2, 0.0],
                        [1.0, -1.0, 0.0, 0.4, -0.2]])
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    bin_values = jnp.broadcast_to(support, log_probs.shape)

    target = _c51_project_target(log_probs, bin_values, num_atoms, min_v, max_v)
    target = np.asarray(target)

    # identity projection: mass lands back on the same bins
    np.testing.assert_allclose(target, np.asarray(jnp.exp(log_probs)), atol=_ATOL)
    # a valid probability distribution per row
    np.testing.assert_allclose(target.sum(axis=-1), np.ones(2), atol=_ATOL)


def test_c51_project_target_conserves_mass_when_shifted():
    """A shift that stays within [min_v, max_v] splits mass across bins but
    conserves total probability per row."""
    num_atoms, min_v, max_v = 5, -1.0, 1.0
    support = jnp.linspace(min_v, max_v, num_atoms)

    log_probs = jax.nn.log_softmax(jnp.array([[0.2, 0.1, 0.0, -0.1, 0.3]]), axis=-1)
    # shift by +0.15 (less than one bin width of 0.5), still inside the support
    bin_values = jnp.clip(support + 0.15, min_v, max_v)[None, :]

    target = np.asarray(
        _c51_project_target(log_probs, bin_values, num_atoms, min_v, max_v)
    )
    assert target.shape == (1, num_atoms)
    np.testing.assert_allclose(target.sum(axis=-1), np.ones(1), atol=_ATOL)
    assert np.all(target >= -_ATOL)
