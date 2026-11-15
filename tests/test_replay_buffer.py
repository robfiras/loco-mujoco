"""Tests for the circular ReplayBuffer used by the off-policy and imitation algorithms."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from loco_mujoco.algorithms.common.dataclasses import ReplayBuffer

jax.config.update("jax_platform_name", "cpu")

_OBS_DIM, _ACTION_DIM, _CAPACITY = 3, 2, 4


def _batch(start, n):
    """n transitions whose every column encodes the transition index start..start+n-1."""
    idx = jnp.arange(start, start + n, dtype=jnp.float32)
    return dict(obs=jnp.tile(idx[:, None], (1, _OBS_DIM)),
                next_obs=jnp.tile(idx[:, None] + 0.5, (1, _OBS_DIM)),
                action=jnp.tile(-idx[:, None], (1, _ACTION_DIM)),
                reward=10.0 * idx,
                done=(idx % 2 == 0))


def test_add_batch_fills_and_wraps_around():
    buffer = ReplayBuffer.create(_OBS_DIM, _ACTION_DIM, _CAPACITY)
    buffer = buffer.add_batch(**_batch(0, 3))
    assert (int(buffer.ptr), int(buffer.size)) == (3, 3)

    buffer = buffer.add_batch(**_batch(3, 3))   # transitions 4 and 5 overwrite slots 0 and 1
    assert (int(buffer.ptr), int(buffer.size)) == (2, _CAPACITY)
    assert np.allclose(buffer.obs[:, 0], [4.0, 5.0, 2.0, 3.0])
    assert np.allclose(buffer.next_obs[:, 0], [4.5, 5.5, 2.5, 3.5])
    assert np.allclose(buffer.action[:, 0], [-4.0, -5.0, -2.0, -3.0])
    assert np.allclose(buffer.reward, [40.0, 50.0, 20.0, 30.0])
    assert np.allclose(buffer.done, [1.0, 0.0, 1.0, 0.0])


@pytest.mark.parametrize("add_subsample", (False, True))
def test_optional_columns(add_subsample):
    buffer = ReplayBuffer.create(_OBS_DIM, _ACTION_DIM, _CAPACITY, store_next_obs=False,
                                 store_value_target=True, add_subsample=add_subsample)
    assert buffer.next_obs.size == 0

    rng = jax.random.PRNGKey(0)
    buffer = buffer.add_batch(**_batch(0, 2), value_target=jnp.array([7.0, 8.0]), rng=rng)
    assert buffer.next_obs.size == 0
    assert np.allclose(buffer.value_target[:2], [7.0, 8.0])

    with pytest.raises(AssertionError, match="value_target=None"):
        buffer.add_batch(**_batch(2, 2), rng=rng)


def test_subsampling_with_probability_one_keeps_everything():
    plain = ReplayBuffer.create(_OBS_DIM, _ACTION_DIM, _CAPACITY)
    sub = ReplayBuffer.create(_OBS_DIM, _ACTION_DIM, _CAPACITY, add_subsample=True, add_prob=1.0)
    for start in (0, 3):
        plain = plain.add_batch(**_batch(start, 3))
        sub = sub.add_batch(**_batch(start, 3), rng=jax.random.PRNGKey(start))

    for column in ("obs", "next_obs", "action", "reward", "done"):
        assert np.allclose(getattr(sub, column), getattr(plain, column)), column
    assert (int(sub.ptr), int(sub.size)) == (int(plain.ptr), int(plain.size))


def test_subsampling_with_probability_zero_keeps_nothing():
    buffer = ReplayBuffer.create(_OBS_DIM, _ACTION_DIM, _CAPACITY, add_subsample=True, add_prob=0.0)
    buffer = buffer.add_batch(**_batch(1, 3), rng=jax.random.PRNGKey(0))
    assert (int(buffer.ptr), int(buffer.size)) == (0, 0)
    for column in ("obs", "next_obs", "action", "reward", "done"):
        assert not np.any(np.asarray(getattr(buffer, column))), column


def test_subsampling_keeps_a_subset_in_order():
    capacity = 64
    buffer = ReplayBuffer.create(_OBS_DIM, _ACTION_DIM, capacity, add_subsample=True, add_prob=0.5)
    buffer = buffer.add_batch(**_batch(1, 32), rng=jax.random.PRNGKey(3))

    size = int(buffer.size)
    assert 0 < size < 32
    assert int(buffer.ptr) == size
    kept = np.asarray(buffer.obs[:size, 0])
    assert np.all(np.diff(kept) > 0)             # original order, no duplicates
    assert np.allclose(buffer.reward[:size], 10.0 * kept)
    assert np.allclose(buffer.next_obs[:size, 0], kept + 0.5)
    assert not np.any(np.asarray(buffer.obs[size:]))


def test_subsampling_requires_a_random_key():
    buffer = ReplayBuffer.create(_OBS_DIM, _ACTION_DIM, _CAPACITY, add_subsample=True, add_prob=0.5)
    with pytest.raises(AssertionError, match="rng=None"):
        buffer.add_batch(**_batch(0, 2))
