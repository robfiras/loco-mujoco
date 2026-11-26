"""
Experimental learning algorithms.

These algorithms share the interface of
:class:`~loco_mujoco.algorithms.common.base_algorithm.JaxRLAlgorithmBase` but are less thoroughly tested than
the core algorithms in :mod:`loco_mujoco.algorithms`:

- :class:`~loco_mujoco.algorithms.experimental.s2pg_ppo_jax.S2PGPPOJax`: PPO with stochastic stateful policies.
- :class:`~loco_mujoco.algorithms.experimental.bptt_ppo_jax.BPTTPPOJax`: PPO with backpropagation through time.
- :class:`~loco_mujoco.algorithms.experimental.history_ppo_jax.HistoryPPOJax`: PPO with an observation history.
- :class:`~loco_mujoco.algorithms.experimental.sac_jax.SACJax`: Soft Actor-Critic.
- :class:`~loco_mujoco.algorithms.experimental.td3_jax.TD3Jax`: Twin Delayed DDPG.
- :class:`~loco_mujoco.algorithms.experimental.vanilla_dagger_jax.VanillaDaggerJax`: DAgger teacher-student
  distillation.

:class:`~loco_mujoco.algorithms.experimental.offpolicy_base.OffPolicyBase` holds the machinery shared by SAC
and TD3.

"""

from .s2pg_ppo_jax import S2PGPPOJax
from .bptt_ppo_jax import BPTTPPOJax
from .history_ppo_jax import HistoryPPOJax
from .offpolicy_base import OffPolicyBase, OffPolicyCriticNet, ReplayBuffer
from .sac_jax import SACJax
from .td3_jax import TD3Jax
from .vanilla_dagger_jax import VanillaDaggerJax
