import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import jax.numpy as jnp
import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

import loco_mujoco.utils
from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax
from loco_mujoco.core.wrappers.mjx import SummaryMetrics
from loco_mujoco.utils.metrics import QuantityContainer, ValidationSummary


@pytest.mark.parametrize(
    "site_errors, expected",
    [([2.0, 3.0, 7.0], 12.0), ([2.0, 3.0, None], None), ([None, None, None], None)],
)
def test_mimic_sweep_metric(monkeypatch, tmp_path, site_errors, expected):
    run = Mock()
    monkeypatch.setitem(sys.modules, "wandb", Mock(init=Mock(return_value=run)))
    monkeypatch.setenv("XLA_FLAGS", os.environ.get("XLA_FLAGS", ""))
    monkeypatch.setattr(HydraConfig, "get", lambda: SimpleNamespace(runtime=SimpleNamespace(output_dir=tmp_path)))
    factory = Mock()
    factory.make.return_value = (SimpleNamespace(video_file_path="example.mp4"), None)
    monkeypatch.setattr(TaskFactory, "get_factory_cls", Mock(return_value=factory))
    monkeypatch.setattr(PPOJax, "init_agent_conf", Mock())
    monkeypatch.setattr(PPOJax, "init_agent_state", Mock(return_value=None))
    monkeypatch.setattr(PPOJax, "save_agent", Mock(return_value=tmp_path / "agent"))
    monkeypatch.setattr(PPOJax, "play_policy", Mock())
    monkeypatch.setattr(loco_mujoco.utils, "MetricsHandler", Mock())

    summary = dict(mean_episode_return=jnp.array([5.0]), mean_episode_length=jnp.array([8.0]),
                   max_timestep=jnp.array([17]))
    position, orientation, velocity = (jnp.array([] if value is None else [value]) for value in site_errors)
    validation = ValidationSummary(
        **summary, euclidean_distance=QuantityContainer(site_rpos=position, site_rrotvec=orientation, site_rvel=velocity)
    )
    results = dict(agent_state=None, training_metrics=SummaryMetrics(**summary), validation_metrics=validation)

    def train(*_):
        return results

    monkeypatch.setattr(PPOJax, "build_train_fn", Mock(return_value=train))

    config = OmegaConf.create({"wandb": {"project": "offline"}, "experiment": {
        "task_factory": {"name": "ImitationFactory", "params": {}}, "env_params": {}, "n_seeds": 1,
        "validation": {"active": True}, "validation_interval": 1, "debug": False,
    }})
    path = Path(__file__).resolve().parents[1] / "examples/training_examples/jax_rl_mimic/experiment.py"
    runpy.run_path(str(path))["experiment"](config)

    sweep = [(float(call.args[0]["Metric for Sweep"]), call.kwargs["step"]) for call in run.log.call_args_list
             if "Metric for Sweep" in call.args[0]]
    assert sweep == ([] if expected is None else [(expected, 17)])
    assert any("Validation Info/Mean Episode Return" in call.args[0] for call in run.log.call_args_list)
