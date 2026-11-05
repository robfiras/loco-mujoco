
### Training Examples

Within this directory, you can find different imitation and reinforcement learning training examples:

- `jax_rl`: Reinforcement Learning example.
- `jax_gail`: GAIL (Generative Adversarial Imitation Learning) example.
- `jax_amp`: AMP (Adversarial Motion Prior) example.
- `jax_rl_mimic`: DeepMimic imitation learning example.
- `jax_distill`: Vanilla DAgger distillation example.

The remaining `jax_rl_mimic_*` directories (`bptt`, `history`, `s2pg`, `s2pg4`, `sac`, `td3`) contain
examples of the experimental algorithms in `loco_mujoco/algorithms/experimental`.

We use WandB to log the training process and results. So you have to install this dependency to run the training examples:

```bash
pip install wandb
```

The examples listed above have their own `README.md` file, which provides more detailed instructions on how to
run the training and evaluation process.
