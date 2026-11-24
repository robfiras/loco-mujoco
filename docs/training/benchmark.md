# Benchmark results

This page reports experiments run with LocoMuJoCo 1.2.0. They give reference values for the default tracking
recipe on every humanoid and the human models, compare algorithms on one humanoid, and show SuperMimic on the
two real-robot environments. Use them as baselines for your own methods, and to check that a setup
reproduces the expected behavior.

## Setup

Experiments 1 to 3 use the same 13 clips of the [default dataset](../user_guide/datasets.md): `walk`, `run`,
`squat`, `stepinplace1` to `stepinplace3`, `onestepleft`, `onestepright`, `onestepside1`, `onestepside2`,
`onesteplong`, `onehopforward` and `highknee`. Every pair of embodiment and clip trains its own policy, with
five seeds. All runs of an experiment share one hyperparameter file and the same horizon of 1,000 steps.

Besides the return and the episode length, the results report how closely a policy follows its reference
motion, using the three trajectory-similarity measures of the [evaluation](evaluation.md) module:

- **Euclidean**: the mean distance between rollout and reference at the same time step. It also penalizes a
  motion that is correct but early or late.
- **Dynamic time warping (DTW)**: aligns the two time series first and sums the distances along the best
  alignment, so it ignores timing offsets. Because it is a sum, it grows with the length of the clip.
- **Discrete Fréchet distance**: uses the same kind of alignment but reports the largest distance along it, so
  it measures the worst deviation rather than the average one.

For all three, lower is better.

## Tracking on every embodiment

**Experiment 1** trains [PPO with the DeepMimic-style tracking reward](deepmimic.md) on 11 embodiments, 13 clips
and 5 seeds, 715 runs in total, for about $5 \times 10^8$ environment steps each.

```{figure} ../_static/figures/benchmark_embodiments.png
:alt: Training return, episode length and final tracking error per embodiment
:width: 100%
:align: center

Top: training return and episode length per embodiment, mean ± standard deviation over clips and seeds.
Bottom: final validation error on the joint positions under the three measures, one box per embodiment over
13 clips × 5 seeds (log scale).
```

The final values per embodiment, with the tracking error measured on the positions of the mimic sites
relative to the root (`site_rpos`, in meters), as mean ± standard deviation over the 13 clips and 5 seeds:

| Embodiment | Return | Episode length | Euclidean | DTW | Fréchet |
|---|---:|---:|---:|---:|---:|
| UnitreeH1 | 1535 ± 81 | 925 ± 27 | 0.033 ± 0.007 | 2.84 ± 0.57 | 0.064 ± 0.013 |
| UnitreeH1v2 | 1086 ± 128 | 890 ± 41 | 0.084 ± 0.025 | 7.38 ± 2.16 | 0.162 ± 0.071 |
| UnitreeG1 | 1378 ± 77 | 926 ± 27 | 0.027 ± 0.004 | 2.37 ± 0.31 | 0.053 ± 0.007 |
| Atlas | 725 ± 89 | 923 ± 28 | 0.092 ± 0.021 | 8.06 ± 1.95 | 0.159 ± 0.035 |
| Talos | 1222 ± 91 | 913 ± 32 | 0.057 ± 0.017 | 4.98 ± 1.64 | 0.104 ± 0.034 |
| Apollo | 1068 ± 96 | 961 ± 13 | 0.078 ± 0.025 | 6.75 ± 2.22 | 0.111 ± 0.043 |
| BoosterT1 | 1184 ± 83 | 913 ± 31 | 0.061 ± 0.014 | 5.59 ± 1.47 | 0.111 ± 0.031 |
| FourierGR1T2 | 1647 ± 102 | 926 ± 27 | 0.033 ± 0.009 | 2.84 ± 0.74 | 0.061 ± 0.017 |
| ToddlerBot | 1587 ± 139 | 925 ± 26 | 0.016 ± 0.006 | 1.38 ± 0.51 | 0.029 ± 0.012 |
| SkeletonTorque | 1096 ± 66 | 921 ± 28 | 0.047 ± 0.017 | 4.26 ± 1.59 | 0.127 ± 0.082 |
| SkeletonMuscle | 838 ± 403 | 671 ± 276 | 0.104 ± 0.060 | 9.65 ± 5.76 | 0.242 ± 0.153 |

```{note}
The tracking errors on the site quantities, in this table and in the figure of Experiment 3, were computed
after training with the corrected {class}`~loco_mujoco.utils.metrics.MetricsHandler`: every final policy was
rolled out again, five validation rollouts each, and the results were averaged. Versions of LocoMuJoCo up to
1.2.0 evaluated the site quantities on the wrong axis (see the [changelog](../development/changelog.md)), so
values of `site_rpos`, `site_rrotvec` and `site_rvel` logged with those versions are not comparable with the
ones here. The return, the episode length and the errors on the joint positions are not affected. A few
policies whose checkpoints were no longer available were trained again with the same configuration and seed.
```

What the results show:

- The Unitree G1 and H1, the Fourier GR1T2 and ToddlerBot track most closely. Atlas, the Unitree H1-2, Apollo
  and the muscle-driven SkeletonMuscle have the largest errors.
- SkeletonMuscle is the hardest embodiment. It is the only one whose episodes regularly end early, and its
  results vary most between clips and seeds.
- The three measures rank the embodiments in nearly the same order, on the joint positions as on the site
  quantities. For converged tracking policies, any one of them is a reasonable summary.
- A few seeds fail and end with an error 50 to 500 times that of a typical seed. When you compare methods
  over many runs, report the median and the interquartile range, which such outliers do not dominate.

The MyoSkeleton is not part of this evaluation.

## PPO, SAC and TD3

**Experiment 2** compares PPO with the off-policy algorithms [SAC and TD3](experimental.md) on the Unitree H1,
13 clips and 5 seeds, 65 runs per algorithm, all with the same tracking reward.

```{figure} ../_static/figures/benchmark_ppo_sac_td3.png
:alt: Training return and episode length of PPO, SAC and TD3 on the Unitree H1
:width: 100%
:align: center

Training return (left) and episode length (right) on the Unitree H1, mean ± standard deviation over 13 clips
and 5 seeds, for the first $3 \times 10^8$ environment steps.
```

SAC and TD3 rise faster in the first $5 \times 10^7$ steps, but PPO overtakes them. At the end of training, after
about $5 \times 10^8$ steps, PPO reaches a return of about 1,530, SAC about 1,270 and TD3 about 1,070. The
episode lengths converge to between 900 and 925 for all three. This is one reason why PPO with the tracking
reward is the default recipe of LocoMuJoCo.

## PPO and GAIL

**Experiment 3** compares the tracking reward with adversarial imitation on the Unitree H1. The two runs differ
only in the reward: the DeepMimic-style tracking reward for PPO, the output of the discriminator for
[GAIL](amp_gail.md). It uses 12 of the 13 clips (`onestepside1` is left out) and 5 seeds.

```{figure} ../_static/figures/benchmark_ppo_gail.png
:alt: Training curves and final tracking errors of PPO and GAIL on the Unitree H1
:width: 100%
:align: center

Top: training return and episode length, mean ± standard deviation over 12 clips and 5 seeds. Bottom: final
validation error under the three measures, on joint positions, site rotations and site positions, one box per
algorithm over 12 clips × 5 seeds (log scale).
```

PPO reaches about twice the return of GAIL and a lower tracking error under every measure and on every
quantity: the median error of PPO is about 1.8 to 2.2 times lower. Training a discriminator on high-dimensional whole-body motion without a per-step tracking signal is
hard. LocoMuJoCo includes GAIL mainly as a baseline and as a starting point for methods such as AMP, which
combine a learned style reward with a task reward.

## SuperMimic on the real-robot environments

**Experiment 4** trains one [SuperMimic](supermimic.md) policy per robot on the real-robot environments of
[LocoMuJoCo Real](../sim2real/index.md), on all 40 clips of LAFAN1. Each policy is trained for
$1.53 \times 10^8$ environment steps with 4,096 parallel MuJoCo Warp environments, with
[domain randomization](../sim2real/training.md#randomization), pushes and self-collisions. This takes about
50 minutes on one NVIDIA RTX 6000 Ada.

The trained policy then plays every clip once from its first frame, without noise or randomization. A clip
counts as failed if the episode terminates before the clip ends. The evaluation therefore measures tracking
quality, not robustness.

| Robot | Normalized return | Failed clips (of 40), per seed | Completed frames |
|---|---|---|---|
| AgiBot X2 Ultra | 0.776 ± 0.004 | 1, 0, 0 | 99.7 % |
| Unitree G1 | 0.674 ± 0.020 | 2, 0, 1 | 99.1 % |

The normalized return is the sum of rewards divided by the maximum attainable one, averaged over the clips,
as mean ± standard deviation over three seeds. Completed frames are averaged over the seeds.

The policy that runs on the real X2 Ultra in the [sim-to-real guide](../sim2real/robot.md#a-run-on-the-x2-ultra) was
trained in the same environment on the [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed)
dataset. In simulation it completes 99.9 % of the clips after
$4.56 \times 10^8$ environment steps.

## See also

- [Evaluation and metrics](evaluation.md) for computing the measures on your own policies.
- [Simulation backends](../user_guide/backends.md#throughput) for the throughput of the three backends.
- [DeepMimic-style tracking](deepmimic.md) for the recipe of experiment 1.
