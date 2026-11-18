# Get started

<p class="lmj-lead">Five short pages take you from an empty Python environment to a humanoid robot that tracks a
motion capture clip. Each page ends with something you can see.</p>

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 1 · Installation
:link: installation
:link-type: doc

Install LocoMuJoCo, add GPU support and pick the optional extras.
+++
5 minutes
:::

:::{grid-item-card} 2 · Your first environment
:link: first_environment
:link-type: doc

Create a Unitree G1, step it with random actions and look at observations and actions.
+++
5 minutes · CPU
:::

:::{grid-item-card} 3 · Replay a motion
:link: replay_motion
:link-type: doc

Load motion capture clips from the hosted datasets and play them back on the robot.
+++
5 minutes · CPU
:::

:::{grid-item-card} 4 · Train a tracking policy
:link: train_policy
:link-type: doc

Train a PPO policy with a DeepMimic-style reward on thousands of parallel MJX environments.
+++
30 to 60 minutes · GPU
:::

:::{grid-item-card} 5 · Evaluate and customize
:link: next_steps
:link-type: doc

Evaluate the policy, swap an observation or a reward, and find your way around the rest of the docs.
+++
10 minutes
:::

:::{grid-item-card} Optional · AMASS and SMPL setup
:link: amass
:link-type: doc

Set up the license-restricted AMASS dataset and the SMPL body model for more than 10,000 extra motions.
+++
20 minutes
:::
::::

```{toctree}
:hidden:
:maxdepth: 1

installation
first_environment
replay_motion
train_policy
next_steps
amass
```
