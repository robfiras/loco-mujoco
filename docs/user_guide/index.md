# User guide

<p class="lmj-lead">The user guide explains how LocoMuJoCo is put together and how to use each part.
If you are new, work through <a href="../getting_started/index.html">Get started</a> first.</p>

## Concepts

How the main pieces relate to each other.

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} Simulation backends
:link: backends
:link-type: doc

MuJoCo on the CPU, MJX and MuJoCo Warp on the GPU, behind one environment API.
:::

:::{grid-item-card} Task factories
:link: task_factories
:link-type: doc

`ImitationFactory` and `RLFactory`: the entry points for creating environments.
:::

:::{grid-item-card} Trajectories
:link: trajectories
:link-type: doc

The trajectory data format and the `TrajectoryHandler` that samples from it.
:::

:::{grid-item-card} Datasets
:link: datasets
:link-type: doc

The native dataset, LAFAN1 and AMASS: what they contain and how they are loaded and cached.
:::

:::{grid-item-card} Retargeting
:link: retargeting
:link-type: doc

How human motion is mapped to a robot, and how motion is moved from one robot to another.
:::

:::{grid-item-card} Modular architecture
:link: modular_architecture
:link-type: doc

How an environment is assembled from interchangeable, registered modules.
:::
::::

## Components

One page per pluggable module, with the built-in options and how to configure them.

::::{grid} 1 2 3 4
:gutter: 3

:::{grid-item-card} Observations
:link: observations
:link-type: doc
:::

:::{grid-item-card} Rewards
:link: rewards
:link-type: doc
:::

:::{grid-item-card} Control functions
:link: control_functions
:link-type: doc
:::

:::{grid-item-card} Domain randomization
:link: domain_randomization
:link-type: doc
:::

:::{grid-item-card} Terrain
:link: terrain
:link-type: doc
:::

:::{grid-item-card} Initial and terminal states
:link: initial_terminal_states
:link-type: doc
:::

:::{grid-item-card} Rendering and recording
:link: rendering
:link-type: doc
:::

:::{grid-item-card} Gymnasium interface
:link: gymnasium
:link-type: doc
:::
::::

## Extending LocoMuJoCo

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} Writing custom modules
:link: custom_modules
:link-type: doc

Define and register your own observation, reward, control function or handler.
:::

:::{grid-item-card} Creating custom trajectories
:link: custom_trajectories
:link-type: doc

Build a trajectory from your own joint data and use it for imitation learning.
:::

:::{grid-item-card} Adding a new robot
:link: new_robot
:link-type: doc

Add a new embodiment and retarget the existing datasets to it.
:::
::::

```{toctree}
:hidden:
:caption: Concepts
:maxdepth: 1

backends
task_factories
trajectories
datasets
retargeting
modular_architecture
```

```{toctree}
:hidden:
:caption: Components
:maxdepth: 1

observations
rewards
control_functions
domain_randomization
terrain
initial_terminal_states
rendering
gymnasium
```

```{toctree}
:hidden:
:caption: Extending LocoMuJoCo
:maxdepth: 1

custom_modules
custom_trajectories
new_robot
```
