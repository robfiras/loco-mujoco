# Setup

<ul class="lmj-steps">
  <li class="current">Setup</li>
  <li><a href="training.html">1 · Train</a></li>
  <li><a href="export.html">2–3 · Export and validate</a></li>
  <li><a href="sim2sim.html">4–5 · Sim-to-sim</a></li>
  <li><a href="robot.html">6 · On the robot</a></li>
</ul>

Training needs a GPU. Install LocoMuJoCo 1.2.0 or newer with the extras for retargeting and for the browser
viewers, then LocoMuJoCo Real from a checkout of its repository:

```bash
pip install "loco-mujoco[smpl,viser]>=1.2.0"
```

```bash
git clone https://github.com/robfiras/loco-mujoco-real
cd loco-mujoco-real
pip install -e .
```

Tell LocoMuJoCo where the retargeting configurations of the two robots are:

```bash
loco-mujoco-add-variable --name LOCOMUJOCO_CUSTOM_ROBOT_CONF_PATH --value /path/to/loco-mujoco-real/lmj_real/smpl_robot_confs
```

The reference clips are retargeted to the robot the first time they are used, which needs the SMPL-H model
(see [AMASS and SMPL](../getting_started/amass.md)). The AMASS dataset itself is not needed: the source
clips are downloaded from the
[LocoMuJoCo dataset repository](https://huggingface.co/datasets/robfiras/loco-mujoco-datasets).

The commands in this guide are run from the root of the LocoMuJoCo Real repository. `<robot>` is `g1` or
`x2`.

Next: [train in a deployable environment](training.md).
