"""
Rendering in the browser with viser.

Every environment has a `render_viser()` counterpart to `render()` (and `mjx_render_viser()`
for the parallel Mjx path). Instead of opening an OpenGL window, the scene is served over the
network, which makes it usable over SSH and on machines without a display.

Requires the optional dependencies:

    pip install loco-mujoco[viser]

Open the URL printed on startup (http://localhost:8080 by default) to see the robot.
"""
import numpy as np
from loco_mujoco import ImitationFactory


# `port`, `host` and `verbose` configure the viser server; all other viewer parameters
# (camera_params, default_camera_mode, geom_group_visualization_on_startup, ...) are the same
# ones the OpenGL viewer accepts.
env, traj = ImitationFactory.make("UnitreeH1",
                                  default_dataset_conf=dict(task="walk"),
                                  # GoalTrajMimicv2 draws the target pose as a translucent ghost
                                  # robot, i.e. a full copy of the robot's geoms
                                  goal_type="GoalTrajMimicv2",
                                  goal_params=dict(visualize_goal=True),
                                  port=8080)

action_dim = env.info.action_space.shape[0]

env.reset()
env.render_viser()

absorbing = False
i = 0

while True:
    if i == 1000 or absorbing:
        env.reset()
        i = 0
    action = np.random.randn(action_dim)
    nstate, reward, absorbing, done, info = env.step(action)

    env.render_viser()
    i += 1

# Dataset replay works the same way, via the `viser` flag:
#
#     env.play_trajectory(n_episodes=3, viser=True)
#
# And for the parallel Mjx path:
#
#     env.mjx_render_viser(state)
#
# Note on recording: `record=True` reads frames back from a connected browser
# (`client.get_render`). Open the viser URL before recording, otherwise the frames are empty.
