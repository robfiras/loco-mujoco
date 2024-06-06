
import mediapy as media
import matplotlib.pyplot as plt
import mujoco
path = "stompy.xml"
mj_model = mujoco.MjModel.from_xml_path(path)
mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
mj_model.opt.iterations = 6
data = mujoco.MjData(mj_model)
model = mj_model
mujoco.mj_forward(mj_model, data)
data.qpos = data.qpos + 0.5
mujoco.mj_saveLastXML("stompy_test.xml", mj_model)




# mypy: disable-error-code="valid-newtype"
"""Defines a simple demo script for simulating a MJCF to observe the physics.

Run with mjpython:
    mjpython sim/scripts/simulate_mjcf.py --record
"""

import argparse
import logging
import time
from pathlib import Path
from typing import List, Union

import mediapy as media
import mujoco
import mujoco.viewer
import numpy as np


new_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 1.03673, -2.2774, 0.0, 0.0, 0.0, -1.571, -1.47674, -1.69123, -0.01989, 0.0, 0.0, 0.0, -1.3632, 0.0, -0.67553, 0.0, -1.26711, 0.0, -0.03281, -1.94804, 0.0, 0.0, -0.824282, -1.2408, 0.0, 0.0, 0.0, -0.644028, -0.683298, -0.61773, -1.03431, 0.0, 0.0, 0.7854, 0.0, 0.0, 0.0]



def simulate(model_path: Union[str, Path], duration: float, framerate: float, record_video: bool) -> None:
    frames: List[np.ndarray] = []
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while data.time < duration:
            step_start = time.time()

            mujoco.mj_step(model, data)
            data.qpos[:] = new_pose
            viewer.sync()

            # # Print the positions and quaternions of all bodies
            # print("Body Positions and Quaternions:")
            # for i in range(model.nbody):
            #     body_name = model.names[model.name_bodyadr[i]:].split(b'\x00')[0].decode()
            #     body_pos = data.xpos[i]
            #     body_quat = data.xquat[i]
            #     print(f"Body {i} ({body_name}): Position = {body_pos}, Quaternion = {body_quat}")

            print("Joint Values:")
            for i in range(model.njnt):
                joint_name = model.names[model.name_jntadr[i]:].split(b'\x00')[0].decode()
                joint_value = data.qpos[i]
                print(f"Joint {i} ({joint_name}): Value = {joint_value}")
            # # Print the positions and quaternions of all geoms
            # print("\nGeom Positions and Quaternions:")
            # for i in range(model.ngeom):
            #     geom_name = model.names[model.name_geomadr[i]:].split(b'\x00')[0].decode()
            #     geom_pos = data.geom_xpos[i]

            #     # Convert the geom orientation matrix to a quaternion
            #     geom_mat = data.geom_xmat[i].reshape(3, 3)
            #     geom_quat = mujoco.mju_mat2Quat(geom_mat)
            #     print(f"Geom {i} ({geom_name}): Position = {geom_pos}, Quaternion = {geom_quat}")

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            if record_video and (len(frames) < data.time * framerate):
                renderer.update_scene(data)
                pixels = renderer.render()
                frames.append(pixels)

        if record_video:
            video_path = "mjcf_simulation.mp4"
            media.write_video(video_path, frames, fps=framerate)
            # print(f"Video saved to {video_path}")
            logger.info("Video saved to %s", video_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="MuJoCo Simulation")
    parser.add_argument(
        "--model_path", type=str, default="stompy.xml", help="Path to the MuJoCo XML file"
    )
    parser.add_argument("--duration", type=int, default=3, help="Duration of the simulation in seconds")
    parser.add_argument("--framerate", type=int, default=30, help="Frame rate for video recording")
    parser.add_argument("--record", action="store_true", help="Flag to record video")
    args = parser.parse_args()

    simulate(args.model_path, args.duration, args.framerate, args.record)
