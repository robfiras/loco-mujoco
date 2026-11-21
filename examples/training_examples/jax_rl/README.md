## Reinforcement Learning Example

This example showcases the training of a locomotion policy using pure reinforcement learning on the Unitree Go2 robot. 
The policy is designed to enable the robot to walk in various directions and rotate around its z-axis. 
The training process employs the PPO algorithm and does not rely on any expert datasets.

The example utilizes the `GoalRandomRootVelocity` goal vector, which generates random target root velocities for the robot
to achieve. These velocities are set in the xy-plane and include rotation around the z-axis. This goal is 
integrated into the agent's observation space. Additionally, a classical locomotion reward, `LocomotionReward`, 
is used. This reward incorporates various penalty terms to promote smoothness and energy efficiency while ensuring that 
the robot accurately tracks the target velocity.

---

### 🚀 Training

To train the agent, run:

```bash
python experiment.py
```

This command will:

- Train the PPO agent on the Unitree Go2 robot for 100 million steps (approximately 20 minutes on an RTX 3080 Ti).
- Save the trained agent (as `PPOJax_saved.pkl` in the `outputs` folder).
- Perform a final rendering of the trained policy.
- Save a video of the rendering to the `LocoMuJoCo_recordings/` directory.
- Upload the video to Weights & Biases (WandB) for further analysis (check the command line logs for details).

---

### 📈 Evaluation

To evaluate the trained agent, run:

```bash
python eval.py --path path/to/agent_file
```

If you'd like to evaluate the agent using MuJoCo (instead of Mjx), run:

```bash
python eval.py --path path/to/agent_file --use_mujoco
```

> ⚠️ **Note:** Evaluating with MuJoCo may not yield results as robust as with Mjx due to simulator differences. For reliable policy transfer between the two, consider applying domain randomization techniques.
