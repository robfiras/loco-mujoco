## DeepMimic Imitation Learning Example

This example demonstrates training a PPO agent on the Unitree G1 robot using a DeepMimic‑style reward
to imitate a walking clip from the LAFAN1 dataset (`walk1_subject5`). Other clips and robots can be selected in `conf.yaml`.

The example leverages the `GoalTrajMimic` goal vector, which encodes target positions, orientations, and velocities
of both site and joint states from the expert dataset. The reward function defined in the configuration, `MimicReward`,
implements DeepMimic‑style reward. By the end of training, the agent tracks the reference motion.

---

### 🚀 Training

To train the agent, run:

```bash
python experiment.py
```

This command will:

- Train the PPO agent on the Unitree G1 robot for 300 million steps (the Unitree H1 version of this example took approximately 36 minutes on an RTX 3080 Ti). To get even smoother policies train it for longer!
- Save the trained agent (as `PPOJax_saved.pkl` in the `outputs` folder).
- Perform a final rendering of the trained policy.
- Save a video of the rendering to the `LocoMuJoCo_recordings/` directory.
- Upload the video to Weights & Biases (WandB) for further analysis (check the command line logs for details).


#### Validation Loop During Training

Throughout training, the agent will be evaluated using various trajectory-based metrics, including 
Euclidean distance, Dynamic Time Warping (DTW), and discrete Fréchet distance. These metrics will be 
computed on different entities such as joint positions, joint velocities, and site positions and orientations. 
All results will be logged to Weights & Biases (WandB).

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
