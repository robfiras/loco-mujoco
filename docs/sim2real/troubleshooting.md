# Troubleshooting and limitations

## When it works in simulation but not on the robot

| Symptom | Where to look |
|---|---|
| The robot shakes or hums while standing | Gains or armature differ from the drives; control latency. Compare a small sine on a single joint between the robot and the model. |
| The policy fails in the first step | Joint order or sign, a normalizer input that was constant in training, a stale first state. |
| The robot drifts or turns away from the reference | Heading alignment at the start; the policy reads a world-frame quantity. |
| Joints lag behind the model | Friction in the real joints that the model lacks. Test with a plant that has `frictionloss` in [stage 4](sim2sim.md#stage-4-synchronous-sim-to-sim). |
| Single late control steps | BLAS threads, rendering in the control process, message callbacks. |
| A fall when a clip ends or is switched | The clip does not start or end standing. |

## Current limitations

These are limitations of the deployment tools at the time of writing; `deployment/REFERENCE.md` in
LocoMuJoCo Real has the current list.

- The exporter supports the position control function of LocoMuJoCo Real and the observation terms that
  `lmj_deploy.py` implements. Of SuperMimic's actors, it supports the encoder-decoder one.
- Observation delay and observation history exist in the environments but are off by default, and a bundle
  does not carry them. A policy trained with them cannot be exported faithfully yet.
- Only the mean action is exported, so the deployed policy is deterministic.
- Reference clips can ask for joint positions beyond the joint range. Look at a clip before playing it on the
  robot.
