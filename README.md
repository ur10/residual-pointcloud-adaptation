# Fine-Grained Point-Cloud Manipulation with Residual RL

This repository studies how a pretrained **point-cloud behavior cloning (BC) policy** can be adapted for fine-grained manipulation using **residual reinforcement learning**. The base policy provides a 3D manipulation prior, while a separate residual policy learns small, closed-loop action corrections. The pretrained policy remains frozen throughout residual training.

The central idea is to preserve the geometric generalization of a 3D policy while improving control during alignment and contact.

## Why Point Clouds?

Manipulation depends on spatial relationships between the robot, the object, and the target. A point cloud directly represents observed surfaces in 3D, making it useful for reasoning about relative pose, clearance, and approach direction.

Our base policy follows a **DP3-style point-cloud diffusion policy**. Compared with using RGB images alone, its 3D input exposes metric geometry that is useful when objects are repositioned or approached from different directions. The residual policy builds on this geometric prior rather than retraining it.

## Manipulation Scenes

The project focuses on tasks where a good approach trajectory must be followed by precise alignment or contact-aware control.

| Scene | Manipulation objective | Role in the experiments |
| --- | --- | --- |
| **RoboMimic Square** | Grasp a square nut and place it onto a peg | Precision alignment and insertion |
| **Adroit Pen** | Reorient and manipulate a pen with a dexterous hand | Fine-grained corrections during contact and reorientation |
| **Adroit Door** | Manipulate a door handle and open the door | Evaluate whether residual control preserves a strong base behavior |
| **UR10 ring-on-peg** | Align and insert a ring onto a peg using a UR10 and Robotiq 2F gripper | Real-robot manipulation setup using two RealSense cameras |

## Policy Architecture

The policy has two action pathways:

1. **Frozen point-cloud BC:** a pretrained 3D diffusion policy produces the base action, `a_base`, from the point-cloud observation and robot state.
2. **Residual RL:** a separate actor produces a small action correction, `delta_a`, using additional observations. The residual is trained with off-policy reinforcement learning while the BC policy remains fixed.

The executed action is

```text
a_exec = a_base + gate * delta_a
```

The residual actor uses a squashed action distribution, and `max_delta` scales the correction. A critic-based gate compares the corrected action against the base action and uses hysteresis to avoid rapid switching. When the gate is closed, the controller executes the base action without a residual correction.

The residual pathway can use robot state, RGB observations, or point-cloud features. This lets us study which observations help most during fine-grained manipulation while keeping the base policy unchanged.

## Point-Cloud Observation Pipeline

For the UR10 scene, the depth-processing pipeline:

- Fuses observations from two RealSense cameras.
- Transforms the points into a common frame and crops them to the manipulation workspace.
- Filters isolated depth points and artifacts around the ring and peg.
- Uses farthest-point sampling (FPS) to form a fixed-size input cloud for the policy.

The observation pipeline and policy are separated so the same residual-learning structure can be used across simulation and robot scenes.

## Repository Structure

```text
residual_adapt/obs/pointcloud.py         depth to fused, cropped, sampled point cloud
residual_adapt/policies/base.py          frozen pretrained BC policy wrapper
residual_adapt/policies/residual.py      residual actor and composed policy
residual_adapt/policies/gate.py          critic-based gate with hysteresis
residual_adapt/policies/critic.py        twin Q-functions
residual_adapt/train/sac.py              off-policy residual SAC update
residual_adapt/train/buffer.py           transition replay buffer
residual_adapt/envs/ur10_ring_on_peg.py  simulation and hardware environment interface
scripts/train_residual.py                training entry point
scripts/eval_policy.py                   base and residual policy evaluation
tests/test_obs_and_bounds.py             observation and action-bound tests
```

## Running

```bash
pip install -r requirements.txt
python scripts/train_residual.py --config configs/ur10_ring_on_peg.yaml
python -m pytest tests/ -q
```

Project videos and additional context are available on the [project page](https://ur10.github.io/#p-manip).

## License

MIT
