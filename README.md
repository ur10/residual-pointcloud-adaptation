# Fine-Grained Point-Cloud Manipulation with Residual RL

A manipulation policy can bring an object close to its target yet struggle with the final
alignment. This project studies whether small learned corrections can adapt a pretrained
point-cloud policy for precise manipulation while keeping its existing behavior fixed.
The physical demonstration is ring-on-peg insertion with a UR10 and two RealSense cameras.

The base policy proposes an action; a separate residual controller proposes a bounded
correction. A critic-based gate selects when to use that correction. The research goal is
to improve alignment and contact behavior under imperfect observations.

## Public release and evidence

The portfolio shows example hardware runs comparing the base and adapted policies.
This repository provides the observation processing, residual actor, critic, gate, SAC
update, and environment interfaces. It does not include aggregate hardware success rates,
trained checkpoints, camera/controller drivers, or the lab's reset and re-grasp routine.

Training and evaluation entry points require those integrations before they can collect
episodes. The listed simulation tasks describe the experimental scope; their simulator
adapters and comparative results are not included in this release.

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

## Setup and integration

Install the dependencies and run the component tests from the repository root:

```bash
pip install -r requirements.txt
python -m pytest tests/ -q
```

To train or evaluate against an environment:

1. Supply a compatible pretrained checkpoint and adapt `FrozenBasePolicy._load` to its format.
2. Connect the simulator or the camera, controller, and pose-tracker interfaces. For hardware,
   implement the reset and re-grasp routine in `RealUR10.reset`.
3. Complete episode collection in `scripts/train_residual.py` and connect observations to the
   residual input representation.
4. Load the trained policy and environment into `scripts/eval_policy.py`. Use matched initial
   conditions for base and adapted policies, and report trial counts alongside success rates.

The entry points currently stop with integration instructions; they are not complete
training or evaluation commands. The configuration records the intended UR10 setup,
including correction bounds and the control rate.

[Watch the example hardware comparison and read the project context](https://ur10.github.io/#p-manip).

## License

MIT
