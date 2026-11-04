# Adapting Point-Cloud Policies to Real-World Manipulation

Point-cloud manipulation policies trained in simulation transfer poorly to real
robots, where depth observations contain missing points, occlusions and
calibration error. This repository adapts a **pretrained 3D policy without
changing the base model**, by learning a small residual policy on top of it with
off-policy reinforcement learning, and gating when that residual is allowed to
act.

The task used throughout is ring-on-peg insertion on a UR10 with a Robotiq 2F
gripper, observed by two RealSense cameras.

## Why a residual, and why a gate

Fine-tuning the base policy on the small amount of real interaction that is
practical to collect destroys the geometry prior that made it generalize in the
first place. So the base policy is **frozen** and treated as a fixed function of
the observation.

An unconstrained residual, though, is just a second policy: wherever the reward
is ambiguous it drifts the behaviour away from the base policy. Two things keep
that in check.

- **The correction is bounded by construction.** The residual actor's output is
  squashed through `tanh` and scaled by `max_delta`, so a single control step
  cannot move the end effector further than a configured distance. The bound is
  not a penalty term that training can trade away.
- **A gate decides when it applies at all.** The gate opens only when the critic
  prefers the corrected action to the base action, with hysteresis so it does
  not stutter at the control rate. When it is closed, the robot executes exactly
  the base policy, which makes the adapted policy a strict superset of the
  behaviour validated in simulation.

## Layout

```
residual_adapt/obs/pointcloud.py        depth -> fused, cropped, FPS-sampled cloud
residual_adapt/policies/base.py         frozen pretrained policy wrapper
residual_adapt/policies/residual.py     squashed-Gaussian residual + composed policy
residual_adapt/policies/gate.py         advantage gate with hysteresis
residual_adapt/policies/critic.py       twin Q, shared by the update and the gate
residual_adapt/train/sac.py             SAC over the residual, base held fixed
residual_adapt/train/buffer.py          replay over real transitions
residual_adapt/envs/ur10_ring_on_peg.py one interface, sim and hardware backends
scripts/train_residual.py               training entry point
scripts/eval_policy.py                  paired base-vs-adapted rollouts
tests/test_obs_and_bounds.py            geometry and safety-bound tests
```

The observation pipeline is worth a look even in isolation: fusing both camera
views, cropping to the workspace, dropping the speckle that depth holes leave
around the peg, and farthest-point sampling so the ring and peg survive
resampling instead of being swamped by the table.

## Status

Trained checkpoints are coming soon. The method itself is here: the observation
pipeline, the bounded residual, the gate and the SAC update. Videos of the base
and adapted policies on the real UR10 are on the
[project page](https://ur10.github.io/#p-manip).

## Running

```sh
pip install -r requirements.txt
python scripts/train_residual.py --config configs/ur10_ring_on_peg.yaml
python -m pytest tests/ -q
```

The tests cover the parts that need neither a robot nor a checkpoint, including
the two properties the safety argument rests on: that the residual's output
stays inside `max_delta` for arbitrary inputs, and that the gate holds its state
for `min_open_steps` before closing.

## License

MIT
