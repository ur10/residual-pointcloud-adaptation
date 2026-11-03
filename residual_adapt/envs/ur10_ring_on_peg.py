"""Environment interface for the UR10 ring-on-peg task.

One interface, two backends: the simulator the base policy was trained in and
the real cell. Keeping them behind the same signature is what makes a
sim-trained policy loadable against hardware without touching the policy code,
and it is where the mismatch this project is about becomes visible - the real
backend returns the depth that has holes in it.

`RealUR10` is the integration point for the lab's controller and camera
drivers; the calls are named but not implemented here.
"""

from dataclasses import dataclass

import numpy as np

from ..obs.pointcloud import CameraIntrinsics, build_observation


@dataclass
class Step:
    obs: np.ndarray
    state: np.ndarray
    reward: float
    done: bool
    info: dict


class RingOnPegEnv:
    """Common reward and termination logic for both backends."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.max_steps = cfg["task"]["max_steps"]
        self.success_depth = cfg["task"]["success_depth"]
        self._t = 0

    def reward(self, info: dict) -> float:
        """Dense shaping toward the peg, plus a terminal bonus on seating.

        Distance is measured ring-centre to peg-axis rather than end effector
        to peg, so the reward does not credit approaches that bring the gripper
        close while holding the ring at the wrong angle.
        """
        lateral = info["ring_to_axis"]
        depth = info["insertion_depth"]
        shaped = -2.0 * lateral + 4.0 * max(depth, 0.0)
        return shaped + (10.0 if depth >= self.success_depth else 0.0)

    def done(self, info: dict) -> bool:
        return (info["insertion_depth"] >= self.success_depth
                or self._t >= self.max_steps
                or info.get("protective_stop", False))


class SimRingOnPeg(RingOnPegEnv):
    """Backend used to collect the warm-up transitions before touching hardware."""

    def __init__(self, cfg: dict, sim):
        super().__init__(cfg)
        self.sim = sim

    def reset(self) -> Step:
        self._t = 0
        raw = self.sim.reset()
        return self._observe(raw, reward=0.0, done=False)

    def step(self, action: np.ndarray) -> Step:
        self._t += 1
        raw = self.sim.step(action)
        info = raw["info"]
        return self._observe(raw, self.reward(info), self.done(info))

    def _observe(self, raw: dict, reward: float, done: bool) -> Step:
        obs = build_observation(raw["depths"], raw["intrinsics"],
                                raw["extrinsics"], self.cfg["observation"])
        return Step(obs, raw["state"], reward, done, raw["info"])


class RealUR10(RingOnPegEnv):
    """Backend for the physical cell: UR10 + Robotiq 2F + two RealSense cameras.

    Expects an `rtde` control interface and a camera driver exposing
    `.frames()`. Both are injected so this module stays importable on a machine
    with no robot attached.
    """

    def __init__(self, cfg: dict, rtde, cameras, pose_tracker):
        super().__init__(cfg)
        self.rtde = rtde
        self.cameras = cameras
        self.pose_tracker = pose_tracker
        self.intrinsics = [CameraIntrinsics.from_matrix(k)
                           for k in cameras.intrinsic_matrices()]
        self.extrinsics = cameras.extrinsics()

    def reset(self) -> Step:
        self._t = 0
        self.rtde.move_to_home()
        raise NotImplementedError(
            "reset() needs the lab's re-grasp routine: home the arm, re-present "
            "the ring to the gripper, confirm the peg pose with pose_tracker."
        )

    def step(self, action: np.ndarray) -> Step:
        self._t += 1
        self.rtde.servo_delta(action[:6], gripper=action[6])
        depths = self.cameras.frames()["depth"]
        state = self.rtde.tcp_state()
        info = self.pose_tracker.estimate()
        obs = build_observation(depths, self.intrinsics, self.extrinsics,
                                self.cfg["observation"])
        return Step(obs, state, self.reward(info), self.done(info), info)
