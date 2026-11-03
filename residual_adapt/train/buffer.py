"""Replay buffer over real-robot transitions.

Off-policy is the point: real interaction is expensive and slow, so every
episode collected has to stay usable for the rest of training. The buffer keeps
the base action alongside the correction, because the SAC update is over the
residual only - the base action is part of the state as far as the critic is
concerned.
"""

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.base_action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.delta = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs, base_action, delta, reward, next_obs, done) -> None:
        i = self.ptr
        self.obs[i] = obs
        self.base_action[i] = base_action
        self.delta[i] = delta
        self.reward[i] = reward
        self.next_obs[i] = next_obs
        self.done[i] = float(done)
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng()
        idx = rng.integers(0, self.size, size=batch_size)
        return {
            "obs": self.obs[idx],
            "base_action": self.base_action[idx],
            "delta": self.delta[idx],
            "reward": self.reward[idx],
            "next_obs": self.next_obs[idx],
            "done": self.done[idx],
        }

    def __len__(self) -> int:
        return self.size
