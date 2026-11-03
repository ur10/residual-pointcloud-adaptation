"""The residual actor and the composed policy that deploys it."""

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from .gate import AdvantageGate


def mlp(sizes: Sequence[int], out_dim: int, activation=nn.ReLU) -> nn.Sequential:
    layers: list[nn.Module] = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), activation()]
    layers += [nn.Linear(sizes[-1], out_dim)]
    return nn.Sequential(*layers)


class ResidualActor(nn.Module):
    """Squashed-Gaussian actor over bounded corrections to the base action.

    The output is passed through tanh and scaled by `max_delta`, so the
    correction is bounded by construction rather than by a penalty term. On
    hardware that bound is the safety argument: whatever the residual has
    learned, a single step cannot move the end effector further than
    `max_delta.translation` metres.
    """

    LOG_STD_MIN, LOG_STD_MAX = -10.0, 2.0

    def __init__(self, obs_dim: int, action_dim: int, max_delta: np.ndarray,
                 hidden_sizes: Sequence[int] = (256, 256)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden_sizes], 2 * action_dim)
        self.action_dim = action_dim
        self.register_buffer("max_delta",
                             torch.as_tensor(max_delta, dtype=torch.float32))

    def forward(self, obs: torch.Tensor, deterministic: bool = False,
                with_logprob: bool = True):
        mu, log_std = self.net(obs).chunk(2, dim=-1)
        log_std = log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()

        dist = torch.distributions.Normal(mu, std)
        raw = mu if deterministic else dist.rsample()

        logp = None
        if with_logprob:
            # tanh change of variables, in the numerically stable form
            logp = dist.log_prob(raw).sum(-1)
            logp -= (2 * (np.log(2) - raw - nn.functional.softplus(-2 * raw))).sum(-1)

        delta = torch.tanh(raw) * self.max_delta
        return delta, logp


class ComposedPolicy(nn.Module):
    """base action + gate * bounded residual.

    This is what runs on the robot. The base policy's output is computed first
    and is what gets executed whenever the gate is closed, which makes the
    adapted policy a strict superset of the behaviour that was validated in
    simulation.
    """

    def __init__(self, base, residual: ResidualActor, gate: AdvantageGate,
                 critic=None, condition_on: str = "features"):
        super().__init__()
        self.base = base
        self.residual = residual
        self.gate = gate
        self.critic = critic
        self.condition_on = condition_on

    def _residual_input(self, points: torch.Tensor, state: torch.Tensor,
                        base_action: torch.Tensor) -> torch.Tensor:
        if self.condition_on == "features":
            feats = self.base.features(points, state)
            return torch.cat([feats, state, base_action], dim=-1)
        return torch.cat([state, base_action], dim=-1)

    @torch.no_grad()
    def act(self, points: np.ndarray, state: np.ndarray,
            deterministic: bool = True) -> dict:
        device = next(self.residual.parameters()).device
        p = torch.as_tensor(points, dtype=torch.float32, device=device)[None]
        s = torch.as_tensor(state, dtype=torch.float32, device=device)[None]

        base_action = self.base(p, s)
        obs = self._residual_input(p, s, base_action)
        delta, _ = self.residual(obs, deterministic=deterministic,
                                 with_logprob=False)

        gate_value = 1.0
        if self.critic is not None:
            q_corrected = self.critic(obs, base_action + delta).min(dim=0).values
            q_base = self.critic(obs, base_action).min(dim=0).values
            gate_value = float(self.gate(q_corrected, q_base, hard=True).item())
        gate_value = self.gate.step(gate_value)

        action = base_action + gate_value * delta
        return {
            "action": action[0].cpu().numpy(),
            "base_action": base_action[0].cpu().numpy(),
            "delta": delta[0].cpu().numpy(),
            "gate": gate_value,
        }

    def reset(self) -> None:
        self.gate.reset()
