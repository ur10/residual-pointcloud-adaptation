"""SAC over the residual, with the base policy held fixed.

Only the residual actor and the critics are updated. The action the critic sees
is the full composed action, so the value estimate is about what the robot
actually did, while the gradient that reaches the actor only moves the
correction.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class SACConfig:
    lr: float = 3e-4
    gamma: float = 0.97
    tau: float = 0.005
    target_entropy_scale: float = 0.5
    batch_size: int = 256


class ResidualSAC:
    def __init__(self, actor, critic, critic_target, cfg: SACConfig,
                 device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.critic_target = critic_target.to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.cfg = cfg
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.lr)
        self.target_entropy = -cfg.target_entropy_scale * actor.action_dim

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp().detach()

    def _to_torch(self, batch: dict) -> dict:
        return {k: torch.as_tensor(v, device=self.device) for k, v in batch.items()}

    def update(self, batch: dict) -> dict:
        b = self._to_torch(batch)
        obs, next_obs = b["obs"], b["next_obs"]
        base_action, delta = b["base_action"], b["delta"]
        action = base_action + delta

        # ---- critic ----
        with torch.no_grad():
            next_delta, next_logp = self.actor(next_obs)
            # the base action at the next state is part of next_obs by
            # construction (see ComposedPolicy._residual_input)
            next_base = next_obs[:, -self.actor.action_dim:]
            q_next = self.critic_target(next_obs, next_base + next_delta)
            target = b["reward"] + self.cfg.gamma * (1 - b["done"]) * (
                q_next.min(dim=0).values - self.alpha * next_logp)

        q = self.critic(obs, action)
        critic_loss = F.mse_loss(q[0], target) + F.mse_loss(q[1], target)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        # ---- actor ----
        new_delta, logp = self.actor(obs)
        q_pi = self.critic(obs, base_action + new_delta).min(dim=0).values
        actor_loss = (self.alpha * logp - q_pi).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        # ---- temperature ----
        alpha_loss = -(self.log_alpha.exp()
                       * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(),
                             self.critic_target.parameters()):
                tp.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p)

        return {
            "critic_loss": float(critic_loss),
            "actor_loss": float(actor_loss),
            "alpha": float(self.alpha),
            "delta_norm": float(new_delta.norm(dim=-1).mean()),
        }
