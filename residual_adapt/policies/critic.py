"""Twin Q networks, used both for the SAC update and for the gate."""

from typing import Sequence

import torch
import torch.nn as nn

from .residual import mlp


class TwinQ(nn.Module):
    """Two independent critics; the minimum is used as the value estimate.

    The gate reads the same critics, so the decision to apply a correction and
    the decision to train toward it come from one value estimate rather than
    two that can disagree.
    """

    def __init__(self, obs_dim: int, action_dim: int,
                 hidden_sizes: Sequence[int] = (256, 256)):
        super().__init__()
        self.q1 = mlp([obs_dim + action_dim, *hidden_sizes], 1)
        self.q2 = mlp([obs_dim + action_dim, *hidden_sizes], 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return torch.stack([self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)])
