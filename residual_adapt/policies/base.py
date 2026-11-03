"""Wrapper around the frozen, sim-trained point-cloud policy.

The base policy is never updated. Its weights carry the geometry prior that
makes it generalize across object poses, and fine-tuning it on the small amount
of real interaction data available here is the fastest way to destroy that
prior. Everything downstream treats it as a fixed function of the observation.
"""

from typing import Protocol

import numpy as np
import torch
import torch.nn as nn


class PointCloudPolicy(Protocol):
    """Minimal interface the adaptation stack needs from a pretrained policy."""

    def act(self, points: np.ndarray, state: np.ndarray) -> np.ndarray: ...


class FrozenBasePolicy(nn.Module):
    """Loads a pretrained checkpoint and exposes it read-only.

    The checkpoint is expected to hold a module that maps
    (B, N, 3) points and (B, state_dim) proprioception to (B, action_dim).
    Swap `_load` for whatever the upstream repo's loader is; the rest of the
    stack only depends on the signature.
    """

    def __init__(self, checkpoint: str, action_dim: int, device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        self.net = self._load(checkpoint).to(self.device)

        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _load(checkpoint: str) -> nn.Module:
        obj = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if isinstance(obj, nn.Module):
            return obj
        if isinstance(obj, dict) and "model" in obj:
            return obj["model"]
        raise ValueError(
            f"{checkpoint} does not contain a module or a 'model' entry; "
            "adapt FrozenBasePolicy._load to the upstream checkpoint format."
        )

    @torch.no_grad()
    def forward(self, points: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.net(points, state)

    @torch.no_grad()
    def act(self, points: np.ndarray, state: np.ndarray) -> np.ndarray:
        p = torch.as_tensor(points, dtype=torch.float32, device=self.device)[None]
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device)[None]
        return self.forward(p, s)[0].cpu().numpy()

    def features(self, points: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Encoder features, if the base model exposes them.

        The residual conditions on these rather than re-encoding the cloud from
        scratch: it keeps the correction in the same representation the base
        policy is acting on, and it is far cheaper to train.
        """
        if hasattr(self.net, "encode"):
            with torch.no_grad():
                return self.net.encode(points, state)
        raise NotImplementedError(
            "base model exposes no encode(); set residual.condition_on: action"
        )
