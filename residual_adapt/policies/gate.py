"""When the residual is allowed to act.

An always-on residual is a second policy, and it will drift the behaviour away
from the base policy wherever the reward is ambiguous - which on a real robot
is most of the time. The gate keeps the correction switched off unless there is
evidence it helps, so the failure mode of the adapted policy is "behaves like
the base policy", not "behaves like a half-trained residual".
"""

import torch
import torch.nn as nn


class AdvantageGate(nn.Module):
    """Opens when the critic prefers the corrected action to the base action.

    g = sigmoid((Q(s, a_base + delta) - Q(s, a_base)) / temperature)

    The soft form keeps the gate differentiable for the actor update. At
    evaluation time `hard=True` thresholds it, so a deployed episode is a
    sequence of discrete on/off decisions that can be logged and inspected
    rather than a continuously varying blend.
    """

    def __init__(self, threshold: float = 0.0, temperature: float = 0.05,
                 min_open_steps: int = 1):
        super().__init__()
        self.threshold = threshold
        self.temperature = max(temperature, 1e-6)
        self.min_open_steps = min_open_steps
        self._open_for = 0

    def forward(self, q_corrected: torch.Tensor, q_base: torch.Tensor,
                hard: bool = False) -> torch.Tensor:
        advantage = q_corrected - q_base
        gate = torch.sigmoid((advantage - self.threshold) / self.temperature)
        if hard:
            gate = (advantage > self.threshold).to(gate.dtype)
        return gate

    def step(self, gate_value: float) -> float:
        """Hysteresis for rollouts.

        Flipping the gate on and off between adjacent control steps produces a
        visible stutter at 10 Hz. Once open, it stays open for at least
        `min_open_steps` steps.
        """
        if gate_value > 0.5:
            self._open_for = self.min_open_steps
        elif self._open_for > 0:
            self._open_for -= 1
            return 1.0
        return float(gate_value > 0.5)

    def reset(self) -> None:
        self._open_for = 0
