#!/usr/bin/env python3
"""Train the residual against a frozen base policy.

    python scripts/train_residual.py --config configs/ur10_ring_on_peg.yaml

Runs sim-first: warm-up transitions come from the simulator so the critic is
not random when the residual first touches hardware.
"""

import argparse
import pathlib

import numpy as np
import torch
import yaml

from residual_adapt.policies.base import FrozenBasePolicy
from residual_adapt.policies.critic import TwinQ
from residual_adapt.policies.gate import AdvantageGate
from residual_adapt.policies.residual import ComposedPolicy, ResidualActor
from residual_adapt.train.buffer import ReplayBuffer
from residual_adapt.train.sac import ResidualSAC, SACConfig


def max_delta_vector(cfg: dict) -> np.ndarray:
    d = cfg["residual"]["max_delta"]
    return np.array([d["translation"]] * 3 + [d["rotation"]] * 3 + [d["gripper"]],
                    dtype=np.float32)


def build(cfg: dict, obs_dim: int):
    action_dim = cfg["base_policy"]["action_dim"]
    base = FrozenBasePolicy(cfg["base_policy"]["checkpoint"], action_dim)
    actor = ResidualActor(obs_dim, action_dim, max_delta_vector(cfg),
                          cfg["residual"]["hidden_sizes"])
    critic = TwinQ(obs_dim, action_dim, cfg["residual"]["hidden_sizes"])
    target = TwinQ(obs_dim, action_dim, cfg["residual"]["hidden_sizes"])
    gate = AdvantageGate(**{k: v for k, v in cfg["residual"]["gate"].items()
                            if k != "type"})
    policy = ComposedPolicy(base, actor, gate, critic)
    return policy, ResidualSAC(actor, critic, target,
                               SACConfig(lr=cfg["train"]["lr"],
                                         gamma=cfg["train"]["gamma"],
                                         tau=cfg["train"]["tau"],
                                         batch_size=cfg["train"]["batch_size"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=pathlib.Path)
    ap.add_argument("--obs-dim", type=int, default=519,
                    help="base encoder features + state + base action")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("runs/residual"))
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    rng = np.random.default_rng(cfg["train"]["seed"])
    torch.manual_seed(cfg["train"]["seed"])

    policy, learner = build(cfg, args.obs_dim)
    buffer = ReplayBuffer(cfg["train"]["buffer_capacity"], args.obs_dim,
                          cfg["base_policy"]["action_dim"])

    # Episode collection is the part that needs the robot. Attach an env from
    # residual_adapt.envs and fill the buffer here; the update loop below is
    # backend-agnostic.
    raise SystemExit(
        "Attach an environment (residual_adapt.envs.ur10_ring_on_peg) to collect "
        f"transitions, then call learner.update(buffer.sample({cfg['train']['batch_size']})) "
        f"for {cfg['train']['steps']} steps, checkpointing to {args.out}."
    )


if __name__ == "__main__":
    main()
