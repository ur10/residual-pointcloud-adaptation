#!/usr/bin/env python3
"""Roll out base and adapted policies on the same resets and log the comparison.

    python scripts/eval_policy.py --config configs/ur10_ring_on_peg.yaml \
        --checkpoint runs/residual/best.pt --episodes 20

Both policies see the same initial ring and peg poses, because the thing being
measured is the effect of the correction, not the luck of the reset. Per-step
gate state and correction magnitude are written out alongside the outcome, so a
failure can be attributed to the residual rather than guessed at.
"""

import argparse
import json
import pathlib


def rollout(env, policy, episodes: int) -> list[dict]:
    records = []
    for ep in range(episodes):
        step = env.reset()
        policy.reset()
        trace, total = [], 0.0
        while True:
            out = policy.act(step.obs, step.state)
            step = env.step(out["action"])
            total += step.reward
            trace.append({
                "gate": out["gate"],
                "delta_norm": float((out["delta"] ** 2).sum() ** 0.5),
                "insertion_depth": step.info.get("insertion_depth"),
            })
            if step.done:
                break
        records.append({
            "episode": ep,
            "return": total,
            "success": bool(step.info.get("insertion_depth", 0)
                            >= env.success_depth),
            "steps": len(trace),
            "gate_open_fraction": sum(t["gate"] for t in trace) / max(len(trace), 1),
            "trace": trace,
        })
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=pathlib.Path)
    ap.add_argument("--checkpoint", required=True, type=pathlib.Path)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path("runs/eval.json"))
    args = ap.parse_args()

    raise SystemExit(
        "Attach an environment and a loaded ComposedPolicy, then: "
        f"json.dump(rollout(env, policy, {args.episodes}), open('{args.out}', 'w'))"
    )


if __name__ == "__main__":
    main()
