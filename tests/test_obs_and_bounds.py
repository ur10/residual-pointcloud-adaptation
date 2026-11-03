"""Tests over the parts that run without a robot or a checkpoint."""

import numpy as np

from residual_adapt.obs.pointcloud import (CameraIntrinsics, crop, deproject,
                                           farthest_point_sample, normalize,
                                           transform, voxel_downsample)

INTR = CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0)
BOUNDS = {"lower": [-0.35, -0.55, 0.0], "upper": [0.35, 0.10, 0.45]}


def test_deprojection_round_trips_a_known_point():
    depth = np.zeros((480, 640), dtype=np.float32)
    depth[240, 320] = 0.5                      # principal point, half a metre
    pts = deproject(depth, INTR)
    assert pts.shape == (1, 3)
    assert np.allclose(pts[0], [0.0, 0.0, 0.5], atol=1e-6)


def test_deprojection_drops_invalid_and_truncated_depth():
    depth = np.zeros((8, 8), dtype=np.float32)
    depth[0, 0] = 0.0                          # invalid
    depth[1, 1] = 5.0                          # beyond depth_trunc
    depth[2, 2] = np.nan
    depth[3, 3] = 0.4                          # the only good one
    assert len(deproject(depth, INTR, depth_trunc=1.2)) == 1


def test_transform_applies_rotation_and_translation():
    pts = np.array([[1.0, 0.0, 0.0]])
    T = np.eye(4)
    T[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]   # +90 deg about z
    T[:3, 3] = [0.0, 0.0, 1.0]
    assert np.allclose(transform(pts, T), [[0.0, 1.0, 1.0]])


def test_crop_keeps_only_the_workspace():
    pts = np.array([[0.0, 0.0, 0.2], [10.0, 0.0, 0.2], [0.0, 0.0, -1.0]])
    assert len(crop(pts, BOUNDS["lower"], BOUNDS["upper"])) == 1


def test_voxel_downsample_collapses_duplicates():
    pts = np.repeat(np.array([[0.001, 0.001, 0.001]]), 50, axis=0)
    assert len(voxel_downsample(pts, 0.01)) == 1


def test_farthest_point_sample_returns_exact_count():
    rng = np.random.default_rng(0)
    big = rng.normal(size=(500, 3)).astype(np.float32)
    small = rng.normal(size=(7, 3)).astype(np.float32)
    assert len(farthest_point_sample(big, 128, rng)) == 128
    assert len(farthest_point_sample(small, 128, rng)) == 128     # pads
    assert len(farthest_point_sample(np.zeros((0, 3)), 128, rng)) == 128


def test_farthest_point_sample_prefers_spread_over_density():
    # 200 points in a tight cluster, 2 far away: FPS must keep the far ones
    rng = np.random.default_rng(1)
    cluster = rng.normal(scale=0.001, size=(200, 3))
    far = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    pts = np.concatenate([cluster, far]).astype(np.float32)
    out = farthest_point_sample(pts, 8, rng)
    kept = sum(np.isclose(out, f, atol=1e-6).all(axis=-1).any() for f in far)
    assert kept == 2


def test_normalize_maps_bounds_to_unit_range():
    corners = np.array([BOUNDS["lower"], BOUNDS["upper"]], dtype=np.float32)
    out = normalize(corners, BOUNDS)
    assert np.allclose(out[0], -1.0) and np.allclose(out[1], 1.0)


def test_residual_correction_respects_its_bound():
    import torch

    from residual_adapt.policies.residual import ResidualActor

    max_delta = np.array([0.005] * 3 + [0.02] * 3 + [0.0], dtype=np.float32)
    actor = ResidualActor(obs_dim=16, action_dim=7, max_delta=max_delta)

    obs = torch.randn(256, 16) * 50        # deliberately extreme inputs
    delta, logp = actor(obs)
    assert delta.shape == (256, 7) and logp.shape == (256,)
    assert (delta.abs() <= torch.as_tensor(max_delta) + 1e-6).all()
    assert (delta[:, 6] == 0).all()        # gripper channel is pinned shut


def test_gate_hysteresis_holds_open_then_closes():
    from residual_adapt.policies.gate import AdvantageGate

    gate = AdvantageGate(min_open_steps=3)
    assert gate.step(1.0) == 1.0           # opens
    assert [gate.step(0.0) for _ in range(3)] == [1.0, 1.0, 1.0]   # holds
    assert gate.step(0.0) == 0.0           # then closes
