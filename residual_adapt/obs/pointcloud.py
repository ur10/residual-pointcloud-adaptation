"""Depth images to the point-cloud observation the policies consume.

The base policy was trained in simulation, where depth is clean and complete.
On the real robot it is not: the ring is thin and specular, the peg casts a
shadow into the stereo baseline, and extrinsics drift between sessions. This
module does the part of the gap that is fixable by preprocessing - cropping to
the workspace, removing obvious outliers, fusing the two camera views and
resampling to a fixed point count - so that whatever is left is the part the
residual policy has to absorb.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_matrix(cls, k: np.ndarray) -> "CameraIntrinsics":
        return cls(fx=float(k[0, 0]), fy=float(k[1, 1]),
                   cx=float(k[0, 2]), cy=float(k[1, 2]))


def deproject(depth: np.ndarray, intr: CameraIntrinsics,
              depth_trunc: float = 1.2) -> np.ndarray:
    """Depth image (metres, 0 = invalid) to an (N, 3) cloud in camera frame."""
    h, w = depth.shape
    v, u = np.indices((h, w), dtype=np.float32)
    valid = (depth > 1e-6) & (depth < depth_trunc) & np.isfinite(depth)

    z = depth[valid]
    x = (u[valid] - intr.cx) * z / intr.fx
    y = (v[valid] - intr.cy) * z / intr.fy
    return np.stack([x, y, z], axis=-1)


def transform(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to an (N, 3) cloud."""
    return points @ pose[:3, :3].T + pose[:3, 3]


def crop(points: np.ndarray, lower, upper) -> np.ndarray:
    lower = np.asarray(lower, dtype=np.float32)
    upper = np.asarray(upper, dtype=np.float32)
    keep = np.all((points >= lower) & (points <= upper), axis=-1)
    return points[keep]


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Keep one point per occupied voxel (centroid of the points inside it)."""
    if voxel_size <= 0 or len(points) == 0:
        return points
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)

    out = np.zeros((inverse.max() + 1, 3), dtype=np.float64)
    counts = np.zeros(inverse.max() + 1, dtype=np.int64)
    np.add.at(out, inverse, points)
    np.add.at(counts, inverse, 1)
    return (out / counts[:, None]).astype(np.float32)


def remove_statistical_outliers(points: np.ndarray, k: int = 16,
                                std_ratio: float = 2.0) -> np.ndarray:
    """Drop points whose mean distance to their k nearest neighbours is an
    outlier. Depth holes around the peg leave speckle that otherwise survives
    cropping and shifts the policy's notion of where the hole is."""
    if len(points) <= k:
        return points
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k + 1)      # column 0 is the point itself
    mean_d = dists[:, 1:].mean(axis=1)
    keep = mean_d < mean_d.mean() + std_ratio * mean_d.std()
    return points[keep]


def farthest_point_sample(points: np.ndarray, num: int,
                          rng: np.random.Generator | None = None) -> np.ndarray:
    """Resample to exactly `num` points, spread over the cloud.

    The policy expects a fixed-size input. Uniform random sampling biases
    toward the table, which dominates the point count; FPS keeps the ring and
    the peg, which is where the task actually is.
    """
    rng = rng or np.random.default_rng()
    n = len(points)
    if n == 0:
        return np.zeros((num, 3), dtype=np.float32)
    if n <= num:
        pad = rng.integers(0, n, size=num - n)
        return np.concatenate([points, points[pad]], axis=0)

    idx = np.empty(num, dtype=np.int64)
    idx[0] = rng.integers(0, n)
    dist = np.linalg.norm(points - points[idx[0]], axis=-1)
    for i in range(1, num):
        idx[i] = int(dist.argmax())
        dist = np.minimum(dist, np.linalg.norm(points - points[idx[i]], axis=-1))
    return points[idx]


def normalize(points: np.ndarray, bounds) -> np.ndarray:
    """Map workspace coordinates into [-1, 1], the range the base policy saw."""
    lower = np.asarray(bounds["lower"], dtype=np.float32)
    upper = np.asarray(bounds["upper"], dtype=np.float32)
    centre = 0.5 * (lower + upper)
    scale = 0.5 * (upper - lower)
    return (points - centre) / np.maximum(scale, 1e-6)


def build_observation(depths, intrinsics, extrinsics, cfg,
                      rng: np.random.Generator | None = None) -> np.ndarray:
    """Fuse every camera view into the fixed-size cloud the policy consumes.

    Args:
        depths: list of (H, W) depth images in metres, one per camera.
        intrinsics: list of CameraIntrinsics, one per camera.
        extrinsics: list of 4x4 camera-to-base transforms.
        cfg: the `observation` block of the task config.

    Returns:
        (num_points, 3) float32 array, normalized to [-1, 1].
    """
    clouds = []
    for depth, intr, pose in zip(depths, intrinsics, extrinsics):
        pts = deproject(depth, intr, cfg["depth_trunc"])
        clouds.append(transform(pts, pose))

    points = np.concatenate(clouds, axis=0) if clouds else np.zeros((0, 3))
    points = crop(points, cfg["workspace_bounds"]["lower"],
                  cfg["workspace_bounds"]["upper"])
    points = voxel_downsample(points, cfg.get("voxel_size", 0.0))
    if cfg.get("remove_outliers", False):
        points = remove_statistical_outliers(points)
    points = farthest_point_sample(points, cfg["num_points"], rng)
    return normalize(points, cfg["workspace_bounds"]).astype(np.float32)
