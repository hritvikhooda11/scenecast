"""Scene representation shared by every data source.

A data source (Argoverse 2, the synthetic generator, ...) produces a `RawScene`
in world coordinates. `build_sample` turns it into fixed-size arrays in the
focal agent's frame, which is what the model consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

T_OBS = 50  # 5 s of history at 10 Hz
T_FUT = 60  # 6 s of future at 10 Hz
T_TOTAL = T_OBS + T_FUT
N_AGENTS = 32  # focal agent + 31 nearest neighbours
N_LANES = 96
LANE_POINTS = 10
LANE_RADIUS = 120.0  # metres around the focal agent
AGENT_FEATS = 6  # x, y, cos(heading), sin(heading), vx, vy

# The Argoverse 2 object taxonomy. Static / background / construction objects are kept
# (a cone or a parked truck is real context) but get their own embeddings so the model
# can learn that they do not move.
AGENT_TYPES = {
    "vehicle": 0,
    "pedestrian": 1,
    "motorcyclist": 2,
    "cyclist": 3,
    "bus": 4,
    "static": 5,
    "background": 6,
    "construction": 7,
    "riderless_bicycle": 8,
}
OTHER_TYPE = 9
N_AGENT_TYPES = 10


@dataclass
class RawScene:
    """One scenario in world coordinates. A = number of tracks."""

    positions: np.ndarray  # [A, T_TOTAL, 2]
    headings: np.ndarray  # [A, T_TOTAL]
    velocities: np.ndarray  # [A, T_TOTAL, 2]
    valid: np.ndarray  # [A, T_TOTAL] bool
    types: np.ndarray  # [A] int
    focal: int
    lanes: list[np.ndarray] = field(default_factory=list)  # each [P_i, 2]
    lane_attr: list[int] = field(default_factory=list)  # 1 if inside an intersection
    scenario_id: str = ""


def to_local(xy: np.ndarray, origin: np.ndarray, theta: float) -> np.ndarray:
    """World -> frame centred at `origin` with +x along heading `theta`."""
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    return (xy - origin) @ rot


def to_world(xy: np.ndarray, origin: np.ndarray, theta: float) -> np.ndarray:
    """Inverse of `to_local`."""
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, s], [-s, c]])
    return xy @ rot + origin


def resample_polyline(xy: np.ndarray, n: int) -> np.ndarray:
    """Resample a polyline to `n` points evenly spaced by arc length."""
    xy = np.asarray(xy, dtype=np.float64)
    if len(xy) == 1:
        return np.repeat(xy, n, axis=0)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-6:
        return np.repeat(xy[:1], n, axis=0)
    target = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(target, s, xy[:, 0]), np.interp(target, s, xy[:, 1])], axis=1)


def build_sample(scene: RawScene) -> dict[str, np.ndarray] | None:
    """Convert a RawScene into model inputs. Returns None if the scene is unusable."""
    f = scene.focal
    last = T_OBS - 1
    fut_valid = scene.valid[f, T_OBS:]
    if not scene.valid[f, last] or not fut_valid.any():
        return None

    origin = scene.positions[f, last].astype(np.float64)
    theta = float(scene.headings[f, last])

    # --- agents: focal first, then nearest neighbours observed at the last history step
    present = scene.valid[:, last].copy()
    present[f] = False
    cand = np.flatnonzero(present)
    dist = np.linalg.norm(scene.positions[cand, last] - origin, axis=1)
    order = np.concatenate([[f], cand[np.argsort(dist)][: N_AGENTS - 1]]).astype(int)

    n = len(order)
    agents = np.zeros((N_AGENTS, T_OBS, AGENT_FEATS), dtype=np.float32)
    agent_mask = np.zeros((N_AGENTS, T_OBS), dtype=bool)
    agent_type = np.zeros((N_AGENTS,), dtype=np.int64)

    pos = to_local(scene.positions[order, :T_OBS], origin, theta)
    vel = to_local(scene.velocities[order, :T_OBS], np.zeros(2), theta)
    head = scene.headings[order, :T_OBS] - theta
    mask = scene.valid[order, :T_OBS]
    feats = np.concatenate([pos, np.cos(head)[..., None], np.sin(head)[..., None], vel], axis=-1)
    agents[:n] = np.where(mask[..., None], feats, 0.0)
    agent_mask[:n] = mask
    agent_type[:n] = scene.types[order]

    # --- lanes: nearest centerlines, resampled to a fixed number of points
    lanes = np.zeros((N_LANES, LANE_POINTS, 2), dtype=np.float32)
    lane_mask = np.zeros((N_LANES,), dtype=bool)
    lane_attr = np.zeros((N_LANES,), dtype=np.int64)
    if scene.lanes:
        local = [to_local(resample_polyline(l, LANE_POINTS), origin, theta) for l in scene.lanes]
        local = np.stack(local)
        d = np.linalg.norm(local, axis=-1).min(axis=1)
        keep = np.argsort(d)[:N_LANES]
        keep = keep[d[keep] <= LANE_RADIUS]
        lanes[: len(keep)] = local[keep]
        lane_mask[: len(keep)] = True
        lane_attr[: len(keep)] = np.asarray(scene.lane_attr, dtype=np.int64)[keep]

    future = to_local(scene.positions[f, T_OBS:], origin, theta).astype(np.float32)
    future = np.where(fut_valid[:, None], future, 0.0)

    return {
        "agents": agents,
        "agent_mask": agent_mask,
        "agent_type": agent_type,
        "lanes": lanes,
        "lane_mask": lane_mask,
        "lane_attr": lane_attr,
        "future": future,
        "future_mask": fut_valid.astype(bool),
        "origin": origin.astype(np.float64),
        "theta": np.float64(theta),
    }


SAMPLE_SPEC: dict[str, tuple[tuple[int, ...], str]] = {
    "agents": ((N_AGENTS, T_OBS, AGENT_FEATS), "float32"),
    "agent_mask": ((N_AGENTS, T_OBS), "bool"),
    "agent_type": ((N_AGENTS,), "int64"),
    "lanes": ((N_LANES, LANE_POINTS, 2), "float32"),
    "lane_mask": ((N_LANES,), "bool"),
    "lane_attr": ((N_LANES,), "int64"),
    "future": ((T_FUT, 2), "float32"),
    "future_mask": ((T_FUT,), "bool"),
    "origin": ((2,), "float64"),
    "theta": ((), "float64"),
}
