"""Synthetic four-way-intersection scenes.

Used to test the whole pipeline without downloading a dataset. The focal car
approaches an intersection and goes left, straight or right, so the future is
genuinely multimodal and the map carries real information.

    python -m scenecast.data.synthetic --out data/processed/synth --train 8000 --val 1000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .scene import T_OBS, T_TOTAL, RawScene, build_sample, resample_polyline
from .store import SplitWriter

LANE_HALF = 1.75
BOX = 9.0  # half-size of the intersection box
ARM = 120.0  # arm length
DT = 0.1
ARMS = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([-1.0, 0.0]), np.array([0.0, -1.0])]


def _right(u: np.ndarray) -> np.ndarray:
    return np.array([u[1], -u[0]])


def _dense(points: np.ndarray, step: float = 0.5) -> np.ndarray:
    length = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
    return resample_polyline(points, max(2, int(length / step) + 1))


def _connector(p0, u0, p1, u1) -> np.ndarray:
    """Quadratic Bezier leaving p0 along u0 and arriving at p1 along u1."""
    if abs(u0 @ u1) > 0.99:
        ctrl = (p0 + p1) / 2
    else:
        a = np.linalg.solve(np.stack([u0, u1], axis=1), p1 - p0)
        ctrl = p0 + a[0] * u0
    t = np.linspace(0, 1, 40)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * ctrl + t**2 * p1


def _build_network():
    incoming, outgoing = [], []
    for d in ARMS:
        r_in, r_out = _right(-d), _right(d)
        incoming.append(_dense(np.stack([d * ARM + r_in * LANE_HALF, d * BOX + r_in * LANE_HALF])))
        outgoing.append(_dense(np.stack([d * BOX + r_out * LANE_HALF, d * ARM + r_out * LANE_HALF])))
    routes, connectors = [], []
    for i, d_in in enumerate(ARMS):
        for j, d_out in enumerate(ARMS):
            if i == j:
                continue
            conn = _dense(_connector(incoming[i][-1], -d_in, outgoing[j][0], d_out))
            connectors.append(conn)
            route = np.concatenate([incoming[i], conn[1:], outgoing[j][1:]])
            turn = float(d_in[1] * d_out[0] - d_in[0] * d_out[1])  # >0 left, <0 right, 0 straight
            routes.append((route, ARM - BOX, turn))
    lanes, attr = [], []
    for poly in incoming + outgoing:
        n_seg = 4
        idx = np.linspace(0, len(poly) - 1, n_seg + 1).astype(int)
        for a, b in zip(idx[:-1], idx[1:]):
            lanes.append(poly[a : b + 1])
            attr.append(0)
    for conn in connectors:
        lanes.append(conn)
        attr.append(1)
    return routes, lanes, attr


_ROUTES, _LANES, _LANE_ATTR = _build_network()


def _drive(route: np.ndarray, s_ref: float, speed: float, accel: float, rng) -> tuple:
    """Position, heading and velocity along `route`, with arclength s_ref at the last history step."""
    seg = np.linalg.norm(np.diff(route, axis=0), axis=1)
    s_axis = np.concatenate([[0.0], np.cumsum(seg)])
    t = (np.arange(T_TOTAL) - (T_OBS - 1)) * DT
    v = np.clip(speed + accel * t, 0.5, None)
    s = np.concatenate([[0.0], np.cumsum((v[1:] + v[:-1]) / 2 * DT)])
    s = s - s[T_OBS - 1] + s_ref
    inside = (s >= 0) & (s <= s_axis[-1])
    sc = np.clip(s, 0, s_axis[-1])
    pos = np.stack([np.interp(sc, s_axis, route[:, 0]), np.interp(sc, s_axis, route[:, 1])], axis=1)
    ahead = np.clip(sc + 0.5, 0, s_axis[-1])
    behind = np.clip(sc - 0.5, 0, s_axis[-1])
    tang = np.stack(
        [
            np.interp(ahead, s_axis, route[:, 0]) - np.interp(behind, s_axis, route[:, 0]),
            np.interp(ahead, s_axis, route[:, 1]) - np.interp(behind, s_axis, route[:, 1]),
        ],
        axis=1,
    )
    tang /= np.linalg.norm(tang, axis=1, keepdims=True).clip(1e-6)
    heading = np.arctan2(tang[:, 1], tang[:, 0])
    pos = pos + rng.normal(0, 0.03, pos.shape)
    return pos, heading, tang * v[:, None], inside


def make_scene(rng: np.random.Generator) -> RawScene:
    n_other = int(rng.integers(3, 12))
    pos = np.zeros((1 + n_other, T_TOTAL, 2))
    head = np.zeros((1 + n_other, T_TOTAL))
    vel = np.zeros((1 + n_other, T_TOTAL, 2))
    valid = np.zeros((1 + n_other, T_TOTAL), dtype=bool)

    # focal: 5-40 m before the intersection at the prediction time
    route, len_in, turn = _ROUTES[rng.integers(len(_ROUTES))]
    speed = rng.uniform(4, 9) if turn != 0 else rng.uniform(6, 13)
    s_ref = len_in - rng.uniform(5, 40)
    pos[0], head[0], vel[0], valid[0] = _drive(route, s_ref, speed, rng.uniform(-0.3, 0.3), rng)
    valid[0] = True

    for k in range(1, 1 + n_other):
        r, _, _ = _ROUTES[rng.integers(len(_ROUTES))]
        total = np.linalg.norm(np.diff(r, axis=0), axis=1).sum()
        pos[k], head[k], vel[k], inside = _drive(
            r, rng.uniform(20, total - 20), rng.uniform(3, 13), rng.uniform(-0.5, 0.5), rng
        )
        valid[k] = inside
        if rng.random() < 0.3:  # late-appearing track
            valid[k, : rng.integers(1, T_OBS - 5)] = False

    # random rigid transform so nothing is axis-aligned in world coordinates
    ang = rng.uniform(-np.pi, np.pi)
    shift = rng.uniform(-500, 500, 2)
    c, s = np.cos(ang), np.sin(ang)
    rot = np.array([[c, s], [-s, c]])
    return RawScene(
        positions=pos @ rot + shift,
        headings=head + ang,
        velocities=vel @ rot,
        valid=valid,
        types=np.zeros(1 + n_other, dtype=np.int64),
        focal=0,
        lanes=[l @ rot + shift for l in _LANES],
        lane_attr=list(_LANE_ATTR),
    )


def write_split(out_dir: Path, n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    writer = SplitWriter(out_dir, n)
    for i in tqdm(range(n), desc=out_dir.name):
        sample = build_sample(make_scene(rng))
        assert sample is not None
        writer.add(sample, f"synth-{seed}-{i}")
    writer.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/processed/synth"))
    ap.add_argument("--train", type=int, default=8000)
    ap.add_argument("--val", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    write_split(args.out / "train", args.train, args.seed)
    write_split(args.out / "val", args.val, args.seed + 1)


if __name__ == "__main__":
    main()
