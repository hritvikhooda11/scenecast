"""Argoverse 2 Motion Forecasting -> processed splits.

Reads the raw scenario folders (one parquet of tracks + one JSON map each) with
pandas, so the heavy `av2` package is not needed.

    python -m scenecast.data.av2 --raw data/raw/av2/train --out data/processed/av2/train
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .scene import AGENT_TYPES, OTHER_TYPE, T_TOTAL, RawScene, build_sample, resample_polyline
from .store import SplitWriter


def _xy(points: list[dict]) -> np.ndarray:
    return np.array([[p["x"], p["y"]] for p in points], dtype=np.float64)


def load_map(path: Path) -> tuple[list[np.ndarray], list[int]]:
    data = json.loads(path.read_text())
    lanes, attr = [], []
    for seg in data.get("lane_segments", {}).values():
        if seg.get("centerline"):
            center = _xy(seg["centerline"])
        else:  # centerline not stored: average the two boundaries
            left = resample_polyline(_xy(seg["left_lane_boundary"]), 10)
            right = resample_polyline(_xy(seg["right_lane_boundary"]), 10)
            center = (left + right) / 2
        lanes.append(center)
        attr.append(int(bool(seg.get("is_intersection", False))))
    return lanes, attr


def load_scenario(folder: Path) -> RawScene:
    sid = folder.name
    df = pd.read_parquet(folder / f"scenario_{sid}.parquet")
    codes, track_ids = pd.factorize(df["track_id"])
    t = df["timestep"].to_numpy()
    keep = t < T_TOTAL
    codes, t = codes[keep], t[keep]
    df = df[keep]
    a = len(track_ids)

    pos = np.zeros((a, T_TOTAL, 2))
    head = np.zeros((a, T_TOTAL))
    vel = np.zeros((a, T_TOTAL, 2))
    valid = np.zeros((a, T_TOTAL), dtype=bool)
    pos[codes, t] = df[["position_x", "position_y"]].to_numpy()
    vel[codes, t] = df[["velocity_x", "velocity_y"]].to_numpy()
    head[codes, t] = df["heading"].to_numpy()
    valid[codes, t] = True

    first = pd.Series(df["object_type"].to_numpy()).groupby(codes).first()
    types = np.array([AGENT_TYPES.get(first[i], OTHER_TYPE) for i in range(a)], dtype=np.int64)
    focal = int(np.flatnonzero(track_ids == df["focal_track_id"].iloc[0])[0])

    lanes, attr = load_map(folder / f"log_map_archive_{sid}.json")
    return RawScene(pos, head, vel, valid, types, focal, lanes, attr, sid)


def _process(folder: Path):
    try:
        return folder.name, build_sample(load_scenario(folder))
    except Exception as exc:  # one corrupt scenario should not kill a multi-hour job
        return folder.name, exc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True, help="folder of raw scenario folders")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    folders = sorted(p for p in args.raw.iterdir() if p.is_dir())[: args.limit]
    if not folders:
        raise SystemExit(f"no scenario folders in {args.raw}")
    writer = SplitWriter(args.out, len(folders))
    skipped, failed = 0, []
    with Pool(args.workers) as pool:
        for sid, result in tqdm(pool.imap(_process, folders, chunksize=16), total=len(folders)):
            if isinstance(result, Exception):
                failed.append((sid, repr(result)))
            elif result is None:
                skipped += 1
            else:
                writer.add(result, sid)
    writer.close()
    print(f"wrote {writer.count} samples to {args.out} ({skipped} unusable, {len(failed)} failed)")
    for sid, err in failed[:10]:
        print(f"  failed {sid}: {err}")


if __name__ == "__main__":
    main()
