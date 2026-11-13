"""On-disk format: one directory per split, one memory-mapped .npy file per field."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .scene import SAMPLE_SPEC


class SplitWriter:
    """Writes samples into preallocated memmaps so large splits never sit in RAM."""

    def __init__(self, out_dir: str | Path, capacity: int):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.capacity = capacity
        self.count = 0
        self.ids: list[str] = []
        self.arrays = {
            k: np.lib.format.open_memmap(
                self.dir / f"{k}.tmp.npy", mode="w+", dtype=dtype, shape=(capacity, *shape)
            )
            for k, (shape, dtype) in SAMPLE_SPEC.items()
        }

    def add(self, sample: dict[str, np.ndarray], scenario_id: str = "") -> None:
        for k, arr in self.arrays.items():
            arr[self.count] = sample[k]
        self.ids.append(scenario_id)
        self.count += 1

    def close(self) -> None:
        for k, arr in self.arrays.items():
            arr.flush()
            final = np.lib.format.open_memmap(
                self.dir / f"{k}.npy", mode="w+", dtype=arr.dtype, shape=(self.count, *arr.shape[1:])
            )
            final[:] = arr[: self.count]
            final.flush()
            del final
            (self.dir / f"{k}.tmp.npy").unlink()
        self.arrays = {}
        (self.dir / "meta.json").write_text(json.dumps({"count": self.count, "ids": self.ids}))


class SceneDataset(Dataset):
    def __init__(self, split_dir: str | Path, limit: int | None = None):
        self.dir = Path(split_dir)
        if not (self.dir / "meta.json").exists():
            raise FileNotFoundError(f"{self.dir} is not a processed split (no meta.json)")
        meta = json.loads((self.dir / "meta.json").read_text())
        self.ids: list[str] = meta["ids"]
        self.arrays = {k: np.load(self.dir / f"{k}.npy", mmap_mode="r") for k in SAMPLE_SPEC}
        self.n = meta["count"] if limit is None else min(limit, meta["count"])

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {k: torch.from_numpy(np.array(a[i])) for k, a in self.arrays.items()}
