"""Evaluate a checkpoint against the constant-velocity baseline.

    python -m scenecast.evaluate --data data/processed/av2/val --ckpt runs/av2-base/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data.store import SceneDataset
from .metrics import constant_velocity
from .model import ModelConfig, SceneCast
from .train import evaluate, pick_device


def load_model(ckpt: Path, device: torch.device) -> SceneCast:
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model = SceneCast(ModelConfig(**state["config"])).to(device)
    model.load_state_dict(state["model"])
    return model.eval()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="a processed split, e.g. data/processed/av2/val")
    ap.add_argument("--ckpt", type=Path, nargs="*", default=[])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path, default=None, help="write results as JSON")
    args = ap.parse_args()

    device = pick_device(args.device)
    loader = DataLoader(SceneDataset(args.data), args.batch_size)
    results = {"constant velocity": evaluate(constant_velocity, loader, device)}
    for ckpt in args.ckpt:
        results[ckpt.parent.name] = evaluate(load_model(ckpt, device), loader, device)

    cols = ["minADE", "minFDE", "MR", "brier-minFDE", "minADE1", "minFDE1"]
    print(f"| {'model':<20} | " + " | ".join(f"{c:>12}" for c in cols) + " |")
    print(f"|{'-' * 22}|" + "|".join("-" * 14 for _ in cols) + "|")
    for name, m in results.items():
        print(f"| {name:<20} | " + " | ".join(f"{m[c]:>12.3f}" for c in cols) + " |")
    if args.out:
        args.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
