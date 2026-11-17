"""Mode-collapse study: does the decoder use dropout noise as its source of mode diversity?

Trains nothing. For every run named runs/collapse-std{S}-dd{D}-s{seed} it evaluates the final
checkpoint twice on the synthetic validation split, once in eval mode (dropout off) and once in
train mode (dropout on), and reports minFDE plus how far apart the six predicted endpoints are.

    for seed in 0 1; do for cfg in "0.02 0.1" "0.02 0.0" "1.0 0.1" "1.0 0.0"; do
        set -- $cfg
        python -m scenecast.train --data data/processed/synth --name collapse-std$1-dd$2-s$seed \
            --epochs 20 --warmup-steps 100 --seed $seed --query-init-std $1 --decoder-dropout $2
    done; done
    python scripts/collapse_study.py --out results/collapse_study.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from scenecast.data.store import SceneDataset
from scenecast.evaluate import load_model
from scenecast.metrics import batch_metrics

PATTERN = re.compile(r"collapse-std([\d.]+)-dd([\d.]+)-s(\d+)")


@torch.no_grad()
def score(model, loader) -> dict[str, float]:
    sums = {"minFDE": 0.0, "MR": 0.0, "spread": 0.0}
    count = 0
    for batch in loader:
        batch = {k: v for k, v in batch.items() if k not in ("origin", "theta")}
        pred, logits = model(batch)
        m = batch_metrics(pred, logits, batch["future"], batch["future_mask"])
        ends = pred[:, :, -1]  # [B, K, 2]
        k = ends.shape[1]
        pairwise = torch.cdist(ends, ends).sum(dim=(1, 2)) / (k * (k - 1))  # mean endpoint distance between modes
        sums["minFDE"] += m["minFDE"].sum().item()
        sums["MR"] += m["MR"].sum().item()
        sums["spread"] += pairwise.sum().item()
        count += len(pred)
    return {k: v / count for k, v in sums.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--data", type=Path, default=Path("data/processed/synth/val"))
    ap.add_argument("--out", type=Path, default=Path("results/collapse_study.json"))
    args = ap.parse_args()

    device = torch.device("cpu")  # MPS has no dropout inside attention under no_grad
    loader = DataLoader(SceneDataset(args.data), 200)
    rows = []
    for run in sorted(args.runs.glob("collapse-*")):
        match = PATTERN.fullmatch(run.name)
        if not match or not (run / "last.pt").exists():
            continue
        model = load_model(run / "last.pt", device)
        torch.manual_seed(0)
        row = {
            "query_init_std": float(match[1]),
            "decoder_dropout": float(match[2]),
            "seed": int(match[3]),
            "eval_mode": score(model.eval(), loader),
            "train_mode": score(model.train(), loader),
        }
        rows.append(row)
        e, t = row["eval_mode"], row["train_mode"]
        print(
            f"std {row['query_init_std']:<5} dropout {row['decoder_dropout']:<4} seed {row['seed']} | "
            f"eval minFDE {e['minFDE']:6.2f} MR {e['MR']:.2f} spread {e['spread']:5.1f} | "
            f"train-mode minFDE {t['minFDE']:6.2f} spread {t['spread']:5.1f}"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
