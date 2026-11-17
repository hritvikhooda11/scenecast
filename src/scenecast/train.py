"""Train SceneCast.

    python -m scenecast.train --data data/processed/av2 --name av2-base --epochs 30
    python -m scenecast.train --data data/processed/av2 --name av2-nomap --no-map   # ablation
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data.store import SceneDataset
from .metrics import batch_metrics, wta_loss
from .model import ModelConfig, SceneCast


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_device(batch: dict, device: torch.device) -> dict:
    # origin/theta are float64 bookkeeping for visualisation; MPS has no float64
    return {k: v.to(device) for k, v in batch.items() if k not in ("origin", "theta")}


@torch.no_grad()
def evaluate(model_fn, loader: DataLoader, device: torch.device) -> dict[str, float]:
    sums: dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = to_device(batch, device)
        pred, logits = model_fn(batch)
        for k, v in batch_metrics(pred, logits, batch["future"], batch["future_mask"]).items():
            sums[k] = sums.get(k, 0.0) + v.sum().item()
        count += len(pred)
    return {k: v / count for k, v in sums.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="folder with train/ and val/ splits")
    ap.add_argument("--name", default="run")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--limit-val", type=int, default=None)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--enc-layers", type=int, default=4)
    ap.add_argument("--dec-layers", type=int, default=2)
    ap.add_argument("--modes", type=int, default=6)
    ap.add_argument("--decoder-dropout", type=float, default=0.0, help="mode-collapse study only")
    ap.add_argument("--query-init-std", type=float, default=1.0, help="mode-collapse study only")
    ap.add_argument("--no-map", action="store_true", help="ablation: hide lane tokens")
    ap.add_argument("--no-neighbors", action="store_true", help="ablation: hide other agents")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    run_dir = Path("runs") / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = ModelConfig(
        d_model=args.d_model,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        n_modes=args.modes,
        use_map=not args.no_map,
        use_neighbors=not args.no_neighbors,
        decoder_dropout=args.decoder_dropout,
        query_init_std=args.query_init_std,
    )
    model = SceneCast(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    (run_dir / "config.json").write_text(json.dumps({"model": cfg.to_dict(), "args": vars(args) | {"data": str(args.data)}}, indent=2))

    loader_kw = dict(num_workers=args.workers, persistent_workers=args.workers > 0)
    train_loader = DataLoader(
        SceneDataset(args.data / "train", args.limit_train), args.batch_size, shuffle=True, drop_last=True, **loader_kw
    )
    val_loader = DataLoader(SceneDataset(args.data / "val", args.limit_val), args.batch_size * 2, **loader_kw)
    print(f"{n_params / 1e6:.2f}M params | {len(train_loader.dataset)} train / {len(val_loader.dataset)} val | {device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total = args.epochs * len(train_loader)

    def lr_at(step: int) -> float:
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / max(1, total - args.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    best = float("inf")
    log = (run_dir / "log.jsonl").open("a")

    for epoch in range(1, args.epochs + 1):
        model.train()
        start, running, seen = time.time(), 0.0, 0
        bar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for batch in bar:
            batch = to_device(batch, device)
            pred, logits = model(batch)
            loss, parts = wta_loss(pred, logits, batch["future"], batch["future_mask"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            running += loss.item()
            seen += 1
            bar.set_postfix(loss=f"{running / seen:.3f}", reg=f"{parts['reg']:.3f}")

        model.eval()
        metrics = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": running / seen, "seconds": round(time.time() - start, 1), **metrics}
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(
            f"epoch {epoch:3d} | loss {row['train_loss']:.3f} | minADE {metrics['minADE']:.3f} "
            f"| minFDE {metrics['minFDE']:.3f} | MR {metrics['MR']:.3f} | {row['seconds']}s"
        )
        state = {"model": model.state_dict(), "config": cfg.to_dict(), "epoch": epoch, "metrics": metrics}
        torch.save(state, run_dir / "last.pt")
        if metrics["minFDE"] < best:
            best = metrics["minFDE"]
            torch.save(state, run_dir / "best.pt")


if __name__ == "__main__":
    main()
