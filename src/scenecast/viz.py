"""Render predictions as a PNG grid or an animated GIF.

    python -m scenecast.viz --data data/processed/av2/val --ckpt runs/av2-base/best.pt --out assets/grid.png
    python -m scenecast.viz --data data/processed/av2/val --ckpt runs/av2-base/best.pt --out assets/demo.gif --index 12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from torch.utils.data import DataLoader

from .data.scene import T_FUT
from .data.store import SceneDataset
from .evaluate import load_model
from .metrics import batch_metrics
from .train import pick_device, to_device

BG, LANE, OTHER, FOCAL, TRUTH = "#0f1419", "#3a4250", "#7f8a99", "#ffffff", "#35d07f"
PRED, PRED_TEXT = "#ff8a3d", "#ffb27d"
MIN_VIEW = 40.0  # metres, so slow scenes are not zoomed in absurdly far


def predict(model, sample: dict, device) -> tuple[np.ndarray, np.ndarray]:
    batch = to_device({k: v.unsqueeze(0) for k, v in sample.items()}, device)
    with torch.no_grad():
        traj, logits = model(batch)
    return traj[0].cpu().numpy(), logits[0].softmax(-1).cpu().numpy()


def draw(ax, sample: dict, traj: np.ndarray, prob: np.ndarray, upto: int = T_FUT, title: str = "") -> None:
    """Draw one scene in the focal frame. `upto` = how many future steps of ground truth to reveal."""
    ax.clear()
    ax.set_facecolor(BG)
    for lane, ok in zip(sample["lanes"].numpy(), sample["lane_mask"].numpy()):
        if ok:
            ax.plot(lane[:, 0], lane[:, 1], color=LANE, lw=1.0, zorder=1)
    agents, mask = sample["agents"].numpy(), sample["agent_mask"].numpy()
    for i in range(1, len(agents)):
        if mask[i].any():
            xy = agents[i][mask[i], :2]
            ax.plot(xy[:, 0], xy[:, 1], color=OTHER, lw=1.2, alpha=0.7, zorder=2)
            ax.scatter(*xy[-1], color=OTHER, s=14, zorder=2)
    for k in np.argsort(prob):  # most likely mode drawn last, on top
        alpha = 0.25 + 0.75 * prob[k] / prob.max()
        ax.plot(traj[k, :, 0], traj[k, :, 1], color=PRED, lw=2.0, alpha=alpha, zorder=4)
        ax.scatter(*traj[k, -1], color=PRED, s=18, alpha=alpha, zorder=4)
        ax.annotate(f"{prob[k]:.0%}", traj[k, -1], color=PRED_TEXT, fontsize=7, xytext=(3, 3), textcoords="offset points")
    hist = agents[0][mask[0], :2]
    ax.plot(hist[:, 0], hist[:, 1], color=FOCAL, lw=2.2, zorder=5)
    fut = sample["future"].numpy()[sample["future_mask"].numpy()][:upto]
    if len(fut):
        ax.plot(fut[:, 0], fut[:, 1], color=TRUTH, lw=2.2, zorder=6)
        ax.scatter(*fut[-1], color=TRUTH, s=28, zorder=7)
    else:
        ax.scatter(0, 0, color=FOCAL, s=28, zorder=7)
    # frame the action: history, every predicted mode and the true future, with a margin
    pts = [hist, traj.reshape(-1, 2)]
    if len(fut):
        pts.append(fut)
    extent = np.concatenate(pts)
    cx, cy = (extent.max(0) + extent.min(0)) / 2
    half = max(MIN_VIEW, (extent.max(0) - extent.min(0)).max() * 0.62)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, color="#c6ccd6", fontsize=9, pad=4)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help=".png for a 3x3 grid, .gif for an animation")
    ap.add_argument("--index", type=int, default=0, help="first scene index")
    ap.add_argument("--worst", action="store_true", help="show the highest-minFDE scenes instead (failure analysis)")
    ap.add_argument("--scan", type=int, default=2000, help="scenes to score when picking the worst")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    model = load_model(args.ckpt, device)
    data = SceneDataset(args.data)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    order, scores = None, None
    if args.worst:
        loader = DataLoader(SceneDataset(args.data, args.scan), 128)
        fde = []
        for batch in loader:
            batch = to_device(batch, device)
            with torch.no_grad():
                pred, logits = model(batch)
            fde.append(batch_metrics(pred, logits, batch["future"], batch["future_mask"])["minFDE"].cpu())
        scores = torch.cat(fde)
        order = scores.argsort(descending=True).tolist()

    def pick(i: int) -> tuple[dict, str]:
        if order is not None:
            idx = order[i % len(order)]
            return data[idx], f"minFDE {scores[idx]:.1f} m"
        return data[(args.index + i) % len(data)], ""

    if args.out.suffix == ".gif":
        sample, title = pick(args.index if order is None else 0)
        traj, prob = predict(model, sample, device)
        fig, ax = plt.subplots(figsize=(6, 6), facecolor=BG)
        fig.subplots_adjust(0, 0, 1, 1)
        frames = [0] * 6 + list(range(2, T_FUT + 1, 2)) + [T_FUT] * 10
        anim = FuncAnimation(fig, lambda f: draw(ax, sample, traj, prob, f, title), frames=frames)
        anim.save(args.out, writer=PillowWriter(fps=12))
    else:
        fig, axes = plt.subplots(3, 3, figsize=(12, 12), facecolor=BG)
        fig.subplots_adjust(0.01, 0.01, 0.97, 0.97, 0.02, 0.06)
        for i, ax in enumerate(axes.flat):
            sample, title = pick(i)
            draw(ax, sample, *predict(model, sample, device), title=title)
        fig.savefig(args.out, dpi=110, facecolor=BG)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
