"""Plot validation curves for one or more runs.

    python -m scenecast.curves --runs runs/base runs/no-map runs/no-neighbors --out assets/curves.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, FG, GRID = "#0f1419", "#c6ccd6", "#2a3140"
PALETTE = ["#35d07f", "#ff8a3d", "#4da6ff", "#c084fc", "#ffd166", "#ff6b6b"]
METRICS = [("minADE", "minADE (m)"), ("minFDE", "minFDE (m)"), ("MR", "miss rate")]


def read_log(run: Path) -> list[dict]:
    log = run / "log.jsonl"
    if not log.exists():
        raise FileNotFoundError(f"no log.jsonl in {run}")
    return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("assets/curves.png"))
    args = ap.parse_args()

    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.6 * len(METRICS), 3.8), facecolor=BG)
    for ax, (key, label) in zip(axes, METRICS):
        ax.set_facecolor(BG)
        for i, run in enumerate(args.runs):
            rows = read_log(run)
            ax.plot(
                [r["epoch"] for r in rows],
                [r[key] for r in rows],
                color=PALETTE[i % len(PALETTE)],
                lw=1.8,
                label=run.name,
            )
        ax.set_xlabel("epoch", color=FG, fontsize=9)
        ax.set_ylabel(label, color=FG, fontsize=9)
        ax.tick_params(colors=FG, labelsize=8)
        ax.grid(color=GRID, lw=0.6, alpha=0.6)
        for spine in ax.spines.values():
            spine.set_color(GRID)
    axes[0].legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=8)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, facecolor=BG)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
