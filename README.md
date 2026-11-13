# SceneCast

Multimodal motion forecasting for driving scenes. Given 5 seconds of history for
every agent in a scene plus the lane map, SceneCast predicts 6 candidate
trajectories for the next 6 seconds of a focal agent, each with a confidence.

It is a compact (about 1.6M parameter) vectorized scene transformer written from
scratch in PyTorch, trained and evaluated on the
[Argoverse 2 Motion Forecasting](https://www.argoverse.org/av2.html) dataset. It
trains on a laptop GPU (Apple Silicon MPS) or a single CUDA GPU.

<!-- After training, generate these with `python -m scenecast.viz` and uncomment:
![Predictions on Argoverse 2 validation scenes](assets/grid.png)
-->

## How it works

```
agent tracks  [32 x 50 steps] --> polyline encoder --+
                                                     +--> scene transformer --> 6 mode queries --> 6 trajectories
lane centerlines [96 x 10 pts] --> polyline encoder --+        (4 layers)        (2-layer decoder)   + confidences
```

1. **Focal-centric input.** Every scene is translated and rotated so the focal
   agent sits at the origin facing +x at the prediction time. The model never has
   to learn rotation invariance.
2. **Polyline tokens.** Each agent track and each lane centerline is a polyline.
   A small PointNet-style encoder (point MLP, max-pool, concat, MLP, max-pool)
   turns each one into a single token, in the spirit of VectorNet.
3. **Scene transformer.** Agent and lane tokens attend to each other, so the focal
   agent's token absorbs who is around it and where the road goes.
4. **Mode queries.** Six learned queries cross-attend to the scene and each
   decodes one future as per-step displacements, plus a confidence logit.
5. **Winner-takes-all loss.** Only the mode closest to the ground truth is
   regressed (smooth L1), and a cross-entropy term teaches the confidence head to
   pick that mode. This is what stops the modes collapsing to their average.

## Results

Argoverse 2 validation, 6 modes, 6 second horizon. Lower is better everywhere.

| Model | minADE | minFDE | Miss rate | brier-minFDE |
|---|---|---|---|---|
| Constant velocity (1 mode) | _todo_ | _todo_ | _todo_ | _todo_ |
| SceneCast | _todo_ | _todo_ | _todo_ | _todo_ |
| SceneCast, no map | _todo_ | _todo_ | _todo_ | _todo_ |
| SceneCast, no neighbours | _todo_ | _todo_ | _todo_ | _todo_ |
| SceneCast, 1 mode | _todo_ | _todo_ | _todo_ | _todo_ |

Fill this table from `python -m scenecast.evaluate` (see below), and state how
many training scenarios were used.

- **minADE / minFDE**: average / final displacement error in metres of the best of the 6 modes.
- **Miss rate**: fraction of scenes where the best mode ends more than 2 m from the truth.
- **brier-minFDE**: minFDE plus a penalty for giving the best mode low confidence.

## A bug worth remembering: modes that only differ by dropout

The first version looked fine in training and fell apart in evaluation: on the
synthetic intersection data, minFDE was about 1.0 m with the model in train mode
and about 10 m in eval mode, with all six modes drawn on top of each other.

The cause was the decoder. The mode queries were initialised almost identically
(std 0.02) and the decoder used dropout, so during training the six outputs
differed mainly through dropout noise. Winner-takes-all training happily used
that noise as its source of diversity. With dropout off at eval time, the modes
collapsed into one. Initialising the queries at unit scale and removing dropout
from the decoder fixed it:

| Synthetic intersections (800 val scenes) | minADE | minFDE | Miss rate |
|---|---|---|---|
| Constant velocity | 3.35 | 12.05 | 0.83 |
| SceneCast, before the fix | 2.90 | 9.82 | 0.61 |
| SceneCast, after the fix | 0.21 | 0.38 | 0.02 |

![Predictions on synthetic intersections](assets/synth.png)

White is the observed history, green the true future, orange the six predicted
modes with their confidences. These are synthetic scenes used as a pipeline
check, not a benchmark result.

## Quick start

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"
source .venv/bin/activate
pytest
```

### Smoke test without downloading anything

A synthetic four-way intersection generator exercises the whole pipeline. The
focal car turns left, goes straight or turns right, so the task is multimodal.

```bash
python -m scenecast.data.synthetic --out data/processed/synth
python -m scenecast.train --data data/processed/synth --name synth --epochs 20 --warmup-steps 100
python -m scenecast.evaluate --data data/processed/synth/val --ckpt runs/synth/best.pt
python -m scenecast.viz --data data/processed/synth/val --ckpt runs/synth/best.pt --out assets/synth.png
```

### Argoverse 2

The dataset is public and needs no sign-up. A scenario is about 250 KB, so
20,000 training scenarios is roughly 5 GB.

```bash
python scripts/download_av2.py --split train --num 20000
python scripts/download_av2.py --split val --num 3000

python -m scenecast.data.av2 --raw data/raw/av2/train --out data/processed/av2/train
python -m scenecast.data.av2 --raw data/raw/av2/val --out data/processed/av2/val

python -m scenecast.train --data data/processed/av2 --name base --epochs 30
```

Ablations reuse the same command with one flag each:

```bash
python -m scenecast.train --data data/processed/av2 --name no-map --no-map
python -m scenecast.train --data data/processed/av2 --name no-neighbors --no-neighbors
python -m scenecast.train --data data/processed/av2 --name one-mode --modes 1
```

Evaluate everything in one table, then render figures:

```bash
python -m scenecast.evaluate --data data/processed/av2/val --ckpt runs/base/best.pt runs/no-map/best.pt runs/no-neighbors/best.pt runs/one-mode/best.pt
python -m scenecast.viz --data data/processed/av2/val --ckpt runs/base/best.pt --out assets/grid.png
python -m scenecast.viz --data data/processed/av2/val --ckpt runs/base/best.pt --out assets/demo.gif --index 12
```

## Repository layout

| Path | What it does |
|---|---|
| `src/scenecast/data/scene.py` | Scene format, focal-frame normalization, fixed-size sample building |
| `src/scenecast/data/av2.py` | Argoverse 2 parser (pandas only, no `av2` dependency) |
| `src/scenecast/data/synthetic.py` | Synthetic intersection scenes |
| `src/scenecast/data/store.py` | Memory-mapped on-disk splits and the PyTorch dataset |
| `src/scenecast/model.py` | Polyline encoders, scene transformer, mode-query decoder |
| `src/scenecast/metrics.py` | Winner-takes-all loss, metrics, constant-velocity baseline |
| `src/scenecast/train.py`, `evaluate.py`, `viz.py` | Training loop, evaluation table, figures |
| `scripts/download_av2.py` | Subset downloader for the public Argoverse 2 bucket |

## Limitations and next steps

- Predicts one focal agent per scene (marginal prediction), not joint futures for interacting agents.
- Uses lane centerlines only: no crosswalks, lane boundaries or traffic-light state.
- No data augmentation or ensembling; a larger model and the full 200k training scenarios should close part of the gap to published methods.
- Port the data pipeline to the Waymo Open Motion Dataset, which adds traffic-light state and an 8 second horizon.

## References

- Gao et al., *VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation*, CVPR 2020
- Wilson et al., *Argoverse 2: Next Generation Datasets for Self-Driving Perception and Forecasting*, NeurIPS Datasets 2021
- Ettinger et al., *Large Scale Interactive Motion Forecasting for Autonomous Driving: The Waymo Open Motion Dataset*, ICCV 2021
