import numpy as np
import torch

from scenecast.data.scene import N_AGENTS, N_LANES, SAMPLE_SPEC, T_FUT, build_sample, resample_polyline, to_local, to_world
from scenecast.data.synthetic import make_scene
from scenecast.metrics import batch_metrics, constant_velocity, displacement, wta_loss
from scenecast.model import ModelConfig, SceneCast


def _batch(n=4, seed=0):
    rng = np.random.default_rng(seed)
    samples = [build_sample(make_scene(rng)) for _ in range(n)]
    return {k: torch.from_numpy(np.stack([s[k] for s in samples])) for k in SAMPLE_SPEC}


def test_local_world_roundtrip():
    rng = np.random.default_rng(0)
    xy, origin = rng.normal(size=(20, 2)), rng.normal(size=2)
    assert np.allclose(to_world(to_local(xy, origin, 0.7), origin, 0.7), xy)
    # a point 1 m ahead of an agent heading north lands on the local +x axis
    assert np.allclose(to_local(np.array([[0.0, 1.0]]), np.zeros(2), np.pi / 2), [[1.0, 0.0]])


def test_resample_is_evenly_spaced():
    out = resample_polyline(np.array([[0, 0], [1, 0], [10, 0]]), 11)
    assert np.allclose(out[:, 0], np.arange(11))


def test_sample_is_focal_centric():
    s = build_sample(make_scene(np.random.default_rng(1)))
    for k, (shape, dtype) in SAMPLE_SPEC.items():
        assert s[k].shape == shape and s[k].dtype == dtype, k
    last = s["agents"][0, -1]
    assert np.abs(last[:2]).max() < 1e-4  # focal sits at the origin...
    assert last[2] > 0.999 and last[4] > 0 and abs(last[5]) < 0.3  # ...facing and moving along +x
    assert s["lane_mask"].any()


def test_metrics_on_known_values():
    gt = torch.zeros(1, T_FUT, 2)
    pred = torch.zeros(1, 2, T_FUT, 2)
    pred[0, 0, :, 0] = 3.0  # mode 0 is 3 m off everywhere, mode 1 is perfect
    mask = torch.ones(1, T_FUT, dtype=torch.bool)
    ade, fde = displacement(pred, gt, mask)
    assert torch.allclose(ade, torch.tensor([[3.0, 0.0]])) and torch.allclose(fde, torch.tensor([[3.0, 0.0]]))
    m = batch_metrics(pred, torch.tensor([[5.0, -5.0]]), gt, mask)  # confident in the wrong mode
    assert m["minFDE"].item() == 0 and m["MR"].item() == 0
    assert abs(m["minFDE1"].item() - 3.0) < 1e-6 and m["brier-minFDE"].item() > 0.99


def test_fde_uses_last_valid_step():
    gt = torch.zeros(1, T_FUT, 2)
    pred = torch.zeros(1, 1, T_FUT, 2)
    pred[0, 0, 9, 0] = 4.0
    pred[0, 0, 10:, 0] = 100.0
    mask = torch.zeros(1, T_FUT, dtype=torch.bool)
    mask[0, :10] = True
    assert displacement(pred, gt, mask)[1].item() == 4.0


def test_model_shapes_and_gradients():
    batch = _batch()
    model = SceneCast(ModelConfig(d_model=32, n_heads=4, enc_layers=1, dec_layers=1))
    traj, logits = model(batch)
    assert traj.shape == (4, 6, T_FUT, 2) and logits.shape == (4, 6)
    loss, _ = wta_loss(traj, logits, batch["future"], batch["future_mask"])
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_ablations_ignore_hidden_inputs():
    batch = _batch()
    model = SceneCast(ModelConfig(d_model=32, n_heads=4, enc_layers=1, dec_layers=1, use_map=False, use_neighbors=False)).eval()
    out = model(batch)[0]
    changed = dict(batch)
    changed["lanes"] = batch["lanes"] + 5.0
    changed["agents"] = batch["agents"].clone()
    changed["agents"][:, 1:] += 5.0
    assert torch.allclose(out, model(changed)[0], atol=1e-5)
    assert batch["agents"].shape[1] == N_AGENTS and batch["lanes"].shape[1] == N_LANES


def test_constant_velocity_baseline():
    batch = _batch()
    pred, _ = constant_velocity(batch)
    assert pred.shape == (4, 1, T_FUT, 2)
    assert torch.allclose(pred[:, 0, 9], batch["agents"][:, 0, -1, 4:6], atol=1e-5)  # 1 s ahead = velocity
