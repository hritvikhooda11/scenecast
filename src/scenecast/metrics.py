"""Loss, standard motion-forecasting metrics and the constant-velocity baseline."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .data.scene import T_FUT

MISS_THRESHOLD = 2.0  # metres, as in the Argoverse and Waymo benchmarks


def displacement(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-mode ADE and FDE. pred [B, K, T, 2], gt [B, T, 2], mask [B, T] -> two [B, K] tensors."""
    err = torch.linalg.norm(pred - gt.unsqueeze(1), dim=-1)
    m = mask.unsqueeze(1).float()
    ade = (err * m).sum(-1) / m.sum(-1).clamp(min=1)
    last = (mask.float() * torch.arange(1, mask.shape[1] + 1, device=mask.device)).argmax(dim=1)
    fde = err.gather(2, last.view(-1, 1, 1).expand(-1, err.shape[1], 1)).squeeze(-1)
    return ade, fde


def wta_loss(pred, logits, gt, mask) -> tuple[torch.Tensor, dict[str, float]]:
    """Winner-takes-all: regress only the mode closest to the truth, and classify which mode that was."""
    with torch.no_grad():
        ade, _ = displacement(pred, gt, mask)
        best = ade.argmin(dim=1)
    chosen = pred[torch.arange(len(pred), device=pred.device), best]
    reg = F.smooth_l1_loss(chosen, gt, reduction="none").sum(-1)
    reg = (reg * mask).sum() / mask.sum().clamp(min=1)
    cls = F.cross_entropy(logits, best)
    return reg + cls, {"reg": reg.item(), "cls": cls.item()}


@torch.no_grad()
def batch_metrics(pred, logits, gt, mask) -> dict[str, torch.Tensor]:
    """Per-sample metrics [B]; average them over the dataset."""
    ade, fde = displacement(pred, gt, mask)
    prob = logits.softmax(dim=1)
    best = fde.argmin(dim=1, keepdim=True)
    top = prob.argmax(dim=1, keepdim=True)
    min_fde = fde.gather(1, best).squeeze(1)
    return {
        "minADE": ade.min(dim=1).values,
        "minFDE": min_fde,
        "MR": (min_fde > MISS_THRESHOLD).float(),
        "brier-minFDE": min_fde + (1 - prob.gather(1, best).squeeze(1)) ** 2,
        "minADE1": ade.gather(1, top).squeeze(1),
        "minFDE1": fde.gather(1, top).squeeze(1),
    }


def constant_velocity(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Baseline: the focal agent keeps its last observed velocity. One mode."""
    vel = batch["agents"][:, 0, -1, 4:6]
    t = torch.arange(1, T_FUT + 1, device=vel.device).view(1, -1, 1) * 0.1
    pred = (vel.unsqueeze(1) * t).unsqueeze(1)
    return pred, torch.zeros(len(vel), 1, device=vel.device)
