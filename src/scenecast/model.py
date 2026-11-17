"""Vectorized scene transformer for multimodal trajectory prediction.

Every agent track and every lane centerline is a polyline. Each polyline is
compressed to one token by a small PointNet-style encoder, a transformer lets
all tokens attend to each other, and K learned mode queries decode K candidate
futures for the focal agent plus a confidence for each.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .data.scene import AGENT_FEATS, N_AGENT_TYPES, T_FUT, T_OBS

POS_SCALE = 50.0
VEL_SCALE = 10.0


@dataclass
class ModelConfig:
    d_model: int = 128
    n_heads: int = 8
    enc_layers: int = 4
    dec_layers: int = 2
    dropout: float = 0.1
    decoder_dropout: float = 0.0  # see the note in SceneCast.__init__ before raising this
    query_init_std: float = 1.0
    n_modes: int = 6
    use_map: bool = True
    use_neighbors: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def mlp(d_in: int, d_hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.LayerNorm(d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_out))


def masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Max over the points axis (-2), ignoring masked points. All-masked rows give zeros."""
    x = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
    pooled = x.max(dim=-2).values
    return torch.where(mask.any(dim=-1, keepdim=True), pooled, torch.zeros_like(pooled))


class PolylineEncoder(nn.Module):
    """[B, N, P, F] points -> [B, N, D] tokens (two rounds of point MLP + max-pool)."""

    def __init__(self, d_in: int, d_model: int):
        super().__init__()
        self.point = mlp(d_in, d_model, d_model)
        self.mix = mlp(2 * d_model, d_model, d_model)
        self.out = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.point(x)
        ctx = masked_max(h, mask).unsqueeze(-2).expand_as(h)
        h = self.mix(torch.cat([h, ctx], dim=-1))
        return self.out(masked_max(h, mask))


class SceneCast(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or ModelConfig()
        d = cfg.d_model
        self.agent_enc = PolylineEncoder(AGENT_FEATS + 2, d)  # + time, + validity
        self.lane_enc = PolylineEncoder(5, d)  # x, y, dx, dy, in-intersection
        self.agent_type = nn.Embedding(N_AGENT_TYPES, d)
        self.kind = nn.Embedding(3, d)  # focal agent / other agent / lane

        layer = nn.TransformerEncoderLayer(d, cfg.n_heads, 4 * d, cfg.dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.enc_layers, enable_nested_tensor=False)
        # No dropout in the decoder: with it, winner-takes-all training learns to use dropout noise as
        # the source of mode diversity, and the modes collapse onto one another at eval time.
        dec = nn.TransformerDecoderLayer(d, cfg.n_heads, 4 * d, cfg.decoder_dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec, cfg.dec_layers)

        # unit scale by default so the modes start out distinct
        self.mode_queries = nn.Parameter(torch.randn(cfg.n_modes, d) * cfg.query_init_std)
        self.focal_proj = nn.Linear(d, d)
        self.traj_head = mlp(d, 2 * d, T_FUT * 2)
        self.conf_head = mlp(d, d, 1)
        self.register_buffer("time", torch.linspace(-1.0, 0.0, T_OBS), persistent=False)

    def encode(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        agents, amask = batch["agents"], batch["agent_mask"]
        b, n, t, _ = agents.shape
        scale = agents.new_tensor([POS_SCALE, POS_SCALE, 1, 1, VEL_SCALE, VEL_SCALE])
        feats = torch.cat(
            [agents / scale, self.time.view(1, 1, t, 1).expand(b, n, t, 1), amask.unsqueeze(-1).float()], dim=-1
        )
        a_tok = self.agent_enc(feats, amask) + self.agent_type(batch["agent_type"])
        kind = torch.ones(b, n, dtype=torch.long, device=agents.device)
        kind[:, 0] = 0
        a_tok = a_tok + self.kind(kind)
        a_valid = amask.any(dim=-1)
        if not self.cfg.use_neighbors:
            a_valid = a_valid.clone()
            a_valid[:, 1:] = False

        lanes = batch["lanes"]
        m, p = lanes.shape[1:3]
        delta = torch.zeros_like(lanes)
        delta[:, :, :-1] = lanes[:, :, 1:] - lanes[:, :, :-1]
        delta[:, :, -1] = delta[:, :, -2]
        attr = batch["lane_attr"].view(b, m, 1, 1).expand(b, m, p, 1).float()
        l_valid = batch["lane_mask"] if self.cfg.use_map else torch.zeros_like(batch["lane_mask"])
        l_feats = torch.cat([lanes / POS_SCALE, delta, attr], dim=-1)
        l_tok = self.lane_enc(l_feats, l_valid.unsqueeze(-1).expand(b, m, p)) + self.kind.weight[2]

        tokens = torch.cat([a_tok, l_tok], dim=1)
        valid = torch.cat([a_valid, l_valid], dim=1)
        tokens = self.encoder(tokens, src_key_padding_mask=~valid)
        return tokens, valid

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns trajectories [B, K, T_FUT, 2] in metres (focal frame) and mode logits [B, K]."""
        tokens, valid = self.encode(batch)
        b = tokens.shape[0]
        queries = self.mode_queries.unsqueeze(0) + self.focal_proj(tokens[:, :1])
        h = self.decoder(queries, tokens, memory_key_padding_mask=~valid)
        # the head predicts per-step displacements; summing them keeps outputs near unit scale
        traj = self.traj_head(h).view(b, self.cfg.n_modes, T_FUT, 2).cumsum(dim=2)
        return traj, self.conf_head(h).squeeze(-1)
