from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    branch_hidden: int = 128
    global_hidden: int = 64
    context_hidden: int = 128
    depth: int = 3
    dropout: float = 0.05


def _mlp(in_dim: int, hidden: int, out_dim: int, depth: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = in_dim
    for _ in range(max(depth - 1, 1)):
        layers.extend([nn.Linear(current, hidden), nn.SiLU()])
        if dropout:
            layers.append(nn.Dropout(dropout))
        current = hidden
    layers.append(nn.Linear(current, out_dim))
    return nn.Sequential(*layers)


class BranchCorrectionNet(nn.Module):
    """Permutation-equivariant DeepSets model supporting any branch count."""

    def __init__(self, branch_dim: int, global_dim: int, config: ModelConfig):
        super().__init__()
        self.branch_dim = branch_dim
        self.global_dim = global_dim
        self.config = config
        self.branch_encoder = _mlp(branch_dim, config.branch_hidden, config.branch_hidden,
                                   config.depth, config.dropout)
        self.global_encoder = _mlp(global_dim, config.global_hidden, config.global_hidden,
                                   2, config.dropout)
        fused = 2 * config.branch_hidden + config.global_hidden
        self.correction_head = _mlp(fused, config.context_hidden, 2,
                                    config.depth, config.dropout)

    def forward(self, branch_x: torch.Tensor, global_x: torch.Tensor) -> torch.Tensor:
        branch_h = self.branch_encoder(branch_x)
        pooled = branch_h.mean(dim=1, keepdim=True).expand_as(branch_h)
        global_h = self.global_encoder(global_x).unsqueeze(1).expand(-1, branch_h.size(1), -1)
        return self.correction_head(torch.cat([branch_h, pooled, global_h], dim=-1))

    def metadata(self) -> dict:
        return {"branch_dim": self.branch_dim, "global_dim": self.global_dim,
                "model_config": asdict(self.config)}


def decode_prediction(prediction: torch.Tensor, q0: torch.Tensor, x0: torch.Tensor,
                      q_total: torch.Tensor, target_mean: torch.Tensor,
                      target_std: torch.Tensor, target_mode: str = "correction",
                      flow_parameterization: str = "logit") -> tuple[torch.Tensor, torch.Tensor]:
    """Decode standardized model output while enforcing flow conservation."""
    if target_mode not in {"correction", "direct"}:
        raise ValueError(target_mode)
    if flow_parameterization not in {"logit", "raw_ratio"}:
        raise ValueError(flow_parameterization)
    value = prediction * target_std + target_mean
    r0 = torch.clamp(q0 / q_total[:, None], min=1e-10)
    if flow_parameterization == "logit":
        flow_logits = value[..., 0]
        if target_mode == "correction":
            flow_logits = torch.log(r0) + flow_logits
        ratios = torch.softmax(flow_logits, dim=1)
    else:
        ratios = value[..., 0]
        if target_mode == "correction":
            ratios = r0 + ratios
        ratios = torch.clamp(ratios, min=1e-10)
        ratios = ratios / ratios.sum(dim=1, keepdim=True)
    q = ratios * q_total[:, None]
    x = value[..., 1]
    if target_mode == "correction":
        x = x0 + x
    x = torch.clamp(x, min=1e-3, max=100.0)
    return q, x


def decode_correction(prediction: torch.Tensor, q0: torch.Tensor, x0: torch.Tensor,
                      q_total: torch.Tensor, target_mean: torch.Tensor,
                      target_std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward-compatible decoder for version-1 correction/logit checkpoints."""
    return decode_prediction(prediction, q0, x0, q_total, target_mean, target_std,
                             target_mode="correction", flow_parameterization="logit")
