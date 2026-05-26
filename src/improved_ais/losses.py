from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class WeightedL1(nn.Module):
    """Learned per-controller-dimension L1 weighting."""

    def __init__(self, pose_dim: int) -> None:
        super().__init__()
        self.rho = nn.Parameter(torch.zeros(pose_dim))

    def forward(self, pred: torch.Tensor, target: torch.Tensor, frame_mask: torch.Tensor | None = None) -> torch.Tensor:
        sigma = F.softplus(self.rho) + 1e-4
        err = torch.abs(pred - target)
        loss = 0.5 * err / (sigma**2) + torch.log1p(sigma**2)
        if frame_mask is not None:
            loss = loss * frame_mask[..., None]
            denom = torch.clamp(frame_mask.sum() * pred.shape[-1], min=1.0)
            return loss.sum() / denom
        return loss.mean()


def velocity_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape[1] < 2:
        return pred.new_tensor(0.0)
    return torch.mean(torch.abs((pred[:, 1:] - pred[:, :-1]) - (target[:, 1:] - target[:, :-1])))


def acceleration_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape[1] < 3:
        return pred.new_tensor(0.0)
    pred_acc = pred[:, 2:] - 2.0 * pred[:, 1:-1] + pred[:, :-2]
    target_acc = target[:, 2:] - 2.0 * target[:, 1:-1] + target[:, :-2]
    return torch.mean(torch.abs(pred_acc - target_acc))
