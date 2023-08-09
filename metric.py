from typing import List
import torch
import torch.nn as nn


class QuantileLoss(nn.Module):
    """Quantile loss for time series predictions"""

    def __init__(self, quantiles: List[float]) -> None:
        super(QuantileLoss, self).__init__()
        self.quantiles = quantiles

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute loss"""
        assert y_pred.size() == y_true.size()

        losses = []

        for i, tau in enumerate(self.quantiles):
            errors = y_true - y_pred[..., i]
            loss = torch.max((tau - 1) * errors, tau * errors).unsqueeze(-1)
            losses.append(loss.mean())

        losses = 2 * torch.cat(losses, dim=2)

        return losses
