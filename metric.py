from typing import List

import torch
import torch.nn as nn


class QuantileLoss(nn.Module):
    """Quantile loss for time series predictions"""

    def __init__(self, quantiles: List[float]) -> None:
        super(QuantileLoss, self).__init__()
        assert len(quantiles) % 2 == 1, "Quantiles list length should be odd."
        self.quantiles = quantiles

    @staticmethod
    def quantile_loss(y_pred: torch.Tensor, y_true: torch.Tensor, tau: float) -> torch.Tensor:
        """Compute quantile loss"""
        loss = torch.where(y_true >= y_pred, tau * (y_true - y_pred), (1 - tau) * (y_pred - y_true))
        return torch.mean(loss)

    @staticmethod
    def weighted_quantile_loss(
        y_pred: torch.Tensor, y_true: torch.Tensor, tau: float
    ) -> torch.Tensor:
        """Compute quantile loss"""
        zero_tensor = torch.tensor(0.0, device=y_true.device, dtype=y_true.dtype)
        nom = torch.sum(
            tau * torch.max(y_true - y_pred, zero_tensor)
            + (1.0 - tau) * torch.max(y_pred - y_true, zero_tensor)
        )
        dnom = torch.sum(torch.abs(y_true))
        if dnom == 0:
            return torch.tensor(0.0, device=y_true.device, dtype=y_true.dtype)

        loss = 2 * nom / dnom
        return torch.mean(loss)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute loss"""
        assert y_pred.shape[:2] == y_true.shape[:2]
        total_loss = 0.0
        for i, tau in enumerate(self.quantiles):
            # TODO: need to handle multi-variates)
            total_loss += self.weighted_quantile_loss(
                y_pred=y_pred[..., [i]], y_true=y_true, tau=tau
            )

        return total_loss / len(self.quantiles)
