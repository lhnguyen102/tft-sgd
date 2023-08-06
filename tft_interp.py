from typing import Union
import torch
from tft import TFTOutput
from config import TFTConfig
import matplotlib.pyplot as plt
import numpy as np


class TFTInterpreter:
    """Analysis of TFT results"""

    def __init__(self, cfg: TFTConfig) -> None:
        self.cfg = cfg

    def analyze_tft_output(self, prediction: TFTOutput) -> dict:
        """Analyze attention score and variable selections"""
        # Attention score. Compute the average over number of attention heads
        attn_score = torch.concat(
            [prediction.encoder_attn_weight, prediction.decoder_attn_weight], dim=-1
        )

        # Variable selection for each timestep
        encoder_var_score = (
            prediction.encoder_var_selection_weight.squeeze(-2).sum(dim=1) / self.cfg.encoder_len
        )
        decoder_var_score = (
            prediction.decoder_var_selection_weight.squeeze(-2).sum(dim=1) / self.cfg.decoder_len
        )
        static_var_score = prediction.static_var_weight.squeeze(1)

        return dict(
            attn_score=attn_score,
            encoder_var_score=encoder_var_score,
            decoder_var_score=decoder_var_score,
            static_var_score=static_var_score,
        )


class Visualizer:
    """Visual data interpretation"""

    def __init__(self, figsize: tuple = (12, 6), lw: float = 1, fontsize: float = 12) -> None:
        self.figsize = figsize
        self.lw = lw
        self.fontsize = fontsize

    def plot_prediction_vs_actual(
        self,
        pred_dt: np.ndarray,
        pred_val: np.ndarray,
        pred_low: Union[np.ndarray, None] = None,
        pred_high: Union[np.ndarray, None] = None,
        actual_dt: Union[np.ndarray, None] = None,
        actual_val: Union[np.ndarray, None] = None,
    ) -> (plt.Figure, plt.Axes):
        """Plot prediction versus actual"""

        fig, ax = plt.subplots()

        # Plot prediction
        ax.plot(pred_dt, pred_val, label="Prediction")

        # Plot uncertainty bounds if they exist
        if pred_low is not None and pred_high is not None:
            ax.fill_between(
                pred_dt, pred_low, pred_high, color="gray", alpha=0.5, label="Uncertainty"
            )

        # Plot actual values if they exist
        if actual_dt is not None and actual_val is not None:
            ax.plot(actual_dt, actual_val, label="Actual")

        ax.legend()  # Show legend
        ax.grid(True)  # Add grid
        ax.set_title("Prediction vs Actual")  # Add title

        return fig, ax
