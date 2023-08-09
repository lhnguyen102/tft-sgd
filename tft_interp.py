from typing import Union, Tuple, List
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

        # Variable selection for each timestep.
        assert encoder_var_score.size(-2) == 1, "3rd dimension must be 1"
        assert decoder_var_score.size(-2) == 1, "3rd dimension must be 1"
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

    def get_uncertainty_bounds(
        self, raw_pred: torch.Tensor, pos: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get upper and lower bounds for the prediction from quantile outputs. Make sure that the
        quantile vector follows the increasing order from left to right. For example,
        quantile = [0.05, 0.25, 0.5, 0.75, 0.95]
        """
        pos_limit = raw_pred.size(-1) // 2
        if pos > pos_limit:
            pos = pos_limit
        lower_bound = raw_pred[..., pos]
        upper_bound = raw_pred[..., -pos - 1]

        return lower_bound, upper_bound

    def average_quantile_prediciton(self, raw_pred: torch.Tensor) -> np.ndarray:
        """Compute averaged prediction for quantiled prediciton
        raw_pred (sample, timestep, num_quant)
        """
        return torch.mean(raw_pred, dim=-2).detach().cpu().numpy()


class Visualizer:
    """Visual data interpretation"""

    def __init__(self, figsize: tuple = (12, 6), lw: float = 1, fontsize: float = 12) -> None:
        self.figsize = figsize
        self.lw = lw
        self.fontsize = fontsize

    def plot_prediction(
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

    def plot_prediction_with_attention_score(
        self,
        x_pred: np.ndarray,
        pred_val: np.ndarray,
        x_attn: np.ndarray,
        attn_score: np.ndarray,
        x_actual: Union[np.ndarray, None] = None,
        actual_val: Union[np.ndarray, None] = None,
    ) -> None:
        """Plot the twin axis where the prediction and actual are in left side and the attention
        score is on the right side.
        """
        # Create the main figure and axis
        fig, ax1 = plt.subplots()

        # Plot predicted values on main y-axis
        ax1.set_xlabel("Index")
        ax1.set_ylabel("Predicted Value", color="tab:blue")
        ax1.plot(x_pred, pred_val, color="tab:blue", label="Prediction")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        # If actual values are provided, plot them on the same axis
        if x_actual is not None and actual_val is not None:
            ax1.plot(x_actual, actual_val, color="tab:green", label="Actual")
            ax1.legend(loc="upper left")

        # Create a second y-axis for attention scores
        ax2 = ax1.twinx()
        ax2.set_ylabel("Attention Score", color="tab:red")
        ax2.plot(x_attn, attn_score, color="tab:red", linestyle="--", label="Attention Score")
        ax2.tick_params(axis="y", labelcolor="tab:red")
        ax2.set_ylim([0, 1])

        # Show the plot
        plt.title("Prediction with Attention Scores")
        fig.tight_layout()
        plt.show()

    def plot_importance_features(
        self, var_names: List[str], score: np.ndarray, var_type: str
    ) -> None:
        """Plot the score of the features that impacts on the prediction"""

        # Sort the scores and names
        sorted_indices = np.argsort(score)[::-1]
        sorted_names = [var_names[i] for i in sorted_indices]
        sorted_scores = score[sorted_indices]

        # Create the bar plot
        plt.figure(figsize=(10, 6))
        plt.barh(sorted_names, sorted_scores, color="tab:blue")
        plt.xlabel("Feature Importance Score")
        plt.title(f"Feature Importance for {var_type}")
        plt.gca().invert_yaxis()  # To display the feature with the highest score at the top
        plt.tight_layout()
        plt.show()
