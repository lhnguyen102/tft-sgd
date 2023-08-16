import os
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import TFTConfig
from tft import TFTOutput


class TFTInterpreter:
    """Analysis of TFT results"""

    def __init__(self, cfg: TFTConfig) -> None:
        self.cfg = cfg

    def analyze_tft_output(self, prediction: TFTOutput, attn_min: float = 1e-5) -> dict:
        """Analyze attention score and variable selections"""
        # Attention score. Compute the average over number of attention heads
        attn_score = torch.concat(
            [prediction.encoder_attn_weight, prediction.decoder_attn_weight], dim=-1
        )
        attn_score[attn_score < attn_min] = float("nan")

        # Variable selection for each timestep
        assert prediction.encoder_var_weight.size(-2) == 1, "3rd dimension must be 1"
        assert prediction.decoder_var_weight.size(-2) == 1, "3rd dimension must be 1"
        encoder_var_score = (
            prediction.encoder_var_weight.squeeze(-2).sum(dim=1) / self.cfg.encoder_len
        )
        decoder_var_score = (
            prediction.decoder_var_weight.squeeze(-2).sum(dim=1) / self.cfg.decoder_len
        )
        static_var_score = prediction.static_var_weight.squeeze(1)

        return dict(
            attn_score=attn_score.detach().cpu().numpy(),
            encoder_var_score=encoder_var_score.detach().cpu().numpy(),
            decoder_var_score=decoder_var_score.detach().cpu().numpy(),
            static_var_score=static_var_score.detach().cpu().numpy(),
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

        return lower_bound.detach().cpu().numpy(), upper_bound.detach().cpu().numpy()

    def quantile_to_prediciton(self, raw_pred: torch.Tensor) -> np.ndarray:
        """Compute averaged prediction for quantiled prediciton
        raw_pred (sample, timestep, num_quant)
        """
        return torch.mean(raw_pred, dim=2).detach().cpu().numpy()


class Visualizer:
    """Visual data interpretation"""

    def __init__(
        self,
        figsize: tuple = (12, 6),
        lw: float = 2,
        fontsize: float = 16,
        ndiv_x: int = 5,
        ndiv_y: int = 4,
    ) -> None:
        self.figsize = figsize
        self.lw = lw
        self.fontsize = fontsize
        self.ndiv_x = ndiv_x
        self.ndiv_y = ndiv_y

    def plot_attn_score_heat_map(
        self,
        x_heat: np.ndarray,
        attn_score: np.ndarray,
        filename: str = "heat_map_attention_score",
        saved_dir: Union[str, None] = "./figure",
    ) -> None:
        """Visualize attention score through heat map"""

        fig = plt.figure(figsize=self.figsize)
        ax = plt.axes()
        cax = ax.imshow(attn_score, cmap="plasma", aspect="auto")
        fig.colorbar(cax, ax=ax)

        # x axis
        x_tick_labels = np.linspace(x_heat.min(), x_heat.max(), num=self.ndiv_x, endpoint=True)
        x_positions = np.linspace(0, attn_score.shape[1], self.ndiv_x, endpoint=True)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_tick_labels.astype(int))

        # y axis
        y_positions = np.linspace(1, attn_score.shape[0] + 1, self.ndiv_y, endpoint=True)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_positions.astype(int))
        ax.tick_params(axis="both", which="both", direction="inout", labelsize=self.fontsize)

        ax.set_title("Heat Map Attention Score")
        # Save figure
        if saved_dir is not None:
            os.makedirs(saved_dir, exist_ok=True)
            saving_path = f"{saved_dir}/{filename}.png"
            plt.savefig(saving_path, bbox_inches="tight")
            plt.close()
            print(f"Figure {filename} saved at {saved_dir}")
        else:
            plt.show()

    def plot_prediction(
        self,
        x_pred: np.ndarray,
        pred_val: np.ndarray,
        pred_low: Union[np.ndarray, None] = None,
        pred_high: Union[np.ndarray, None] = None,
        x_actual: Union[np.ndarray, None] = None,
        actual_val: Union[np.ndarray, None] = None,
        filename: str = "prediction",
        saved_dir: Union[str, None] = "./figure",
    ) -> (plt.Figure, plt.Axes):
        """Plot prediction versus actual"""

        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot prediction
        ax.plot(x_pred, pred_val, lw=self.lw, color="tab:blue", label="Prediction")

        # Plot uncertainty bounds if they exist
        if pred_low is not None and pred_high is not None:
            ax.fill_between(
                x_pred, pred_low, pred_high, facecolor="green", alpha=0.3, label="Uncertainty"
            )

        # Plot actual values if they exist
        if x_actual is not None and actual_val is not None:
            ax.plot(x_actual, actual_val, lw=self.lw, color="tab:red", label="Actual")

        # Set number after comma
        ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:.2f}".format(x)))

        # Setting y-ticks for ax1
        y_vals_combined = (
            np.concatenate([pred_high, pred_low, pred_val])
            if x_actual is None
            else np.concatenate([pred_high, pred_low, pred_val, actual_val])
        )
        max_y_vals = y_vals_combined.max()
        min_y_vals = y_vals_combined.min()
        y_ticks = np.linspace(y_vals_combined.min(), y_vals_combined.max(), self.ndiv_y)
        ax.set_yticks(y_ticks)

        # Setting x-ticks based on provided x-values
        x_vals_combined = x_pred if x_actual is None else np.concatenate([x_pred, x_actual])
        x_ticks = np.linspace(x_vals_combined.min(), x_vals_combined.max(), self.ndiv_x)
        x_ticks = np.unique(np.round(x_ticks))
        ax.set_xticks(x_ticks)
        ax.tick_params(axis="both", which="both", direction="inout", labelsize=self.fontsize)
        ax.set_ylim([min_y_vals, max_y_vals])
        ax.plot(
            [x_pred[0], x_pred[0]],
            [min_y_vals, max_y_vals],
            color="black",
            linestyle="--",
            lw=self.lw,
        )

        ax.legend(
            loc="best", edgecolor="black", fontsize=0.9 * self.fontsize, ncol=2, framealpha=0.3
        )
        ax.set_title("Prediction vs Actual")

        # Save figure
        if saved_dir is not None:
            os.makedirs(saved_dir, exist_ok=True)
            saving_path = f"{saved_dir}/{filename}.png"
            plt.savefig(saving_path, bbox_inches="tight")
            plt.close()
            print(f"Figure {filename} saved at {saved_dir}")
        else:
            plt.show()

        return fig, ax

    def plot_prediction_with_attention_score(
        self,
        x_pred: np.ndarray,
        pred_val: np.ndarray,
        x_attn: np.ndarray,
        attn_score: np.ndarray,
        horizon_idx: int,
        x_actual: Union[np.ndarray, None] = None,
        actual_val: Union[np.ndarray, None] = None,
        filename: str = "attention_score",
        saved_dir: Union[str, None] = "./figure",
    ) -> None:
        """Plot the twin axis where the prediction and actual are in left side and the attention
        score is on the right side.
        """
        # Create the main figure and axis
        fig = plt.figure(figsize=self.figsize)
        ax1 = plt.axes()

        # Plot predicted values on main y-axis
        ax1.set_xlabel("Index", fontsize=self.fontsize)
        ax1.set_ylabel("Predicted Value", color="tab:blue", fontsize=self.fontsize)
        ax1.plot(x_pred, pred_val, lw=self.lw, color="tab:blue", label="Prediction")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        # If actual values are provided, plot them on the same axis
        if x_actual is not None and actual_val is not None:
            ax1.plot(x_actual, actual_val, color="tab:red", lw=self.lw, label="Actual")

        # Set number after comma
        ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:.2f}".format(x)))

        # Setting y-ticks for ax1
        y_vals_combined = pred_val if x_actual is None else np.concatenate([pred_val, actual_val])
        y_ticks = np.linspace(y_vals_combined.min(), y_vals_combined.max(), self.ndiv_y)
        ax1.set_yticks(y_ticks)

        # Setting x-ticks based on provided x-values
        x_vals_combined = x_pred if x_actual is None else np.concatenate([x_pred, x_attn])
        x_ticks = np.linspace(x_vals_combined.min(), x_vals_combined.max(), self.ndiv_x)
        x_ticks = np.unique(np.round(x_ticks))
        ax1.set_xticks(x_ticks)
        ax1.set_ylim([y_vals_combined.min(), y_vals_combined.max()])
        ax1.tick_params(axis="both", which="both", direction="inout", labelsize=self.fontsize)

        # Create a second y-axis for attention scores
        ax2 = ax1.twinx()
        ax2.set_ylabel("Attention Score", color="black", fontsize=self.fontsize)
        ax2.plot(x_attn, attn_score, color="grey", lw=self.lw, label="Attention Score")
        ax2.plot(
            [x_pred[0], x_pred[0]],
            [attn_score.min(), attn_score.max()],
            color="black",
            linestyle="--",
            lw=self.lw,
        )
        ax2.tick_params(
            axis="y",
            labelcolor="black",
            which="both",
            direction="inout",
            labelsize=self.fontsize,
        )

        # Set number after comma
        ax2.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:.2f}".format(x)))

        # Setting y-ticks for ax2 (attention scores)
        ax2_y_ticks = np.linspace(attn_score.min(), attn_score.max(), self.ndiv_y)
        ax2.set_yticks(ax2_y_ticks)
        ax2.set_ylim([attn_score.min(), attn_score.max()])

        # Legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()

        lines = lines1 + lines2
        labels = labels1 + labels2
        ax1.legend(
            lines,
            labels,
            loc="best",
            edgecolor="black",
            fontsize=0.9 * self.fontsize,
            ncol=2,
            framealpha=0.3,
        )

        # Show the plot
        plt.title(
            f"Prediction with Attention Scores - horizon {horizon_idx}", fontsize=self.fontsize
        )
        fig.tight_layout()

        # Save figure
        if saved_dir is not None:
            os.makedirs(saved_dir, exist_ok=True)
            saving_path = f"{saved_dir}/{filename}.png"
            plt.savefig(saving_path, bbox_inches="tight")
            plt.close()
            print(f"Figure {filename} saved at {saved_dir}")
        else:
            plt.show()

    def plot_importance_features(
        self,
        var_names: List[str],
        score: np.ndarray,
        var_type: str,
        filename: str = "attention_score",
        saved_dir: Union[str, None] = "./figure",
    ) -> None:
        """Plot the score of the features that impacts on the prediction"""

        # Sort the scores and names
        sorted_indices = np.argsort(score)[::-1]
        sorted_names = [var_names[i] for i in sorted_indices]
        sorted_scores = score[sorted_indices]

        # Create the bar plot
        plt.figure(figsize=self.figsize)
        axe = plt.axes()
        axe.barh(sorted_names, sorted_scores, color="tab:blue")
        axe.set_xlabel("Feature Importance Score", fontsize=self.fontsize)
        axe.tick_params(axis="both", which="both", direction="inout", labelsize=self.fontsize)
        plt.title(f"Feature Importance for {var_type}", fontsize=self.fontsize)
        plt.gca().invert_yaxis()  # To display the feature with the highest score at the top
        plt.tight_layout()

        # Save figure
        if saved_dir is not None:
            os.makedirs(saved_dir, exist_ok=True)
            saving_path = f"{saved_dir}/{filename}.png"
            plt.savefig(saving_path, bbox_inches="tight")
            plt.close()
            print(f"Figure {filename} saved at {saved_dir}")
        else:
            plt.show()
