import os
from typing import List, Tuple, Union

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from config import TFTConfig
from tft import TFTOutput


def add_white_background(image):
    # Create a white background image
    white_background = Image.new("RGB", image.size, "white")

    # Paste the original image onto the white background
    # Using the image itself as the mask ensures that any transparency is respected
    white_background.paste(image, (0, 0), mask=image)
    return white_background


def create_gif(
    cfg: TFTConfig,
    preds: List[TFTOutput],
    observations: List[torch.Tensor],
    obs_dts: List[np.ndarray],
) -> None:
    """Create GIF"""
    # Interpretation
    interpreter = TFTInterpreter(cfg)
    viz = Visualizer()

    pred_plots = []
    actual_plots = []
    x_pred_plots = []
    x_actual_plots = []
    low_plots = []
    hight_plots = []
    hm_attn_scores = []
    for pred, obs in zip(preds, observations):
        # Prediction results
        interp_results = interpreter.analyze_tft_output(pred)
        pred_mean = interpreter.quantile_to_prediciton(pred.prediction)
        lower_bound, upper_bound = interpreter.get_uncertainty_bounds(pred.prediction, pos=1)

        # Indices
        plot_idx = 0  # First obs in the batch
        time_idx = np.arange(cfg.decoder_len)  # Prediction time index

        # Prediction and actual values
        pred_val = pred_mean[plot_idx, :]
        pred_val[np.isnan(pred_val)] = 0.0

        actual_val = obs[plot_idx, :].detach().cpu().numpy().flatten()
        actual_time_idx = np.concatenate(
            [np.arange(-cfg.encoder_len, 0), np.arange(cfg.decoder_len)]
        )

        # Attention score
        hm_attn_score = interp_results["attn_score"][plot_idx, :]
        hm_attn_score = hm_attn_score.sum(axis=1) / hm_attn_score.shape[1]
        hm_attn_score[np.isnan(hm_attn_score)] = 0.0

        pred_plots.append(pred_mean.flatten())
        actual_plots.append(actual_val)
        x_pred_plots.append(time_idx)
        x_actual_plots.append(actual_time_idx)
        low_plots.append(lower_bound.flatten())
        hight_plots.append(upper_bound.flatten())
        hm_attn_scores.append(hm_attn_score)

    viz.create_gif(
        x_pred=x_pred_plots,
        y_pred=pred_plots,
        x_true=x_actual_plots,
        y_true=actual_plots,
        pred_low=low_plots,
        pred_high=hight_plots,
        hm_attn_scores=hm_attn_scores,
    )


def interpret_prediction(
    pred: TFTOutput, cfg: TFTConfig, target: torch.Tensor, obs: Union[torch.Tensor, None] = None
) -> None:
    """Intepret the predictions"""
    # Interpretation
    interpreter = TFTInterpreter(cfg)
    viz = Visualizer()

    interp_results = interpreter.analyze_tft_output(pred)
    pred_mean = interpreter.quantile_to_prediciton(pred.prediction)
    lower_bound, upper_bound = interpreter.get_uncertainty_bounds(pred.prediction, pos=1)

    # Indices
    plot_idx = 0  # First obs in the batch
    horizon_idx = 0  # First prediction
    time_idx = np.arange(cfg.decoder_len)  # Prediction time index
    attn_time_idx = np.concatenate([np.arange(-cfg.encoder_len, 0), np.arange(cfg.decoder_len)])

    # Prediction and actual values
    pred_val = pred_mean[plot_idx, :]
    pred_val[np.isnan(pred_val)] = 0.0
    if obs is None:
        actual_val = target[plot_idx, :].detach().cpu().numpy().flatten()
        actual_time_idx = time_idx
    else:
        actual_val = obs[plot_idx, :].detach().cpu().numpy().flatten()
        actual_time_idx = np.concatenate(
            [np.arange(-cfg.encoder_len, 0), np.arange(cfg.decoder_len)]
        )

    # Attention score
    attn_score = interp_results["attn_score"][plot_idx, horizon_idx]
    attn_score = attn_score.sum(axis=0) / attn_score.shape[0]
    attn_score[np.isnan(attn_score)] = 0.0
    hm_attn_score = interp_results["attn_score"][plot_idx, :]
    hm_attn_score = hm_attn_score.sum(axis=1) / hm_attn_score.shape[1]
    hm_attn_score[np.isnan(hm_attn_score)] = 0.0

    # Selection variable score
    encoder_var_score = interp_results["encoder_var_score"][plot_idx]
    decoder_var_score = interp_results["decoder_var_score"][plot_idx]
    static_var_score = interp_results["static_var_score"][plot_idx]

    # Visualize the prediction with the actual values
    viz.plot_prediction(
        x_pred=time_idx,
        pred_val=pred_val,
        pred_low=lower_bound[plot_idx, :],
        pred_high=upper_bound[plot_idx, :],
        x_actual=actual_time_idx,
        actual_val=actual_val,
    )

    # Overlap the attention score with prediction and actual values
    viz.plot_prediction_with_attention_score(
        x_pred=time_idx,
        pred_val=pred_val,
        x_attn=attn_time_idx,
        attn_score=attn_score,
        x_actual=actual_time_idx,
        actual_val=actual_val,
        horizon_idx=horizon_idx,
        pred_low=lower_bound[plot_idx, :],
        pred_high=upper_bound[plot_idx, :],
    )

    # Visualize the important feature fore the encoder. Note that we can do the same for decoder
    # and static variable
    viz.plot_importance_features(
        var_names=cfg.dynamic_encoder_vars,
        score=encoder_var_score,
        filename="encoder_var_score",
        var_type="Encoder Variables",
    )
    viz.plot_importance_features(
        var_names=cfg.dynamic_decoder_vars,
        score=decoder_var_score,
        filename="decoder_var_score",
        var_type="Decoder Variables",
    )
    viz.plot_importance_features(
        var_names=cfg.static_vars,
        score=static_var_score,
        filename="static_var_score",
        var_type="Static Variables",
    )

    # Heatmap
    viz.plot_attn_score_heat_map(x_heat=attn_time_idx, attn_score=hm_attn_score)


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
        figsize: tuple = (13, 6),
        lw: float = 2,
        fontsize: float = 22,
        ndiv_x: int = 5,
        ndiv_y: int = 4,
    ) -> None:
        self.figsize = figsize
        self.lw = lw
        self.fontsize = fontsize
        self.ndiv_x = ndiv_x
        self.ndiv_y = ndiv_y

    def create_gif(
        self,
        x_pred: List[np.ndarray],
        y_pred: List[np.ndarray],
        x_true: List[np.ndarray],
        y_true: List[np.ndarray],
        pred_low: List[np.ndarray],
        pred_high: List[np.ndarray],
        hm_attn_scores: List[np.ndarray],
        save_dir: str = "./figure",
    ) -> None:
        """Create gif for forecasting"""

        count = 0
        images = []
        for x_p, y_p, x_t, y_t, y_low, y_high, hm_attn_score in zip(
            x_pred, y_pred, x_true, y_true, pred_low, pred_high, hm_attn_scores
        ):
            # Figure setup
            filename = f"prediction_{count}"
            fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(22, 16))
            self.lw = 4
            self.fontsize = 26

            # Plot
            self.plot_attn_score_heat_map(
                x_heat=x_t, attn_score=hm_attn_score, saved_dir=None, ax=axs[1]
            )
            self.plot_prediction(
                x_pred=x_p,
                pred_val=y_p,
                x_actual=x_t,
                actual_val=y_t,
                pred_low=y_low,
                pred_high=y_high,
                filename=filename,
                ax=axs[0],
                saved_dir=None,
                is_subplot=True,
            )
            fig.tight_layout()
            fig.subplots_adjust(hspace=0.2)
            plt.savefig(os.path.join(save_dir, filename))
            plt.close(fig)

            count += 1
            images.append(f"{save_dir}/{filename}.png")

        # Create GIF
        loaded_images = [imageio.imread(image) for image in images]
        imageio.mimsave(f"{save_dir}/forecast.gif", loaded_images, duration=2, loop=0)

        # Delete saved images
        for filepath in images:
            os.remove(filepath)

    def plot_attn_score_heat_map(
        self,
        x_heat: np.ndarray,
        attn_score: np.ndarray,
        filename: str = "heat_map_attention_score",
        ax: plt.Axes = None,
        saved_dir: Union[str, None] = "./figure",
    ) -> (plt.Figure, plt.Axes):
        """Visualize attention score through heat map"""
        fig = plt.figure(figsize=self.figsize)
        if ax is None:
            ax = plt.axes()
        cax = ax.imshow(attn_score, cmap="YlGnBu", aspect="auto")
        cbar = fig.colorbar(cax, ax=ax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=self.fontsize)

        # x axis
        x_tick_labels = np.linspace(x_heat.min(), x_heat.max(), num=self.ndiv_x, endpoint=True)
        x_positions = np.linspace(0, attn_score.shape[1], self.ndiv_x, endpoint=True)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_tick_labels.astype(int))
        ax.set_xlabel("Time Index", fontsize=self.fontsize)

        # y axis
        y_positions = np.linspace(1, attn_score.shape[0] + 1, self.ndiv_y, endpoint=True)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_positions.astype(int))
        ax.set_ylabel("Horizon", fontsize=self.fontsize)

        ax.tick_params(axis="both", which="both", direction="inout", labelsize=self.fontsize)

        ax.set_title("Attention Score Heat Map", fontsize=self.fontsize, fontweight="semibold")
        fig.tight_layout()

        # Save figure
        if saved_dir is not None:
            os.makedirs(saved_dir, exist_ok=True)
            saving_path = f"{saved_dir}/{filename}.png"
            plt.savefig(saving_path, bbox_inches="tight")
            plt.close()
            print(f"Figure {filename} saved at {saved_dir}")
        plt.close()

        return fig, ax

    def plot_prediction(
        self,
        x_pred: np.ndarray,
        pred_val: np.ndarray,
        pred_low: Union[np.ndarray, None] = None,
        pred_high: Union[np.ndarray, None] = None,
        x_actual: Union[np.ndarray, None] = None,
        actual_val: Union[np.ndarray, None] = None,
        ax: plt.Axes = None,
        filename: str = "prediction",
        saved_dir: Union[str, None] = "./figure",
        is_subplot: bool = False,
    ) -> (plt.Figure, plt.Axes):
        """Plot prediction versus actual"""

        fig = plt.figure(figsize=self.figsize)
        if ax is None:
            ax = plt.axes()

        # Plot prediction
        ax.plot(x_pred, pred_val, lw=self.lw, color="tab:blue", label="Prediction")

        # Plot uncertainty bounds if they exist
        if pred_low is not None and pred_high is not None:
            ax.fill_between(
                x_pred, pred_low, pred_high, facecolor="tab:blue", alpha=0.4, label="Uncertainty"
            )

        # Plot actual values if they exist
        if x_actual is not None and actual_val is not None:
            ax.plot(
                x_actual, actual_val, lw=self.lw, linestyle="--", color="tab:red", label="Actual"
            )

        # Set number after comma
        ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:.0f}".format(x)))

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
        ax.plot([x_pred[0], x_pred[0]], [min_y_vals, max_y_vals], color="black", lw=1.2 * self.lw)
        ax.set_xlabel("Time Index", fontsize=self.fontsize)
        ax.set_ylabel("Power Usage", fontsize=self.fontsize)

        if not is_subplot:
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.25),
                edgecolor="black",
                fontsize=0.9 * self.fontsize,
                ncol=3,
                framealpha=0.3,
            )
        else:
            ax.legend(
                loc="lower center",
                edgecolor="black",
                fontsize=0.9 * self.fontsize,
                ncol=3,
                framealpha=0.3,
            )
        ax.set_title("Prediction vs Actual", fontsize=self.fontsize, fontweight="semibold")
        fig.tight_layout()

        # Save figure
        if saved_dir is not None:
            os.makedirs(saved_dir, exist_ok=True)
            saving_path = f"{saved_dir}/{filename}.png"
            plt.savefig(saving_path, bbox_inches="tight")
            plt.close()
            print(f"Figure {filename} saved at {saved_dir}")

        plt.close()

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
        pred_low: Union[np.ndarray, None] = None,
        pred_high: Union[np.ndarray, None] = None,
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
        ax1.set_xlabel("Time Index", fontsize=self.fontsize)
        ax1.set_ylabel("Power Usage", fontsize=self.fontsize)
        ax1.plot(x_pred, pred_val, lw=self.lw, color="tab:blue", label="Prediction")

        # If actual values are provided, plot them on the same axis
        if x_actual is not None and actual_val is not None:
            ax1.plot(
                x_actual, actual_val, linestyle="--", color="tab:red", lw=self.lw, label="Actual"
            )

        # Plot uncertainty bounds if they exist
        if pred_low is not None and pred_high is not None:
            ax1.fill_between(
                x_pred, pred_low, pred_high, facecolor="tab:blue", alpha=0.4, label="Uncertainty"
            )

        # Set number after comma
        ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:.0f}".format(x)))

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
            lw=1.2 * self.lw,
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
            loc="upper center",
            bbox_to_anchor=(0.5, -0.25),
            edgecolor="black",
            fontsize=0.9 * self.fontsize,
            ncol=4,
            framealpha=0.3,
        )

        # Show the plot
        plt.title(
            f"Prediction with Attention Scores - Horizon {horizon_idx}",
            fontsize=self.fontsize,
            fontweight="semibold",
        )
        fig.tight_layout()

        # Save figure
        if saved_dir is not None:
            os.makedirs(saved_dir, exist_ok=True)
            saving_path = f"{saved_dir}/{filename}.png"
            plt.savefig(saving_path, bbox_inches="tight")
            print(f"Figure {filename} saved at {saved_dir}")
        else:
            plt.show()
        plt.close()

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
        axe.set_xlabel("Score", fontsize=self.fontsize)
        axe.tick_params(axis="both", which="both", direction="inout", labelsize=self.fontsize)
        plt.title(
            f"Feature Importance for {var_type}", fontsize=self.fontsize, fontweight="semibold"
        )
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
