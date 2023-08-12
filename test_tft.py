import numpy as np
import pandas as pd
import torch
from torch.optim import Adam

from config import TFTConfig
from data_preprocessor import Preprocessor, TFTDataloader
from metric import QuantileLoss
from tft import TFTModel, TFTOutput
from tft_interp import TFTInterpreter, Visualizer


def load_data_from_txt() -> pd.DataFrame:
    """Load data frame from a text file"""

    # Load raw data
    data = pd.read_csv("LD2011_2014.txt", index_col=0, sep=";", decimal=",")
    data.index = pd.to_datetime(data.index)
    data.sort_index(inplace=True)

    # Resample the data
    data = data.resample("1h").mean().replace(0.0, np.nan)
    earliest_time = data.index.min()
    data_frame = data[["MT_002", "MT_004", "MT_005", "MT_006", "MT_008"]]

    df_list = []

    for label in data_frame:
        ts = data_frame[label]

        start_date = min(ts.fillna(method="ffill").dropna().index)
        end_date = max(ts.fillna(method="bfill").dropna().index)

        active_range = (ts.index >= start_date) & (ts.index <= end_date)
        ts = ts[active_range].fillna(0.0)

        tmp = pd.DataFrame({"power_usage": ts})
        date = tmp.index

        tmp["date"] = date
        tmp["data_id"] = label

        # stack all time series vertically
        df_list.append(tmp)

    time_df = pd.concat(df_list).reset_index(drop=True)

    return time_df


def interpret_prediction(pred: TFTOutput, cfg: TFTConfig, target: torch.Tensor) -> None:
    """Intepret the predictions"""
    # Interpretation
    interpreter = TFTInterpreter(cfg)
    viz = Visualizer()

    interp_results = interpreter.analyze_tft_output(pred)
    pred_mean = interpreter.quantile_to_prediciton(pred.prediction)
    lower_bound, upper_bound = interpreter.get_uncertainty_bounds(pred.prediction, pos=1)

    # Visualization
    plot_idx = 0
    horizon_idx = 0
    time_idx = np.arange(cfg.decoder_len)
    attn_time_idx = np.concatenate([np.arange(-cfg.encoder_len, 0), np.arange(cfg.decoder_len)])
    pred_val = pred_mean[plot_idx, :]
    actual_val = target[plot_idx, :].detach().cpu().numpy().flatten()
    attn_score = interp_results["attn_score"][plot_idx, horizon_idx]
    attn_score = attn_score.sum(axis=0) / attn_score.shape[0]
    attn_score[np.isnan(attn_score)] = 0.0
    viz.plot_prediction(
        x_pred=time_idx,
        pred_val=pred_val,
        pred_low=lower_bound[plot_idx, :],
        pred_high=upper_bound[plot_idx, :],
        x_actual=time_idx,
        actual_val=actual_val,
    )
    viz.plot_prediction_with_attention_score(
        x_pred=time_idx,
        pred_val=pred_val,
        x_attn=attn_time_idx,
        attn_score=attn_score,
        x_actual=time_idx,
        actual_val=actual_val,
    )
    # viz.plot_importance_features(var_names=)


def main():
    """Test data preprocessor"""
    # # Load raw data
    raw_df = load_data_from_txt()

    # Config
    cfg = TFTConfig(
        time_varying_cat_encoder=["hour", "day", "day_of_week", "month"],
        time_varying_cat_decoder=["hour", "day", "day_of_week", "month"],
        time_varying_cont_encoder=["power_usage"],
        time_varying_cont_decoder=["power_usage"],
        static_cats=["data_id"],
        cont_transform_method={"power_usage": "softplus"},
        target_var=["power_usage"],
    )

    # Data preprocessing
    data_prep = Preprocessor(data=raw_df, cfg=cfg)
    data_frame = data_prep.preprocess_data()

    tft_model = TFTModel(cfg)
    optimizer = Adam(params=tft_model.network.parameters(), lr=cfg.lr)
    loss_fn = QuantileLoss(cfg.quantiles)
    dataloder = TFTDataloader(cfg=cfg, data=data_frame, shuffle=True)

    losses = []
    for x_batch, y_batch in dataloder:
        pred = tft_model.network(x_batch)
        loss = loss_fn(pred.prediction, y_batch.target)

        interpret_prediction(pred=pred, cfg=cfg, target=y_batch.target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        tft_model.save()


if __name__ == "__main__":
    main()
