from typing import List

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import TFTConfig
from data_preprocessor import (
    Preprocessor,
    TFTDataset,
    TimeseriesEncoderNormalizer,
    custom_collate_fn,
)
from metric import QuantileLoss
from tft import TemporalFusionTransformer, TFTModel, TFTOutput
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
    pred_val[np.isnan(pred_val)] = 0.0
    actual_val = target[plot_idx, :].detach().cpu().numpy().flatten()
    attn_score = interp_results["attn_score"][plot_idx, horizon_idx]
    attn_score = attn_score.sum(axis=0) / attn_score.shape[0]
    attn_score[np.isnan(attn_score)] = 0.0
    encoder_var_score = interp_results["encoder_var_score"][plot_idx]
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
    viz.plot_importance_features(
        var_names=cfg.time_varying_encoder_vars,
        score=encoder_var_score,
        filename="encoder_var_score",
        var_type="Encoder Variable Scores",
    )


def validate(
    dataloader: DataLoader, network: TemporalFusionTransformer, loss_fn: QuantileLoss
) -> float:
    """Validate model"""

    val_losses = []
    network.eval()
    for x_batch, y_batch in dataloader:
        with torch.no_grad():
            pred = network(x_batch)
            loss = loss_fn(pred.prediction, y_batch.target)
            val_losses.append(loss.item())
    avg_val_loss = sum(val_losses) / len(val_losses)
    network.train()

    return avg_val_loss


def denormalize_target(
    target: torch.Tensor,
    target_var: List[str],
    normalizer: TimeseriesEncoderNormalizer,
) -> torch.Tensor:
    """Denormalize the target data"""
    target_numpy = target.detach().cpu().numpy()

    data_frame = pd.DataFrame(
        {name: target_numpy[..., i].flatten() for i, name in enumerate(target_var)}
    )

    batch_size, seq_len, _ = target.shape

    denorm_df = normalizer.denormalize(data_frame)
    denorm_targets = [
        torch.tensor(denorm_df[name].to_numpy(np.float32), dtype=torch.float32).reshape(
            batch_size, seq_len, 1
        )
        for name in target_var
    ]

    denorm_targets = []
    for target_name in target_var:
        tmp = torch.tensor(
            denorm_df[target_name].to_numpy(np.float32), dtype=torch.float32
        ).reshape(batch_size, seq_len, 1)

        denorm_targets.append(tmp)

    return torch.cat(denorm_targets, dim=2)


def main():
    """Test data preprocessor"""
    # Load raw data
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

    # Split data into training, validatio, and test sets
    split_data_frame = data_prep.split_data(
        data_frame=data_frame, total_seq_len=cfg.encoder_len + cfg.decoder_len
    )

    tft_model = TFTModel(cfg)
    optimizer = Adam(params=tft_model.network.parameters(), lr=cfg.lr)
    loss_fn = QuantileLoss(cfg.quantiles)

    # Dataloader
    train_dataset = TFTDataset(cfg=cfg, data=split_data_frame["train"])
    val_dataset = TFTDataset(cfg=cfg, data=split_data_frame["val"])
    test_dataset = TFTDataset(cfg=cfg, data=split_data_frame["test"])
    dataloader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=custom_collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, collate_fn=custom_collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, collate_fn=custom_collate_fn
    )

    # Training
    best_val_loss = np.inf
    for e in range(cfg.num_epochs):
        losses = []
        for i, (x_batch, y_batch) in tqdm(
            enumerate(dataloader), total=len(dataloader), desc=f"Epoch {e+1}/{cfg.num_epochs}"
        ):
            pred = tft_model.network(x_batch)
            loss = loss_fn(pred.prediction, y_batch.target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            if np.isnan(loss.item()):
                raise ValueError("Loss is NAN")
        avg_train_loss = sum(losses) / len(losses)

        # Validate
        avg_val_loss = validate(
            dataloader=val_dataloader, network=tft_model.network, loss_fn=loss_fn
        )
        print(
            f"Epoch #{e}/{cfg.num_epochs}| Train loss: {avg_train_loss:.4f} | Val loss: {avg_val_loss: .4f}"
        )
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            tft_model.save()

    # Testing
    for x_batch, y_batch in test_dataloader:
        with torch.no_grad():
            pred = tft_model.network(x_batch)

            denorm_preds = []
            for i in range(pred.prediction.shape[2]):
                denorm_pred = denormalize_target(
                    target=pred.prediction[..., [i]],
                    target_var=cfg.target_var,
                    normalizer=data_prep.normalizer_encoder,
                )
                denorm_preds.append(denorm_pred)
            denorm_preds = torch.cat(denorm_preds, dim=2)

            denorm_target = denormalize_target(
                target=y_batch.target,
                target_var=cfg.target_var,
                normalizer=data_prep.normalizer_encoder,
            )
            pred.prediction = denorm_preds
            interpret_prediction(pred=pred, cfg=cfg, target=denorm_target)
            break


if __name__ == "__main__":
    main()
