import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import TFTConfig
from data_preprocessor import Preprocessor, TFTDataset, custom_collate_fn, send_data_to_device
from metric import QuantileLoss
from tft import TemporalFusionTransformer, TFTModel
from tft_interp import interpret_prediction
from utils import denormalize_target


def load_data_from_txt() -> pd.DataFrame:
    """Load data frame from a text file"""

    # Load raw data
    data = pd.read_csv("LD2011_2014.txt", index_col=0, sep=";", decimal=",")
    data.index = pd.to_datetime(data.index)
    data.sort_index(inplace=True)

    # Resample the data
    data = data.resample("1h").mean().replace(0.0, np.nan)
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


def validate(
    dataloader: DataLoader,
    network: TemporalFusionTransformer,
    loss_fn: QuantileLoss,
    device: torch.device,
) -> float:
    """Validation step during the training"""

    val_losses = []
    network.eval()
    for x_batch, y_batch, _ in dataloader:
        x_batch, y_batch = send_data_to_device(
            input_batch=x_batch, output_batch=y_batch, device=device
        )
        with torch.no_grad():
            pred = network(x_batch)
            loss = loss_fn(pred.prediction, y_batch.target)
            val_losses.append(loss.item())

    avg_val_loss = sum(val_losses) / len(val_losses)
    network.train()

    return avg_val_loss


def main():
    """Test data preprocessor"""
    # Load raw data
    raw_df = load_data_from_txt()

    # Config
    quantile = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    cfg = TFTConfig(
        quantiles=quantile,
        output_size=len(quantile),
        time_cat_features=["hour", "day_of_week", "day", "month"],
        dynamic_cat_encoder=["hour", "day_of_week", "day", "month"],
        dynamic_cat_decoder=["hour", "day_of_week", "day"],
        dynamic_cont_encoder=["power_usage"],
        static_cats=["data_id"],
        cont_transform_method={"power_usage": "softplus"},
        target_var=["power_usage"],
    )

    # Device
    if cfg.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Preprocess data
    data_prep = Preprocessor(data=raw_df, cfg=cfg)
    data_frame = data_prep.preprocess_data()

    # Split data into training, validatio, and test sets
    split_data_frame = data_prep.split_data(
        data_frame=data_frame, total_seq_len=cfg.encoder_len + cfg.decoder_len
    )

    # TFT model
    tft_model = TFTModel(cfg)
    optimizer = Adam(params=tft_model.network.parameters(), lr=cfg.lr)
    loss_fn = QuantileLoss(cfg.quantiles)

    # Creat dataloader
    train_dataset = TFTDataset(cfg=cfg, data=split_data_frame["train"])
    val_dataset = TFTDataset(cfg=cfg, data=split_data_frame["val"])
    test_dataset = TFTDataset(cfg=cfg, data=split_data_frame["test"])
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=2,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, collate_fn=custom_collate_fn, num_workers=2
    )
    test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=custom_collate_fn)

    # Train
    best_val_loss = np.inf
    print("Training...")
    for e in range(cfg.num_epochs):
        losses = []
        for i, (x_batch, y_batch, _) in tqdm(
            enumerate(dataloader), total=len(dataloader), desc=f"Epoch {e+1}/{cfg.num_epochs}"
        ):
            x_batch, y_batch = send_data_to_device(
                input_batch=x_batch, output_batch=y_batch, device=device
            )
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
            dataloader=val_dataloader, network=tft_model.network, loss_fn=loss_fn, device=device
        )
        print(
            f"Epoch #{e}/{cfg.num_epochs}| Train loss: {avg_train_loss:.4f} | Val loss: {avg_val_loss: .4f}"
        )
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            tft_model.save()

    # Test
    print("Testing...")
    tft_model.load()
    test_df = split_data_frame["test"]
    for i, (x_batch, y_batch, index_batch) in enumerate(test_dataloader):
        x_batch, y_batch = send_data_to_device(
            input_batch=x_batch, output_batch=y_batch, device=device
        )
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

            # Get the full sequence of observation
            batch_seq_obs = []
            for idx in index_batch:
                start_seq_idx = test_df.loc[idx, "start_seq_idx"]
                end_seq_idx = test_df.loc[idx, "end_seq_idx"] - 1
                seq_obs = test_df.loc[start_seq_idx:end_seq_idx, cfg.target_var].to_numpy(
                    np.float32
                )
                batch_seq_obs.append(torch.tensor(seq_obs, dtype=torch.float32))

            batch_seq_obs = torch.stack(batch_seq_obs)

            # Denormalize target & observation
            denorm_target = denormalize_target(
                target=y_batch.target,
                target_var=cfg.target_var,
                normalizer=data_prep.normalizer_encoder,
            )
            denorm_obs = denormalize_target(
                target=batch_seq_obs,
                target_var=cfg.target_var,
                normalizer=data_prep.normalizer_encoder,
            )
            pred.prediction = denorm_preds
            interpret_prediction(pred=pred, cfg=cfg, target=denorm_target, obs=denorm_obs)
            break


if __name__ == "__main__":
    main()
