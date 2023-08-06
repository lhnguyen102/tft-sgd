from data_preprocessor import Preprocessor, TFTDataloader
from config import TFTConfig
import pandas as pd
import numpy as np
from tft import TemporalFusionTransformer
import torch
from torch.optim import Adam
from torch.nn import MSELoss


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

    tft_net = TemporalFusionTransformer(cfg)
    optimizer = Adam(params=tft_net.parameters(), lr=cfg.lr)
    loss_fn = MSELoss()
    dataloder = TFTDataloader(cfg=cfg, data=data_frame, shuffle=True)

    losses = []
    for x_batch, y_batch in dataloder:
        y_pred = tft_net(x_batch)
        loss = loss_fn(y_pred.prediction, y_batch.target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.items())


if __name__ == "__main__":
    main()
