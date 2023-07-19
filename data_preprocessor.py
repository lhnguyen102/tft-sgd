from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from config import TFTConfig
import torch
from dataclasses import dataclass


@dataclass
class AutoencoderInputBatch:
    """Minitatch for encoder decoder"""

    cont_var: torch.Tensor = None
    cat_var: torch.Tensor = None
    encoder_len: int = None
    decoder_len: int = None


@dataclass
class TFTOutputBatch:
    """Output batch for TFT model"""

    target: torch.Tensor = None


class TFTDataloader:
    """Custom dataloader for time series"""

    def __init__(self, data: pd.DataFrame, cfg: TFTConfig, shuffle: bool = False) -> None:
        self.data = data
        self.cfg = cfg
        self.shuffle = shuffle
        self.indices = list[range(len(self.data)) - self.cfg.seq_len]
        self.num_samples = 0

    def reshuffle(self):
        """Reshuffle indices"""
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self) -> int:
        return len(self.indices) // self.cfg.batch_size

    def __getitem__(self, index: int) -> Tuple[AutoencoderInputBatch, TFTOutputBatch]:
        """Get batch of data"""
        # Index
        batch_indices = self.indices[
            index * self.cfg.batch_size : (index + 1) * self.cfg.batch_size
        ]

        # Input & ouput data
        cont_batches = []
        cat_batches = []
        target_batches = []
        for i in batch_indices:
            # Get sequence index from data frame. TODO: Pass `time_first` and `time_last`
            # as arguments for both Preprossessor and TFTDataloader
            start_seq_idx = int(self.data.loc[i, "time_first"].values)
            end_seq_idx = int(self.data.loc[i, "time_last"].values)

            # Get data from data frame
            cont_batches.append(self.data.loc[start_seq_idx:end_seq_idx, self.cfg.cont_var].values)
            cat_batches.append(self.data.loc[start_seq_idx:end_seq_idx, self.cfg.cat_var].values)
            target_batches.append(
                self.data.loc[
                    start_seq_idx + self.cfg.encoder_len : end_seq_idx,
                    self.cfg.target_var,
                ].values
            )

        input_batch = AutoencoderInputBatch(
            cont_var=torch.stack(cont_batches),
            cat_var=torch.stack(cat_batches),
            encoder_len=self.cfg.encoder_len,
            decoder_len=self.cfg.decoder_len,
        )
        output_batch = TFTOutputBatch(target=torch.stack(target_batches))

        return input_batch, output_batch

    def __iter__(self):
        self.reshuffle()
        self.num_samples = 0
        return self

    def __next__(self) -> Tuple[AutoencoderInputBatch, TFTOutputBatch]:
        if self.num_samples < len(self):
            result = self.__getitem__(self.num_samples)
            self.num_samples += 1
            return result
        else:
            raise StopIteration


class StandardNormalizer:
    """Standard normalization of the data"""

    def __init__(self) -> None:
        self.mean_: np.ndarray = None
        self.scale_: np.ndarray = None

    def fit(self, values: np.ndarray):
        self.mean_ = np.nanmean(values, axis=0)
        self.scale_ = np.nanstd(values, axis=0)
        self.scale_[self.scale_ == 0] = 1

        return self

    def transform(self, original_values: np.ndarray) -> np.ndarray:
        return (original_values - self.mean_) / self.scale_

    def fit_transform(self, original_values: np.ndarray) -> np.ndarray:
        return self.fit(original_values).transform(original_values)

    def inverse_transform(self, scaled_value: np.ndarray) -> np.ndarray:
        return scaled_value * self.scale_ + self.mean_


class LabelEncoder:
    """Transform the categorical variables to numerical label"""

    def __init__(self) -> None:
        self.labels = {}
        self.special_values = [np.nan, np.inf, -np.inf]

    def fit(self, original_values: np.ndarray):
        transform_values = np.array(
            ["SpecialValue" if i in self.special_values else i for i in original_values]
        )
        categories = np.unique(transform_values)

        for i, cat in enumerate(categories):
            self.labels[cat] = i

        return self

    def transform(self, original_values: np.ndarray) -> np.ndarray:
        # Replace values with their labels
        transform_values = np.array(
            ["SpecialValue" if i in self.special_values else i for i in original_values]
        )

        return np.array([self.labels[cat] for cat in transform_values])

    def fit_transform(self, original_values: np.ndarray) -> np.ndarray:
        return self.fit(original_values).transform(original_values)

    def inverse_transform(self, scaled_values: np.ndarray) -> np.ndarray:
        inv_labels = {v: k for k, v in self.labels.items()}
        inv_labels["SpecialValue"] = np.nan

        return np.array([inv_labels[i] for i in scaled_values])


class MultiLabelEncoder:
    """Transform multi categorical variables to the numerical labels. For example, the input data
    could be an array including multiple categorical variables
    """

    def __init__(self):
        self.encoders = {}

    def fit(self, original_values: np.ndarray):
        for i, row in enumerate(original_values):
            self.encoders[i] = LabelEncoder().fit(row)
        return self

    def transform(self, original_values: np.ndarray):
        transformed = np.empty_like(original_values)
        for i, row in enumerate(original_values):
            transformed[i] = self.encoders[i].transform(row)
        return transformed

    def fit_transform(self, original_values: np.ndarray) -> np.ndarray:
        return self.fit(original_values).transform(original_values)

    def inverse_transform(self, scaled_values: np.ndarray):
        inverse = np.empty_like(scaled_values)
        for i, row in enumerate(scaled_values):
            inverse[i] = self.encoders[i].inverse_transform(row)
        return inverse


class TimeseriesEncoderNormalizer:
    """This class is responsible for encoding the categorical variables and normalizing the
    continous variables. It also provide a decoding and denormalizing those variables"""

    def __init__(
        self,
        cat_var: Dict[str, np.ndarray],
        cont_var: Dict[str, np.ndarray],
        cat_encoding_method: Dict[str, str],
        cont_normalizing_method: Dict[str, str],
    ) -> None:
        self.cat_var = cat_var
        self.cont_var = cont_var
        self.cat_encoding_method = cat_encoding_method
        self.cont_normalizing_method = cont_normalizing_method
        self.cat_encoders: dict = {}
        self.cont_normalizers: dict = {}

    @property
    def cat_encoding_method(self) -> Dict[str, str]:
        """Get a dictionary of encoding methods for categorical variables"""

        return self._cat_encoding_method

    @cat_encoding_method.setter
    def cat_encoding_method(self, value: Dict[str, str]) -> None:
        """Set a dictionary of encoding methods for categorical variables"""

        self._cat_encoding_method = value
        for method in self._cat_encoding_method:
            if method == "label_encoder":
                self.cat_encoders[method] = LabelEncoder()
            else:
                raise ValueError(f"Method {method} does not exist")

    @property
    def cont_normalizing_method(self) -> Dict[str, str]:
        """Get a dictionary of normalizing methods for categorical variables"""

        return self._cont_normalizing_method

    @cont_normalizing_method.setter
    def cont_normalizing_method(self, value: Dict[str, str]) -> None:
        """Set a dictionary of normalizing methods for categorical variables"""

        self._cont_normalizing_method = value
        for method in self._cont_normalizing_method:
            if method == "standard":
                self.cont_normalizers[method] = StandardNormalizer()
            else:
                raise ValueError(f"Method {method} does not exist")

    def encode_cat_var(self, cat_data: pd.DataFrame) -> pd.DataFrame:
        """Encode all categorical variables"""

        assert len(cat_data) == len(self.cat_encoders), "Categorical encoders are invalid"
        cat_data_encoded = cat_data.copy()

        for name, encoder in self.cat_encoders.items():
            cat_data_encoded[name] = encoder.fit_transform(cat_data[name].values)

        return cat_data_encoded

    def decode_cat_var(self, cat_data: pd.DataFrame) -> pd.DataFrame:
        """Decode all categorical variables"""

        assert len(cat_data) == len(self.cat_encoders), "Categorical encoders are invalid"
        cat_data_decoded = cat_data.copy()

        for name, encoder in self.cat_encoders.items():
            assert len(encoder.labels) > 0, "{name} does not have encoder"
            cat_data_decoded[name] = encoder.inverse_transform(cat_data[name].values)

        return cat_data_decoded

    def encode_multi_cat_var(
        self, cat_data: pd.DataFrame, multi_cat_var: Dict[str, List[str]]
    ) -> pd.DataFrame:
        """Encode all mutli-categorical variables"""

        cat_data_encoded = cat_data.copy()
        for name, cat_var in multi_cat_var:
            self.cat_encoders[name].fit(cat_var)
            for var in cat_var:
                cat_data_encoded[var] = self.cat_encoders[name].transform(cat_data[var].values)

        return cat_data_encoded

    def decode_multi_cat_var(
        self, cat_data: pd.DataFrame, multi_cat_var: Dict[str, List[str]]
    ) -> pd.DataFrame:
        """Decode all multi-categorical variables"""

        cat_data_decoded = cat_data.copy()
        for name, cat_var in multi_cat_var:
            for var in cat_var:
                cat_data_decoded[var] = self.cat_encoders[name].inverse_transform(
                    cat_data[var].values
                )

        return cat_data_decoded

    def normalize(self, cont_data: pd.DataFrame) -> pd.DataFrame:
        """Normalize all continuous variables"""
        assert len(cont_data) == len(self.cont_normalizers), "Normalizers are invalid"

        cont_data_normalized = cont_data.copy()
        for name, normalizer in self.cont_normalizers.items():
            cont_data_normalized[name] = normalizer.fit_transform(cont_data[name].values)

        return cont_data_normalized

    def denormalize(self, cont_data: pd.DataFrame) -> pd.DataFrame:
        """Denormalize all continuous variables"""
        assert len(cont_data) == len(self.cont_normalizers), "Normalizers are invalid"

        cont_data_denormalized = cont_data.copy()
        for name, normalizer in self.cont_normalizers.items():
            cont_data_denormalized[name] = normalizer.inverse_transform(cont_data[name.values])

        return cont_data_denormalized


class Preprocessor:
    """Preprocess the data for TFT.

    Dataset input table must follow this stucture where the table row defined
    follwowing
    [date, data_id, variable_1, variable_2, ..., variable_N]

    """

    id_col_name: str = "data_id"

    def __init__(self, data: pd.DataFrame, cfg: TFTConfig) -> None:
        self.data = data
        self.cfg = cfg

    def preprocess_data(self) -> pd.DataFrame:
        """Preprocess data"""
        data_id = self.data[self.id_col_name].unique()
        if isinstance(self.data["date"], pd.Timestamp):
            self.data["date"] = pd.to_datetime(self.data["date"])
        earliest_time = min(self.data["date"])
        seq_len = self.cfg.encoder_len + self.cfg.decoder_len

        merged_df = []
        for iden in data_id:
            # Extract all data relating to idenity
            tmp = self.data[self.data[self.id_col_name] == iden]

            # Resample data with the forecast freqency
            tmp.set_index("date", inplace=True)
            tmp = tmp.resample(f"{self.cfg.forecast_freq}S").mean().replace(0.0, np.nan)

            # Fill missing data
            tmp = self.fill_missing_data(raw_df=tmp)

            # Compute the start time index based on forecast frequency
            delta_time = tmp.index - earliest_time
            tmp["time_idx"] = delta_time.seconds / 60 / 60 + delta_time.days * (
                3600 / self.cfg.forecast_freq
            )

            # TODO: Should we provide a separe data frame with all time indices?
            tmp["time_idx"] = tmp["time_idx"].astype("int")
            tmp["time_first"] = tmp["time_idx"].iloc[0]
            tmp["time_last"] = tmp["time_idx"].iloc[-1]
            tmp["count"] = tmp["time_last"] - tmp["time_first"] + 1
            tmp["time_diff_to_next"] = tmp["time_idx"].diff(-1).fillna(-1)
            tmp["day_from_start"] = delta_time.days

            # Add time feature
            tmp["hour"] = tmp.index.hour
            tmp["day"] = tmp.index.day
            tmp["data_of_week"] = tmp.index.dayofweek
            tmp["month"] = tmp.index.month

            # Sort data and drop duplicate
            tmp.sort_values(inplace=True)
            tmp.drop_duplicates(inplace=True)

            # Add sequence indices
            tmp = self.add_sequence_index(data_frame=tmp, seq_len=seq_len, start_point=len(tmp))

            # Store data frame
            merged_df.append(tmp)

        merged_df = pd.concat(merged_df)
        merged_df.reset_index(inplace=True)

        return merged_df

    def _encode_cat_var(self) -> pd.DataFrame:
        """Convert all categorical columns into the numeric classes based on user-specified encoding
        methods.
        """
        # TODO: need to define a variable to contain encoding info for inversing transformation
        raise NotImplementedError

    def _normalize_cat_var(self) -> pd.DataFrame:
        """Normalize the continous variables based on user-specified normalization method"""
        # TODO: need to define a variable to contain normalization info for inversing transformation
        raise NotImplementedError

    @staticmethod
    def add_sequence_index(
        data_frame: pd.DataFrame, seq_len: int, start_point: int = 0
    ) -> pd.DataFrame:
        """Add start and end index for sequence"""
        # Copy dataframe
        added_data_frame = data_frame.copy()
        num_rows = len(data_frame)

        # Add start and end index
        added_data_frame["start_seq_idx"] = np.arange(num_rows) + start_point
        added_data_frame["end_seq_idx"] = np.arange(num_rows) + seq_len - 1

        # Clip indices going beyond num_rows
        added_data_frame["end_seq_idx"] = added_data_frame["end_seq_idx"].clip(upper=num_rows - 1)

        return added_data_frame

    @staticmethod
    def fill_missing_data(raw_df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing data"""
        filled_df = raw_df.copy()
        start_date = min(filled_df.fillna(method="ffill").dropna().index)
        end_date = max(filled_df.fillna(method="bfill").dropna().index)
        active_dt_range = (filled_df.index >= start_date) & (filled_df.index <= end_date)

        # Fill all the missing data outside of active range with nan
        filled_df = filled_df[active_dt_range].fillna(0.0)

        return filled_df
