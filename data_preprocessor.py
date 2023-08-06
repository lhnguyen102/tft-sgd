from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch

from config import TFTConfig


@dataclass
class AutoencoderInputBatch:
    """Minitatch for encoder decoder"""

    cont_var: torch.Tensor = None
    cat_var: torch.Tensor = None
    encoder_len: int = None
    decoder_len: int = None


@dataclass
class TFTTargetBatch:
    """Output batch for TFT model"""

    target: torch.Tensor = None


class TFTDataloader:
    """Custom dataloader for time series"""

    def __init__(self, data: pd.DataFrame, cfg: TFTConfig, shuffle: bool = False) -> None:
        self.data = data
        self.cfg = cfg
        self.shuffle = shuffle
        self.indices = np.arange(len(self.data) - self.cfg.seq_len).tolist()
        self.num_samples = 0

    def reshuffle(self):
        """Reshuffle indices"""
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self) -> int:
        return len(self.indices) // self.cfg.batch_size

    def __getitem__(self, index: int) -> Tuple[AutoencoderInputBatch, TFTTargetBatch]:
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
            start_seq_idx = int(self.data.iloc[i]["start_seq_idx"])
            end_seq_idx = int(self.data.iloc[i]["end_seq_idx"])

            # Get data from data frame. This will ensure the ordering for both categorical and
            # continous variables when feeding to TFT network
            cont_batches.append(
                torch.tensor(
                    self.data.iloc[start_seq_idx:end_seq_idx][self.cfg.cont_var].to_numpy(
                        dtype=np.float32
                    ),
                    dtype=torch.float32,
                )
            )
            cat_batches.append(
                torch.tensor(
                    self.data.iloc[start_seq_idx:end_seq_idx][self.cfg.cat_var].to_numpy(
                        dtype=np.int64
                    ),
                    dtype=torch.int64,
                )
            )
            target_batches.append(
                torch.tensor(
                    self.data.iloc[start_seq_idx + self.cfg.encoder_len : end_seq_idx][
                        self.cfg.target_var
                    ].to_numpy(dtype=np.int64),
                    dtype=torch.int64,
                )
            )
        input_batch = AutoencoderInputBatch(
            cont_var=torch.stack(cont_batches),
            cat_var=torch.stack(cat_batches),
            encoder_len=self.cfg.encoder_len,
            decoder_len=self.cfg.decoder_len,
        )
        output_batch = TFTTargetBatch(target=torch.stack(target_batches))

        return input_batch, output_batch

    def __iter__(self):
        self.reshuffle()
        self.num_samples = 0
        return self

    def __next__(self) -> Tuple[AutoencoderInputBatch, TFTTargetBatch]:
        if self.num_samples < len(self):
            result = self.__getitem__(self.num_samples)
            self.num_samples += 1
            return result
        else:
            raise StopIteration


class TransformationFun:
    """Transformation function of data"""

    def __init__(self, method: str) -> None:
        self.method = method

    @property
    def method(self) -> str:
        """Get transformation method"""
        return self._method

    @method.setter
    def method(self, value: str) -> None:
        """Set transformation method"""
        self._method = value
        self.transformtation = {
            "softplus": (self.softplus, self.inverse_softplus),
            "no_transform": (self.no_transform, self.no_transform),
        }

        try:
            self.transform_fn, self.inverse_fn = self.transformtation[self._method]
        except KeyError:
            raise ValueError(f"Unknown transformation method: {self._method}")

    def transform(self, x_original) -> np.ndarray:
        """Original to transformed spaced"""
        return self.transform_fn(x_original)

    def inverse(self, x_transformed) -> np.ndarray:
        """Transformed to original space"""
        return self.inverse_fn(x_transformed)

    @staticmethod
    def softplus(x_original: np.ndarray) -> np.ndarray:
        """Transform orginal value to transformed space using softplus"""
        cutoff = 20
        return np.where(x_original > cutoff, x_original, np.log1p(np.exp(x_original)))

    @staticmethod
    def inverse_softplus(x_transformed: np.ndarray) -> np.ndarray:
        """Transform the transformed space to original space using inverse sofplus"""
        cutoff = 20
        return np.where(x_transformed > cutoff, x_transformed, np.log(np.expm1(x_transformed)))

    @staticmethod
    def no_transform(x_original: np.ndarray) -> np.ndarray:
        """No transformation"""
        return x_original


class BaseNormalizer(ABC):
    """Base class for normalizer"""

    def __init__(self) -> None:
        pass

    @abstractmethod
    def fit(self, values: np.ndarray):
        pass

    @abstractmethod
    def transform(self, original_values: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def fit_transform(self, original_values: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def inverse_transform(self, scaled_values: np.ndarray) -> np.ndarray:
        pass


class StandardNormalizer(BaseNormalizer):
    """Standard normalization of the data"""

    def __init__(self) -> None:
        self.mean_: np.ndarray = None
        self.scale_: np.ndarray = None

    def fit(self, values: np.ndarray) -> "StandardNormalizer":
        self.mean_ = np.nanmean(values, axis=0, keepdims=True)
        self.scale_ = np.nanstd(values, axis=0, keepdims=True)
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

    @property
    def num_classes(self) -> int:
        """Get number of classes"""
        return len(self.labels)

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
        cont_transform_method: Dict[str, str],
    ) -> None:
        self.cat_var = cat_var
        self.cont_var = cont_var
        self.cat_encoding_method = cat_encoding_method
        self.cont_normalizing_method = cont_normalizing_method
        self.cont_transform_method = cont_transform_method

    @property
    def cat_encoding_method(self) -> Dict[str, str]:
        """Get a dictionary of encoding methods for categorical variables"""

        return self._cat_encoding_method

    @cat_encoding_method.setter
    def cat_encoding_method(self, value: Dict[str, str]) -> None:
        """Set a dictionary of encoding methods for categorical variables"""

        self._cat_encoding_method = value
        self.cat_encoders: Dict[str, LabelEncoder] = {}
        for name, method in self._cat_encoding_method.items():
            if method == "label_encoder":
                self.cat_encoders[name] = LabelEncoder()
            else:
                raise ValueError(f"Unknown encoding method for [{name}]: {method}")

    @property
    def cont_normalizing_method(self) -> Dict[str, str]:
        """Get a dictionary of normalizing methods for categorical variables"""
        return self._cont_normalizing_method

    @cont_normalizing_method.setter
    def cont_normalizing_method(self, value: Dict[str, str]) -> None:
        """Set a dictionary of normalizing methods for categorical variables"""

        self._cont_normalizing_method = value
        self.cont_normalizers: Dict[str, BaseNormalizer] = {}
        for name, method in self._cont_normalizing_method.items():
            if method == "standard":
                self.cont_normalizers[name] = StandardNormalizer()
            else:
                raise ValueError(f"Unknown normalizing method for [{name}]: {method}")

    @property
    def cont_transform_method(self) -> Dict[str, str]:
        """Get the transform method"""
        return self._cont_transform_method

    @cont_transform_method.setter
    def cont_transform_method(self, value: Dict[str, str]) -> None:
        """Set the transform method"""
        self._cont_transform_method = value
        self.cont_transfom_fn: Dict[str, TransformationFun] = {}

        for name, method in self._cont_transform_method.items():
            self.cont_transfom_fn[name] = TransformationFun(method)

    def encode_cat_var(self, cat_data: pd.DataFrame) -> pd.DataFrame:
        """Encode all categorical variables"""
        assert cat_data.shape[1] == len(self.cat_encoders), "Categorical encoders are invalid"
        cat_data_encoded = cat_data.copy()

        for name, encoder in self.cat_encoders.items():
            cat_data_encoded[name] = encoder.fit_transform(cat_data[name].values)

        return cat_data_encoded

    def decode_cat_var(self, cat_data: pd.DataFrame) -> pd.DataFrame:
        """Decode all categorical variables"""

        assert cat_data.shape[1] == len(self.cat_encoders), "Categorical encoders are invalid"
        cat_data_decoded = cat_data.copy()

        for name, encoder in self.cat_encoders.items():
            assert len(encoder.labels) > 0, f"{name} does not have encoder"
            cat_data_decoded[name] = encoder.inverse_transform(cat_data[name].values)

        return cat_data_decoded

    def encode_multi_cat_var(
        self, cat_data: pd.DataFrame, multi_cat_var: Dict[str, List[str]]
    ) -> pd.DataFrame:
        """Encode all mutli-categorical variables"""

        cat_data_encoded = cat_data.copy()
        for name, cat_var in multi_cat_var.items():
            self.cat_encoders[name].fit(cat_var)
            for var in cat_var:
                cat_data_encoded[var] = self.cat_encoders[name].transform(cat_data[var].values)

        return cat_data_encoded

    def decode_multi_cat_var(
        self, cat_data: pd.DataFrame, multi_cat_var: Dict[str, List[str]]
    ) -> pd.DataFrame:
        """Decode all multi-categorical variables"""

        cat_data_decoded = cat_data.copy()
        for name, cat_var in multi_cat_var.items():
            for var in cat_var:
                cat_data_decoded[var] = self.cat_encoders[name].inverse_transform(
                    cat_data[var].values
                )

        return cat_data_decoded

    def normalize(self, cont_data: pd.DataFrame) -> pd.DataFrame:
        """Normalize all continuous variables"""
        assert cont_data.shape[1] == len(self.cont_normalizers), "Normalizers are invalid"
        assert cont_data.shape[1] == len(self.cont_transfom_fn), "Transform fn are invalid"

        cont_data_normalized = cont_data.copy()
        for name, normalizer in self.cont_normalizers.items():
            transform_values = self.cont_transfom_fn[name].transform(cont_data[name].values)
            cont_data_normalized[name] = normalizer.fit_transform(transform_values)

        return cont_data_normalized

    def denormalize(self, cont_data: pd.DataFrame) -> pd.DataFrame:
        """Denormalize all continuous variables"""
        assert cont_data.shape[1] == len(self.cont_normalizers), "Normalizers are invalid"
        assert cont_data.shape[1] == len(self.cont_transfom_fn), "Transform fn are invalid"

        cont_data_denormalized = cont_data.copy()
        for name, normalizer in self.cont_normalizers.items():
            inverse_values = self.cont_transfom_fn[name].inverse(cont_data[name.values])
            cont_data_denormalized[name] = normalizer.inverse_transform(inverse_values)

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
        self.normalizer_encoder = TimeseriesEncoderNormalizer(
            cat_var=self.cfg.cat_var,
            cont_var=self.cfg.cont_var,
            cat_encoding_method=self.cfg.cat_encoding_method,
            cont_normalizing_method=self.cfg.cont_normalizing_method,
            cont_transform_method=self.cfg.cont_transform_method,
        )

    def preprocess_data(self) -> pd.DataFrame:
        """Preprocess data"""
        data_id = self.data[self.id_col_name].unique()
        if isinstance(self.data["date"], pd.Timestamp):
            self.data["date"] = pd.to_datetime(self.data["date"])
        earliest_time = min(self.data["date"])
        seq_len = self.cfg.encoder_len + self.cfg.decoder_len

        merged_df = []
        start_point = 0
        for iden in data_id:
            # Extract all data relating to idenity
            tmp = self.data[self.data[self.id_col_name] == iden]
            tmp.set_index("date", inplace=True)
            tmp.sort_index(inplace=True)

            # Fill missing data
            tmp = self.fill_missing_data(raw_df=tmp)

            # Resample data with the forecast frequency
            cont_resampled = (
                tmp[self.cfg.cont_var]
                .resample(f"{self.cfg.forecast_freq}S")
                .mean()
                .replace(0.0, np.nan)
            )

            cat_resampled = self._resample_cat_var(tmp)

            tmp = cat_resampled.join(cont_resampled)

            # Compute the start time index based on forecast frequency
            delta_time = tmp.index - earliest_time

            # Construcuted time index
            tmp = self._construct_time_idx(
                data_frame=tmp,
                forecast_freq=self.cfg.forecast_freq,
                delta_time=delta_time,
            )

            # Add time feature
            tmp = self.add_time_features(data_frame=tmp, time_features=self.cfg.time_cat_features)

            # Sort data and drop duplicate
            tmp.sort_index(inplace=True)
            tmp.drop_duplicates(inplace=True)

            # Add sequence indices
            tmp = self.add_sequence_index(data_frame=tmp, seq_len=seq_len, start_point=start_point)

            # Update start point
            start_point += len(tmp)

            # Store data frame
            merged_df.append(tmp)
            del tmp

        merged_df = pd.concat(merged_df)
        # merged_df["ordering"] = np.arange(0, len(merged_df))
        merged_df.reset_index(inplace=True)

        # Encode categorical variables
        cat_df = self._encode_cat_var(merged_df)

        # Update the embedding size
        self._update_embedding_size()

        # Normalize continous variables
        norm_df = self._normalize_con_var(merged_df)

        # Update data frame
        merged_df.update(cat_df)
        merged_df.update(norm_df)

        return merged_df

    def _update_embedding_size(self) -> None:
        """Update the embedding size based on the number of classes in raw data if users do
        specify"""
        if self.cfg.embedding_sizes is None:
            self.cfg.embedding_sizes: Dict[str, Dict[str, int]] = {}
            for name in self.cfg.cat_var:
                num_classes = self.normalizer_encoder.cat_encoders[name].num_classes
                self.cfg.embedding_sizes[name] = {
                    "num_classes": num_classes,
                    "emb_size": self._calculate_embedding_size(num_classes),
                }

    def _calculate_embedding_size(self, num_categories: int) -> int:
        """
        Calculate the embedding size based on the number of categories using fast.ai heuristic.

        Args:
            num_categories (int): Number of unique categories.

        Returns:
            int: Embedding size.
        """

        return min(600, round(1.6 * num_categories**0.56))

    def _resample_cat_var(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Resample the time series data at given frequency"""

        cat_var_cols = [col for col in data_frame.columns if col in self.cfg.cat_var]

        if cat_var_cols:
            df_resampled = data_frame[cat_var_cols].resample(f"{self.cfg.forecast_freq}S").first()
        else:
            df_resampled = data_frame.copy()

        return df_resampled

    def _encode_cat_var(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Convert all categorical columns into the numeric classes based on user-specified encoding
        methods.
        """
        # Single categorical variables
        single_cat_var = [item for item in self.cfg.cat_var if item not in self.cfg.multi_cat_var]
        single_cat_df = self.normalizer_encoder.encode_cat_var(data_frame[single_cat_var])

        # Multi-categorical variables
        if len(self.cfg.multi_cat_var) > 0:
            multi_cat_merged = []
            for _, item in self.cfg.multi_cat_var.items():
                multi_cat_merged.extend(item)

            multi_cat_df = self.normalizer_encoder.encode_multi_cat_var(
                data_frame[multi_cat_merged], multi_cat_var=self.cfg.multi_cat_var
            )

            return pd.merge(single_cat_df, multi_cat_df, left_index=True, right_index=True)
        else:
            return single_cat_df

    def _normalize_con_var(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Normalize the continous variables based on user-specified normalization method"""

        norm_data_frame = self.normalizer_encoder.normalize(data_frame[self.cfg.cont_var])

        return norm_data_frame

    @staticmethod
    def _construct_time_idx(
        data_frame: pd.DataFrame, forecast_freq: float, delta_time: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """Construct time index"""
        indexed_df = data_frame.copy()
        indexed_df["time_idx"] = delta_time.seconds / 60 / 60 + delta_time.days * 24 * (
            3600 / forecast_freq
        )
        indexed_df["time_idx"] = indexed_df["time_idx"].astype("int")
        indexed_df["time_first"] = indexed_df["time_idx"].iloc[0]
        indexed_df["time_last"] = indexed_df["time_idx"].iloc[-1]
        indexed_df["count"] = indexed_df["time_last"] - indexed_df["time_first"] + 1
        indexed_df["time_diff_to_next"] = -indexed_df["time_idx"].diff(-1).fillna(-1)
        indexed_df["day_from_start"] = delta_time.days

        return indexed_df

    @staticmethod
    def add_time_features(data_frame: pd.DataFrame, time_features: List[str]) -> pd.DataFrame:
        """Add time feature to data frame"""

        featured_df = data_frame.copy()
        for feature in time_features:
            if feature == "hour":
                featured_df[feature] = data_frame.index.hour
            elif feature == "day":
                featured_df[feature] = data_frame.index.day
            elif feature == "month":
                featured_df[feature] = data_frame.index.month
            elif feature == "day_of_week":
                featured_df[feature] = data_frame.index.dayofweek
            elif feature == "year":
                featured_df[feature] = data_frame.index.year
            elif feature == "day_of_year":
                featured_df[feature] = data_frame.index.dayofyear
            else:
                raise ValueError(f"Unknown time feature: {feature}")
        return featured_df

    @staticmethod
    def add_sequence_index(
        data_frame: pd.DataFrame, seq_len: int, start_point: int = 0
    ) -> pd.DataFrame:
        """Add start and end index for sequence"""
        # Copy dataframe
        added_data_frame = data_frame.copy()
        num_rows = len(data_frame)

        # Add start and end index
        added_data_frame["start_seq_idx"] = np.arange(num_rows)
        added_data_frame["end_seq_idx"] = np.arange(num_rows) + seq_len

        # Clip indices going beyond num_rows
        added_data_frame["end_seq_idx"] = added_data_frame["end_seq_idx"].clip(upper=num_rows - 1)
        idx = (added_data_frame["end_seq_idx"] - added_data_frame["start_seq_idx"]) == seq_len
        added_data_frame = added_data_frame[idx]

        added_data_frame["start_seq_idx"] += start_point
        added_data_frame["end_seq_idx"] += start_point

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
