from dataclasses import dataclass, field
from typing import Union, Dict, List


def empty_dict():
    return {}


@dataclass
class TFTConfig:
    """TFT configuration"""

    # Network hyperparameters
    # max_embedding_size: int
    # seq_len: int
    num_targets: int = 1
    num_epochs: int = 2
    batch_size: int = 64
    output_size: Union[List[int], int] = 7  # number of quantiles
    hidden_size: int = 16
    num_lstm_layers: int = 1
    num_attn_head_size: int = 3
    dropout: float = 0.1
    embedding_sizes: Union[Dict[str, Dict[str, int]], None] = None
    hidden_cont_sizes: Dict[str, int] = field(default_factory=empty_dict)
    hidden_cont_size: int = 8  # default
    decoder_len: int = 24
    encoder_len: int = 168
    lr: float = 0.003

    # Loss params
    quantiles: List[float] = field(default_factory=lambda: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98])

    # Data variables
    target_var: List[str] = field(default_factory=list)
    time_cat_features = ["hour", "day", "day_of_week", "month"]
    time_varying_cat_encoder: List[str] = field(default_factory=list)
    time_varying_cat_decoder: List[str] = field(default_factory=list)
    time_varying_cont_encoder: List[str] = field(default_factory=list)
    time_varying_cont_decoder: List[str] = field(default_factory=list)
    static_conts: List[str] = field(default_factory=list)
    static_cats: List[str] = field(default_factory=list)
    multi_cat_var: Dict[str, List[str]] = field(default_factory=empty_dict)
    is_single_var_grns_shared: bool = False
    forecast_freq: float = 3600

    cont_normalizing_method: Union[Dict[str, str], None] = None
    cat_encoding_method: Union[Dict[str, str], None] = None
    cont_transform_method: Union[Dict[str, str], None] = None
    device: str = "cuda"

    # Post user-specified variables. This is done after the initialization
    cont_var: List[str] = field(default_factory=list)
    cat_var: List[str] = field(default_factory=list)
    cat_var_ordering: Union[Dict[str, int], None] = None
    cont_var_ordering: Union[Dict[str, int], None] = None

    def __post_init__(self):
        """Post-initalize the config variables. Some variables will be defined after
        user-specifying the inputs"""
        self.cont_var = list(
            dict.fromkeys(
                self.time_varying_cont_decoder + self.time_varying_cont_encoder + self.static_conts
            )
        )
        self.cat_var = list(
            dict.fromkeys(
                self.time_varying_cat_decoder
                + self.time_varying_cat_encoder
                + self.static_cats
                + self.time_cat_features
            )
        )

        # Get ordering for variables in dataset
        self.cat_var_ordering = self._get_cat_var_ordering()
        self.cont_var_ordering = self._get_cont_var_ordering()

        # Get default methods for encoding and normalization methods for each data variables.
        # Support incompleted user-specified inputs.
        if self.cat_encoding_method is None or len(self.cat_encoding_method) < len(self.cat_var):
            self.cat_encoding_method = self._default_cat_encoding_method()

        if self.cont_normalizing_method is None or len(self.cont_normalizing_method) < len(
            self.cont_var
        ):
            self.cont_normalizing_method = self._default_cont_normalizer()

        # Get default methods for transformations
        if self.cont_transform_method is None:
            self.cont_transform_method = self._default_cont_transformation()

    @property
    def static_vars(self) -> List[str]:
        """Static variables"""
        return self.static_cats + self.static_conts

    @property
    def time_varying_encoder_vars(self) -> List[str]:
        """Time varying encoder variables"""
        return self.time_varying_cat_encoder + self.time_varying_cont_encoder

    @property
    def time_varying_decoder_vars(self) -> List[str]:
        """Time varying decoder variables"""
        return self.time_varying_cat_decoder + self.time_varying_cont_decoder

    @property
    def seq_len(self) -> int:
        """Sequence length"""

        return self.encoder_len + self.decoder_len

    def _get_cat_var_ordering(self) -> Dict[str, int]:
        """Get all categorical variables including multi-categorical variables.
        The ordering of categorical variable list will be the same with the data frame
        columns in preprocessing pipeline. This allows maintaining the ordering when
        feeding the input & output data to TFT network.
        """

        cat_vars = list(
            dict.fromkeys(
                self.time_varying_cat_decoder
                + self.time_varying_cat_encoder
                + self.static_cats
                + self.time_cat_features
            )
        )
        cat_var_cols = []
        for var in cat_vars:
            cat_var_cols.extend(self.multi_cat_var.get(var, []) if self.multi_cat_var else [var])

        cat_var_ordering: Dict[str, int] = {}
        for i, var in enumerate(cat_var_cols):
            cat_var_ordering[var] = i

        return cat_var_ordering

    def _get_cont_var_ordering(self) -> Dict[str, int]:
        """Get ordering for variables in data frame. This will be required in order to feed data to
        different network in TFT"""
        cont_vars = list(
            dict.fromkeys(
                self.time_varying_cont_decoder + self.time_varying_cont_encoder + self.static_conts
            )
        )
        cont_var_ordering: Dict[str, int] = {}
        for i, col in enumerate(cont_vars):
            cont_var_ordering[col] = i

        return cont_var_ordering

    def _default_cat_encoding_method(self) -> Dict[str, str]:
        """Initalize encoding method for each categorical variable. Default to `labelEncoder`"""

        cat_vars = list(
            dict.fromkeys(
                self.time_varying_cat_decoder
                + self.time_varying_cat_encoder
                + self.static_cats
                + self.time_cat_features
            )
        )
        cat_encoding_method = (
            self.cat_encoding_method.copy() if self.cat_encoding_method is not None else {}
        )
        for var in cat_vars:
            if var not in cat_encoding_method:
                cat_encoding_method[var] = "label_encoder"

        return cat_encoding_method

    def _default_cont_normalizer(self) -> Dict[str, str]:
        """Initialize normalization method for each continous variable. Default to `standard`"""

        cont_vars = list(
            dict.fromkeys(
                self.time_varying_cont_decoder + self.time_varying_cont_encoder + self.static_conts
            )
        )
        cont_normalizing_method = (
            self.cont_normalizing_method.copy() if self.cont_normalizing_method is not None else {}
        )
        for var in cont_vars:
            if var not in cont_normalizing_method:
                cont_normalizing_method[var] = "standard"

        return cont_normalizing_method

    def _default_cont_transformation(self) -> Dict[str, str]:
        """Initalize the transformation method fo each continous variables.Default to None"""
        cont_vars = list(
            dict.fromkeys(
                self.time_varying_cont_decoder + self.time_varying_cont_encoder + self.static_conts
            )
        )
        cont_transform_method = (
            self.cont_transform_method.copy() if self.cont_transform_method is not None else {}
        )
        for var in cont_vars:
            if var not in cont_transform_method:
                cont_transform_method[var] = "no_transform"

        return cont_transform_method
