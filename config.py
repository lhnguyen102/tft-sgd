from dataclasses import dataclass, field
from typing import Union, Dict, Tuple, List


@dataclass
class TFTConfig:
    """TFT configuration"""

    # Network hyperparameters
    max_embedding_size: int
    seq_len: int
    num_targets: int = 1
    batch_size: int = 8
    output_size: Union[List[int], int] = 7  # number of quantiles
    hidden_size: int = 16
    num_lstm_layers: int = 1
    num_attn_head_size: int = 4
    dropout: float = 0.1
    embedding_sizes: Union[
        Dict[str, Tuple[int, int]],
        Dict[str, int],
        List[int],
        List[Tuple[int, int]],
        None,
    ] = None
    embedding_paddings: Union[List[str], None] = None
    x_cats: Union[List[str], None] = None
    hidden_cont_sizes: Union[Dict[str, int], None] = None
    hidden_cont_size: int = 8  # default
    decoder_len: int = 24
    encoder_len: int = 168

    # Data variables
    time_varying_cat_encoder: Union[List[str], None] = None
    time_varying_cat_decoder: Union[List[str], None] = None
    time_varying_real_encoder: Union[List[str], None] = None
    time_varying_real_decoder: Union[List[str], None] = None
    static_reals: Union[List[str], None] = None
    static_cats: Union[List[str], None] = None
    multi_cat_var: Union[Dict[str, List[str]], None] = None
    is_single_var_grns_shared: bool = False
    forecast_freq: float = 3600

    cont_normalizers: Union[Dict[str, str], None] = None
    cat_encoders: Union[Dict[str, str], None] = None

    # Post user-specified variables. This is done after the initialization
    target_var: List[str] = field(default_factory=list)
    cont_var: List[str] = field(default_factory=list)
    cat_var: List[str] = field(default_factory=list)
    cont_var_ordering: List[str] = field(default_factory=list)
    cat_var_ordering: List[str] = field(default_factory=list)

    def __post__init__(self):
        """Post-initalize the config variables. Some variables will be defined after
        user-specifying the inputs"""
        self.cont_var = (
            self.time_varying_real_decoder + self.time_varying_real_encoder + self.static_reals
        )  # TODO: add unique varaibles
        self.cat_var = (
            self.time_varying_cat_decoder + self.time_varying_cat_encoder + self.static_cats
        )
        self.cat_var_ordering = self._get_cat_var_ordering()
        self.cont_var_ordering = self.cont_var

        # Get default methods for encoding and normalization methods for each data variables.
        # Support incompleted user-specified inputs.
        if self.cat_encoders is None or len(self.cat_encoders) < len(self.cat_var):
            self.cat_encoders = self._default_cat_encoders()
        if self.cont_normalizers is None or len(self.cont_normalizers) < len(self.cont_var):
            self.cont_normalizers = self._default_cont_normalizer()

    def _get_cat_var_ordering(self) -> List[str]:
        """Get all categorical variables including multi-categorical variables.
        The ordering of categorical variable list will be the same with the data frame
        columns in preprocessing pipeline. This allows maintaining the ordering when
        feeding the input & output data to TFT network.
        """
        cat_vars = self.time_varying_cat_decoder + self.time_varying_cat_encoder + self.static_cats
        cat_var_ordering = []
        for var in cat_vars:
            if var in self.multi_cat_var:
                cat_var_ordering.extend(self.multi_cat_var[var])
            else:
                cat_var_ordering.append(var)

        return cat_var_ordering

    def _default_cat_encoders(self) -> List[str]:
        """Get encoding method for each categorical variable. Default to `labelEncoder`"""
        cat_vars = self.time_varying_cat_decoder + self.time_varying_cat_encoder + self.static_cats
        cat_encoders = self.cat_encoders or {}
        for var in cat_vars:
            if var not in cat_encoders:
                cat_encoders[var] = "labelEncoder"

        return cat_encoders

    def _default_cont_normalizer(self) -> List[str]:
        """Get normalization method for each continous variable. Default to `standard`"""
        cont_vars = (
            self.time_varying_real_decoder + self.time_varying_real_encoder + self.static_reals
        )
        cont_normalizers = self.cont_normalizers or {}
        for var in cont_vars:
            if var not in cont_normalizers:
                cont_normalizers[var] = "standard"

        return cont_normalizers
