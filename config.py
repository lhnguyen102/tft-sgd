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
    categorical_groups: Union[Dict[str, List[str]], None] = None
    embedding_sizes: Union[
        Dict[str, Tuple[int, int]], Dict[str, int], List[int], List[Tuple[int, int]], None
    ] = None
    embedding_paddings: Union[List[str], None] = None
    x_categoricals: Union[List[str], None] = None
    hidden_continuous_sizes: Union[Dict[str, int], None] = None
    hidden_continuous_size: int = 8  # default
    decoder_len: int = 24
    encoder_len: int = 168

    # Data variables
    time_varying_categorical_encoder: Union[List[str], None] = None
    time_varying_categorical_decoder: Union[List[str], None] = None
    time_varying_real_encoder: Union[List[str], None] = None
    time_varying_real_decoder: Union[List[str], None] = None
    static_reals: Union[List[str], None] = None
    static_categoricals: Union[List[str], None] = None
    is_single_var_grns_shared: bool = False
    forecast_freq: float = 3600

    # Post user-specified variables
    target_var: List[str] = field(default_factory=list)
    cont_var: List[str] = field(default_factory=list)
    cat_var: List[str] = field(default_factory=list)

    def __post__init__(self):
        self.cont_var = (
            self.time_varying_real_decoder + self.time_varying_real_encoder + self.static_reals
        )
        self.cat_var = (
            self.time_varying_categorical_decoder
            + self.time_varying_categorical_encoder
            + self.static_categoricals
        )
        self.seq_len = self.decoder_len + self.encoder_len
