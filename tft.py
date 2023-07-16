from typing import Dict
import torch
import torch.nn as nn
from model import (
    GatedResidualNetwork,
    VariableSelectionNetwork,
    GatedLinearUnit,
    AddNorm,
    MultiHeadAttention,
    GateAddNorm,
)
from embeddings import MultiEmbedding
from rnn import LSTM
from config import TFTConfig


class TemporalFusionTransformer(nn.Module):
    """Temporal Fusion Transformer approach"""

    def __init__(self, cfg: TFTConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Input embeddings
        self.input_embeddings = MultiEmbedding(
            embedding_sizes=self.cfg.embedding_sizes,
            categorical_groups=self.cfg.categorical_groups,
            embedding_paddings=self.cfg.embedding_paddings,
            x_categoricals=self.cfg.x_categoricals,
            max_embedding_size=self.cfg.max_embedding_size,
        )

        # Prescaler
        self.prescalers = nn.ModuleDict(
            {
                name: nn.Linear(
                    1, self.cfg.hidden_continuous_sizes.get(name, self.cfg.hidden_continuous_size)
                )
                for name in self.cfg.reals
            }
        )

        # Variable selection for static variables
        static_input_sizes = {
            name: self.input_embeddings.output_size[name] for name in self.cfg.static_categoricals
        }
        static_input_sizes.update(
            {
                name: self.cfg.hidden_continuous_sizes.get(name, self.cfg.hidden_continuous_size)
                for name in self.cfg.static_reals
            }
        )

        self.static_variable_selection = VariableSelectionNetwork(
            input_sizes=static_input_sizes,
            hidden_size=self.cfg.hidden_size,
            input_embedding_flags={name: True for name in self.cfg.static_categoricals},
            dropout=self.cfg.dropout,
            prescalers=self.prescalers,
        )

        # Input size for encoder and decoder
        encoder_input_sizes = {
            name: self.input_embeddings.output_size[name]
            for name in self.cfg.time_varying_categorical_encoder
        }
        encoder_input_sizes.update(
            {
                name: self.cfg.hidden_continuous_sizes.get(name, self.cfg.hidden_continuous_size)
                for name in self.cfg.time_varying_real_encoder
            }
        )

        decoder_input_sizes = {
            name: self.input_embeddings.output_size[name]
            for name in self.cfg.time_varying_categorical_decoder
        }
        decoder_input_sizes.update(
            {
                name: self.cfg.hidden_continuous_sizes.get(name, self.cfg.hidden_continuous_size)
                for name in self.cfg.time_varying_real_decoder
            }
        )

        # Single variable grns that are shared across decoder and encoder
        if self.cfg.is_single_var_grns_shared:
            self.shared_single_var_grns = nn.ModuleDict()
            for name, input_size in encoder_input_sizes.items():
                self.shared_single_var_grns[name] = GatedResidualNetwork(
                    input_size=input_size,
                    hidden_size=min(input_size, self.cfg.hidden_size),
                    output_size=self.cfg.hidden_size,
                    dropout=self.cfg.dropout,
                )

            for name, input_size in decoder_input_sizes.items():
                if name not in self.shared_single_var_grns:
                    self.shared_single_var_grns[name] = GatedResidualNetwork(
                        input_size=input_size,
                        hidden_size=min(input_size, self.cfg.hidden_size),
                        output_size=self.cfg.hidden_size,
                        dropout=self.cfg.dropout,
                    )

        # Vairable selection for encoder
        self.encoder_variable_selection = VariableSelectionNetwork(
            input_sizes=encoder_input_sizes,
            hidden_size=self.cfg.hidden_size,
            input_embedding_flags={
                name: True for name in self.cfg.time_varying_categorical_encoder
            },
            dropout=self.cfg.dropout,
            context_size=self.cfg.hidden_size,
            prescalers=self.prescalers,
            single_variable_grns={}
            if not self.cfg.is_single_var_grns_shared
            else self.shared_single_var_grns,
        )

        # Variable selection for decoder
        self.decoder_variable_selection = VariableSelectionNetwork(
            input_sizes=decoder_input_sizes,
            hidden_size=self.cfg.hidden_size,
            input_embedding_flags={
                name: True for name in self.cfg.time_varying_categorical_decoder
            },
            dropout=self.cfg.dropout,
            context_size=self.cfg.hidden_size,
            prescalers=self.prescalers,
            single_variable_grns={}
            if not self.cfg.is_single_var_grns_shared
            else self.shared_single_var_grns,
        )

        # Static encoders
        # for variable selection
        self.static_context_variable_selection = GatedResidualNetwork(
            input_size=self.cfg.hidden_size,
            hidden_size=self.cfg.hidden_size,
            output_size=self.cfg.hidden_size,
            dropout=self.cfg.dropout,
        )

        # for hidden state of the lstm
        self.static_context_initial_hidden_lstm = GatedResidualNetwork(
            input_size=self.cfg.hidden_size,
            hidden_size=self.cfg.hidden_size,
            output_size=self.cfg.hidden_size,
            dropout=self.cfg.dropout,
        )

        # for cell state of the lstm
        self.static_context_initial_cell_lstm = GatedResidualNetwork(
            input_size=self.cfg.hidden_size,
            hidden_size=self.cfg.hidden_size,
            output_size=self.cfg.hidden_size,
            dropout=self.cfg.dropout,
        )

        # for static enrichment
        self.static_context_enrichment = GatedResidualNetwork(
            input_size=self.cfg.hidden_size,
            hidden_size=self.cfg.hidden_size,
            output_size=self.cfg.hidden_size,
            dropout=self.cfg.dropout,
        )

        # Encoder-Decoder LSTM
        self.lstm_encoder = LSTM(
            input_size=self.cfg.hidden_size,
            hidden_size=self.cfg.hidden_size,
            num_layers=self.cfg.num_lstm_layers,
            dropout=self.cfg.dropout if self.cfg.num_lstm_layers > 1 else 0,
            batch_first=True,
        )

        self.lstm_decoder = LSTM(
            input_size=self.cfg.hidden_size,
            hidden_size=self.cfg.hidden_size,
            num_layers=self.cfg.num_lstm_layers,
            dropout=self.cfg.dropout if self.cfg.num_lstm_layers > 1 else 0,
            batch_first=True,
        )

        # LSTM skip connection
        self.post_lstm_gate_encoder = GatedLinearUnit(
            input_size=self.cfg.hidden_size,
            hidden_size=self.cfg.hidden_size,
            dropout=self.cfg.dropout,
        )
        self.post_lstm_gate_decoder = self.post_lstm_gate_encoder
        self.post_lstm_add_norm_encoder = AddNorm(
            input_size=self.cfg.hidden_size, skipe_size=self.cfg.hidden_size, trainable_add=False
        )
        self.post_lstm_add_norm_decoder = self.post_lstm_add_norm_encoder

        # Static enrichment
        self.static_enrichment = GatedResidualNetwork(
            input_size=self.cfg.hidden_size,
            hidden_size=self.cfg.hidden_size,
            output_size=self.cfg.hidden_size,
            dropout=self.cfg.dropout,
            context_size=self.cfg.hidden_size,
        )

        # Multi-head self-attention
        self.multihead_attn = MultiHeadAttention(
            n_head=self.cfg.num_attn_head_size,
            d_model=self.cfg.hidden_size,
            dropout=self.cfg.dropout,
        )
        self.post_attn_gate_norm = GateAddNorm(
            input_size=self.cfg.hidden_size,
            hidden_size=self.cfg.hidden_size,
            skip_size=self.cfg.hidden_size,
            dropout=self.cfg.dropout,
            trainable_add=False,
        )
        self.pos_wise_ff = GatedResidualNetwork(
            input_size=self.cfg.hidden_size,
            hidden_size=self.cfg.hidden_size,
            output_size=self.cfg.hidden_size,
            dropout=self.cfg.dropout,
        )

        # Prediction layer
        self.pre_output_gate_norm = GateAddNorm(
            input_size=self.cfg.hidden_size,
            hidden_size=self.cfg.hidden_size,
            skip_size=self.cfg.hidden_size,
            dropout=None,
            trainable_add=False,
        )

        if self.cfg.num_targets > 1:  # Multiple targets
            self.output_layers = nn.ModuleList(
                [
                    nn.Linear(self.cfg.hidden_size, output_size)
                    for output_size in self.cfg.output_size
                ]
            )
        else:
            self.output_layer = nn.Linear(self.cfg.hidden_size, self.cfg.output_size)

    def forward(self, observation: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Observation shape must be (batch_size, time step, covariates)"""
