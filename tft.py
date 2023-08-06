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
from data_preprocessor import AutoencoderInputBatch
from dataclasses import dataclass


@dataclass
class TFTOutput:
    """TFF network's output"""

    prediction: torch.Tensor
    encoder_attn_weight: torch.Tensor
    decoder_attn_weight: torch.Tensor
    static_var_weight: torch.Tensor
    encoder_var_selection_weight: torch.Tensor
    decoder_var_selection_weight: torch.Tensor


class TemporalFusionTransformer(nn.Module):
    """Temporal Fusion Transformer approach"""

    def __init__(self, cfg: TFTConfig, dtype=torch.float32) -> None:
        super().__init__()
        self.cfg = cfg
        self.dtype = dtype
        if self.cfg.device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Input embeddings
        self.input_embeddings = MultiEmbedding(
            embedding_sizes=self.cfg.embedding_sizes,
            cat_var=self.cfg.cat_var,
            cat_var_ordering=self.cfg.cat_var_ordering,
            multi_cat_var=self.cfg.multi_cat_var,
        )

        # Prescaler
        self.prescalers = nn.ModuleDict()
        for name in self.cfg.cont_var:
            output_size = self.cfg.hidden_cont_sizes.get(name, self.cfg.hidden_cont_size)
            self.prescalers[name] = nn.Linear(1, output_size)

        # Variable selection for static variables
        static_input_sizes = {
            name: self.cfg.embedding_sizes[name]["emb_size"] for name in self.cfg.static_cats
        }
        for name in self.cfg.static_conts:
            size = self.cfg.hidden_cont_sizes.get(name, self.cfg.hidden_cont_size)
            static_input_sizes[name] = size

        self.static_variable_selection = VariableSelectionNetwork(
            input_sizes=static_input_sizes,
            hidden_size=self.cfg.hidden_size,
            input_embedding_flags={name: True for name in self.cfg.static_cats},
            dropout=self.cfg.dropout,
            prescalers=self.prescalers,
        )

        # Input size for encoder and decoder
        encoder_input_sizes = {
            name: self.cfg.embedding_sizes[name]["emb_size"]
            for name in self.cfg.time_varying_cat_encoder
        }
        for name in self.cfg.time_varying_cont_encoder:
            size = self.cfg.hidden_cont_sizes.get(name, self.cfg.hidden_cont_size)
            encoder_input_sizes[name] = size

        decoder_input_sizes = {
            name: self.cfg.embedding_sizes[name]["emb_size"]
            for name in self.cfg.time_varying_cat_decoder
        }
        for name in self.cfg.time_varying_cont_decoder:
            size = self.cfg.hidden_cont_sizes.get(name, self.cfg.hidden_cont_size)
            decoder_input_sizes[name] = size

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

        # Variable selection for encoder
        single_variable_grns = (
            {} if not self.cfg.is_single_var_grns_shared else self.shared_single_var_grns
        )
        self.encoder_variable_selection = VariableSelectionNetwork(
            input_sizes=encoder_input_sizes,
            hidden_size=self.cfg.hidden_size,
            input_embedding_flags={name: True for name in self.cfg.time_varying_cat_encoder},
            dropout=self.cfg.dropout,
            context_size=self.cfg.hidden_size,
            prescalers=self.prescalers,
            single_variable_grns=single_variable_grns,
        )

        # Variable selection for decoder
        self.decoder_variable_selection = VariableSelectionNetwork(
            input_sizes=decoder_input_sizes,
            hidden_size=self.cfg.hidden_size,
            input_embedding_flags={name: True for name in self.cfg.time_varying_cat_decoder},
            dropout=self.cfg.dropout,
            context_size=self.cfg.hidden_size,
            prescalers=self.prescalers,
            single_variable_grns=single_variable_grns,
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
        self.post_gate_encoder = GatedLinearUnit(
            input_size=self.cfg.hidden_size,
            hidden_size=self.cfg.hidden_size,
            dropout=self.cfg.dropout,
        )
        self.post_gate_decoder = self.post_gate_encoder
        self.post_add_norm_encoder = AddNorm(
            input_size=self.cfg.hidden_size,
            skipe_size=self.cfg.hidden_size,
            trainable_add=False,
        )
        self.post_add_norm_decoder = self.post_add_norm_encoder

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

    def forward(self, observation: AutoencoderInputBatch) -> Dict[str, torch.Tensor]:
        """Observation shape must be (batch_size, time step, covariates)"""
        # Sequence length
        seq_len = observation.encoder_len + observation.decoder_len

        # Transform the categorical to the continuous space
        hidden_states = self.input_embeddings(observation.cat_var)

        # Add continous data to hidden states
        hidden_states.update(
            {
                name: observation.cont_var[..., [i]]
                for name, i in self.cfg.cont_var_ordering.items()
                if name in self.cfg.cont_var
            }
        )

        # We assume that static embedding valuea are constant across time step
        if len(self.cfg.static_vars) > 0:
            static_emb_input = {name: hidden_states[name][:, 0] for name in self.cfg.static_vars}
            static_emb, static_var_weight = self.static_variable_selection(static_emb_input)
        else:
            static_emb = torch.zeros(
                (observation.cont_var.size(0), self.cfg.hidden_size),
                dtype=self.dtype,
                device=self.device,
            )
            static_var_weight = torch.zeros(
                (observation.cont_var.size(0), 0), dtype=self.dtype, device=self.device
            )

        # Copy context values for each time step. TODO Could we have context calculated for each step?
        static_context = self.static_context_variable_selection(static_emb)

        static_context = static_context[:, None].expand(-1, seq_len, -1)

        # Variable selection for Autoencoder
        time_varying_encoder_input = {
            name: hidden_states[name][:, : observation.encoder_len]
            for name in self.cfg.time_varying_encoder_vars
        }
        time_varying_decoder_input = {
            name: hidden_states[name][:, observation.encoder_len :]
            for name in self.cfg.time_varying_decoder_vars
        }

        (
            emb_varying_encoder,
            encoder_var_selection_weight,
        ) = self.encoder_variable_selection(
            time_varying_encoder_input, static_context[:, : observation.encoder_len]
        )

        (
            emb_varying_decoder,
            decoder_var_selection_weight,
        ) = self.decoder_variable_selection(
            time_varying_decoder_input, static_context[:, observation.encoder_len :]
        )

        # Autoencoder network
        init_lstm_hidden = self.static_context_initial_hidden_lstm(static_emb).expand(
            self.cfg.num_lstm_layers, -1, -1
        )
        init_lstm_cell = self.static_context_initial_cell_lstm(static_emb).expand(
            self.cfg.num_lstm_layers, -1, -1
        )

        encoder_output, (encoder_hidden, encoder_cell) = self.lstm_encoder(
            emb_varying_encoder,
            (init_lstm_hidden, init_lstm_cell),
            enforce_sorted=False,
        )

        decoder_output, _ = self.lstm_decoder(
            emb_varying_decoder, (encoder_hidden, encoder_cell), enforce_sorted=False
        )

        encoder_output = self.post_gate_encoder(encoder_output)
        encoder_output = self.post_add_norm_encoder(encoder_output, emb_varying_encoder)

        decoder_output = self.post_gate_decoder(decoder_output)
        decoder_output = self.post_add_norm_decoder(decoder_output, emb_varying_decoder)

        ae_output = torch.cat([encoder_output, decoder_output], dim=1)

        # Static enrichment
        static_context_enrichment_output = self.static_context_enrichment(static_emb)
        static_context_enrichment_output = static_context_enrichment_output[:, None].expand(
            -1, seq_len, -1
        )
        attn_input = self.static_enrichment(ae_output, static_context_enrichment_output)

        # Attention
        mask = self.get_attention_mask(
            decoder_len=observation.decoder_len, encoder_len=observation.encoder_len
        )
        attn_output, attn_weights = self.multihead_attn(
            x_query=attn_input[:, observation.encoder_len :],
            x_key=attn_input,
            x_value=attn_input,
            mask=mask,
        )
        attn_output = self.post_attn_gate_norm(
            attn_output, attn_input[:, observation.encoder_len :]
        )
        output = self.pos_wise_ff(attn_output)

        # Prediction
        output = self.pre_output_gate_norm(output, ae_output[:, observation.encoder_len :])
        if self.cfg.num_targets > 1:  # Multi-target architecture
            output = [output_layer(output) for output_layer in self.output_layer]
        else:
            output = self.output_layer(output)
        return TFTOutput(
            prediction=output,
            encoder_attn_weight=attn_weights[..., : observation.encoder_len],
            decoder_attn_weight=attn_weights[..., observation.encoder_len :],
            static_var_weight=static_var_weight,
            encoder_var_selection_weight=encoder_var_selection_weight,
            decoder_var_selection_weight=decoder_var_selection_weight,
        )

    def get_attention_mask(self, encoder_len: int, decoder_len: int) -> torch.Tensor:
        """Get masked matrix for attention layer we ensure the temporal dependency are respected"""
        decoder_masked = torch.triu(torch.ones(1, decoder_len, decoder_len, device=self.device))
        encoder_masked = torch.zeros(1, 1, encoder_len, device=self.device)
        mask = torch.cat([encoder_masked.expand(-1, decoder_len, -1), decoder_masked], dim=2)
        return mask
