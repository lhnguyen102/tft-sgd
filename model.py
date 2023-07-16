from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TimeDistributedInterpolation(nn.Module):
    """Reample data using interpolation"""

    def __init__(
        self, output_size: int, batch_first: bool = False, trainable: bool = False
    ) -> None:
        super().__init__()

        self.output_size = output_size
        self.batch_first = batch_first
        self.trainable = trainable

        if self.trainable:
            self.mask = nn.Parameter(torch.zeros(self.output_size, dtype=torch.float32))
            self.gate = nn.Sigmoid()

    def interpolate(self, observation: torch.Tensor):
        upsampled = F.interpolate(
            observation.unsqueeze(1), self.output_size, mode="linear", align_corners=True
        ).squeeze(1)
        if self.trainable:
            upsampled = upsampled * self.gate(self.mask.unsqueeze(0)) * 2.0
        return upsampled

    def forward(self, observation: torch.Tensor):
        if len(observation.size()) <= 2:
            return self.interpolate(observation)

        # Squash samples and timesteps into a single axis
        x_reshape = observation.contiguous().view(
            -1, observation.size(-1)
        )  # (samples * timesteps, input_size)

        y_interp = self.interpolate(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y_interp = y_interp.contiguous().view(
                observation.size(0), -1, y_interp.size(-1)
            )  # (samples, timesteps, output_size)
        else:
            y_interp = y_interp.view(
                -1, observation.size(1), y_interp.size(-1)
            )  # (timesteps, samples, output_size)

        return y_interp


class ResampleNorm(nn.Module):
    def __init__(self, input_size: int, output_size: int = None, trainable_add: bool = True):
        super().__init__()

        self.input_size = input_size
        self.trainable_add = trainable_add
        self.output_size = output_size or input_size

        if self.input_size != self.output_size:
            self.resample = TimeDistributedInterpolation(
                self.output_size, batch_first=True, trainable=False
            )

        if self.trainable_add:
            self.mask = nn.Parameter(torch.zeros(self.output_size, dtype=torch.float))
            self.gate = nn.Sigmoid()
        self.norm = nn.LayerNorm(self.output_size)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if self.input_size != self.output_size:
            observation = self.resample(observation)

        if self.trainable_add:
            observation = observation * self.gate(self.mask) * 2.0

        output = self.norm(observation)
        return output


class GatedLinearUnit(nn.Module):
    """Decide how much info should flow through the network"""

    def __init__(self, input_size: int, hidden_size: int = None, dropout: float = None) -> None:
        super().__init__()

        if dropout is not None:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = dropout
        self.hidden_size = hidden_size or input_size
        self.linear = nn.Linear(input_size, self.hidden_size * 2)

        self.init_weights()

    def init_weights(self) -> None:
        """Initalize the network parameters"""
        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.zeros_(param)
            elif "linear" in name:
                nn.init.xavier_uniform_(param)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        if self.dropout_layer is not None:
            observation = self.dropout_layer(observation)

        observation = self.linear(observation)
        observation = F.glu(observation, dim=-1)
        return observation


class AddNorm(nn.Module):
    """Add residual and normalize all values"""

    def __init__(self, input_size: int, skipe_size: int = None, trainable_add: bool = True) -> None:
        super().__init__()

        self.input_size = input_size
        self.trainable_add = trainable_add
        self.skip_size = skipe_size or input_size

        if self.input_size != self.skip_size:
            self.resample = TimeDistributedInterpolation(
                self.input_size, batch_first=True, trainable=False
            )

        if self.trainable_add:
            self.mask = nn.Parameter(torch.zeros(self.input_size, dtype=torch.float))
            self.gate = nn.Sigmoid()

        self.norm = nn.LayerNorm(self.input_size)

    def forward(self, observation: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if self.input_size != self.skip_size:
            skip = self.resample(skip)

        if self.trainable_add:
            skip = skip * self.gate(self.mask) * 2.0

        output = self.norm(observation + skip)

        return output


class GateAddNorm(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        skip_size: int,
        trainable_add: bool = False,
        dropout: float = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.skip_size = skip_size
        self.trainable_add = trainable_add
        self.dropout = dropout

        self.glu = GatedLinearUnit(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
        )
        self.add_norm = AddNorm(
            input_size=self.hidden_size,
            skipe_size=skip_size,
            trainable_add=trainable_add,
        )

    def forward(self, observation: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        output = self.glu(observation)
        output = self.add_norm(output, skip)

        return output


class GatedResidualNetwork(nn.Module):
    """It allows the model the flexibility to apply non-linear processing"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.1,
        context_size: int = None,
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout
        self.context_size = context_size
        self.residual = residual

        if self.input_size != self.output_size and not self.residual:
            residual_size = self.input_size
        else:
            residual_size = self.output_size

        if self.output_size != residual_size:
            self.resample_norm = ResampleNorm(residual_size, self.output_size)

        self.linear_1 = nn.Linear(self.input_size, self.hidden_size)
        self.elu = nn.ELU()

        if self.context_size is not None:
            self.context = nn.Linear(self.context_size, self.hidden_size, bias=False)

        self.linear_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.init_weights()

        self.gate_norm = GateAddNorm(
            input_size=self.hidden_size,
            skip_size=self.output_size,
            hidden_size=self.output_size,
            dropout=self.dropout,
            trainable_add=False,
        )

    def init_weights(self) -> None:
        """Initalize the network parameters"""
        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.zeros_(param)
            elif "linear_1" in name or "linear_2" in name:
                nn.init.kaiming_normal_(param, a=0, mode="fan_in", nonlinearity="leaky_relu")
            elif "context" in name:
                nn.init.xavier_uniform_(param)

    def forward(self, observation: torch.Tensor, context=None, residual=None) -> torch.Tensor:
        if residual is None:
            residual = observation

        if self.input_size != self.output_size and not self.residual:
            residual = self.resample_norm(residual)

        observation = self.linear_1(observation)
        if context is not None:
            context = self.context(context)
            observation = observation + context

        observation = self.elu(observation)
        observation = self.linear_2(observation)
        observation = self.gate_norm(observation, residual)

        return observation


class MultiHeadAttention(nn.Module):
    """Multi-head self attention"""

    def __init__(self, n_head: int, d_model: int, dropout: float = 0.0) -> None:
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = self.d_q = self.d_v = self.d_model // n_head
        self.dropout = nn.Dropout(p=dropout)

        self.v_layer = nn.Linear(self.d_model, self.d_v)
        self.q_layers = nn.ModuleList(
            [nn.Linear(self.d_model, self.d_q) for _ in range(self.n_head)]
        )
        self.k_layers = nn.ModuleList(
            [nn.Linear(self.d_model, self.d_k) for _ in range(self.n_head)]
        )
        self.softmax = torch.nn.Softmax(dim=2)
        self.output_proj = nn.Linear(self.d_v, self.d_model, bias=False)

    def init_weights(self):
        """Initialize weights and biases"""
        for name, param in self.named_parameters():
            if "bias" not in name:
                torch.nn.init.xavier_uniform_(param)
            else:
                torch.nn.init.zeros_(param)

    def forward(
        self, x_query: torch.Tensor, x_key: torch.Tensor, x_value: torch.Tensor, mask=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        heads = []
        attns = []
        value = self.v_layer(x_value)

        for i in range(self.n_head):
            query = self.q_layers[i](x_query)
            key = self.k_layers[i](x_key)
            attn = (query @ key.permute(0, 2, 1)) * (1.0 / math.sqrt(key.size(-1)))

            if mask is not None:
                attn = attn.masked_fill(mask, float("-inf"))
            attn = self.softmax(attn)
            head = attn & value

            attns.append(attn)
            heads.append(head)

        heads_tensor = torch.stack(heads, dim=2) if self.n_head > 1 else heads[0]
        attns_tensor = torch.tensor(attns, dim=2)

        outputs = torch.mean(heads_tensor, dim=2) if self.n_head > 1 else heads_tensor
        outputs = self.output_proj(outputs)
        outputs = self.dropout(outputs)

        return outputs, attns_tensor


class VariableSelectionNetwork(nn.Module):
    """Select relevant input each time step."""

    def __init__(
        self,
        input_sizes: Dict[str, int],
        hidden_size: int,
        input_embedding_flags: Dict[str, bool] = None,
        dropout: float = 0.1,
        context_size: int = None,
        single_variable_grns: Dict[str, GatedResidualNetwork] = None,
        prescalers: Dict[str, nn.Linear] = None,
    ) -> None:
        super().__init__()

        if input_embedding_flags is None:
            input_embedding_flags = {}

        if single_variable_grns is None:
            single_variable_grns = {}

        if prescalers is None:
            prescalers = {}

        self.hidden_size = hidden_size
        self.input_sizes = input_sizes
        self.input_embedding_flags = input_embedding_flags
        self.dropout = dropout
        self.context_size = context_size

        if self.num_inputs > 1:
            if self.context_size is not None:
                self.flattened_grn = GatedResidualNetwork(
                    input_size=self.input_size_total,
                    hidden_size=min(self.hidden_size, self.num_inputs),
                    output_size=self.num_inputs,
                    dropout=self.dropout,
                    context_size=self.context_size,
                    residual=False,
                )
            else:
                self.flattened_grn = GatedResidualNetwork(
                    input_size=self.input_size_total,
                    hidden_size=min(self.hidden_size, self.num_inputs),
                    output_size=self.num_inputs,
                    dropout=self.dropout,
                    residual=False,
                )

        self.single_variable_grns = nn.ModuleDict()
        self.prescalers = nn.ModuleDict()
        for name, input_size in self.input_sizes.items():
            if name in single_variable_grns:
                self.single_variable_grns[name] = single_variable_grns[name]
            elif self.input_embedding_flags.get(name, False):
                self.single_variable_grns[name] = ResampleNorm(
                    input_size=input_size, output_size=self.hidden_size
                )
            else:
                self.single_variable_grns[name] = GatedResidualNetwork(
                    input_size=input_size,
                    hidden_size=min(input_size, self.hidden_size),
                    output_size=self.hidden_size,
                    dropout=self.dropout,
                )

            if name in prescalers:
                self.prescalers[name] = prescalers[name]
            elif not self.input_embedding_flags.get(name, False):
                self.prescalers[name] = nn.Linear(1, input_size)

        self.softmax = nn.Softmax(dim=-1)

    @property
    def input_size_total(self):
        return sum(
            size if name in self.input_embedding_flags else size
            for name, size in self.input_sizes.items()
        )

    @property
    def num_inputs(self):
        return len(self.input_sizes)

    def forward(
        self, observation: Dict[str, torch.Tensor], context: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.num_inputs > 1:
            var_outputs = []
            weight_inputs = []

            for name in self.input_sizes.keys():
                var_embedding = observation[name]
                if name in self.prescalers:
                    var_embedding = self.prescalers[name](var_embedding)

                weight_inputs.append(var_embedding)
                var_outputs.append(self.single_variable_grns[name](var_embedding))
            var_outputs = torch.stack(var_outputs, dim=-1)

            flat_embedding = torch.cat(weight_inputs, dim=-1)
            sparse_weights = self.flattened_grn(flat_embedding, context)
            sparse_weights = self.softmax(sparse_weights).unsqueeze(-2)

            outputs = var_outputs * sparse_weights
            outputs = outputs.sum(dim=-1)

        else:  # one input
            name = next(iter(self.single_variable_grns.keys()))
            var_embedding = observation[name]
            if name in self.prescalers:
                var_embedding = self.prescalers[name](var_embedding)
            outputs = self.single_variable_grns[name](var_embedding)

            if outputs.ndim == 3:  # -> batch size, time, hidden size, n_vars
                sparse_weights = torch.ones(
                    outputs.size(0), outputs.size(1), 1, 1, device=outputs.device
                )
            else:  # ndim == 2 -> batch size, hidden size, n_vars
                sparse_weights = torch.ones(outputs.size(0), 1, 1, device=outputs.device)

        return outputs, sparse_weights
