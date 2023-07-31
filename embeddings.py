from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn


def get_embedding_size(n: int, max_size: int = 100) -> int:
    """
    Determine empirically good embedding sizes (formula taken from fastai).

    Args:
        n (int): number of classes
        max_size (int, optional): maximum embedding size. Defaults to 100.

    Returns:
        int: embedding size
    """
    if n > 2:
        return min(round(1.6 * n**0.56), max_size)
    else:
        return 1


class TemporalEmbeddingBag(nn.EmbeddingBag):
    """
    A class that extends nn.EmbeddingBag to support an additional time dimension in the input data.
    It reshapes the input data, applies the nn.EmbeddingBag layer, and then reshapes the output.
    """

    def __init__(self, *args, batch_first: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.batch_first = batch_first

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if len(observation.size()) <= 2:
            return super().forward(observation)

        # Shape (sample * timesteps, input size)
        obs_reshape = observation.contiguous().view(-1, observation.size(-1))

        output = super().forward(obs_reshape)

        # Reshape output
        if self.batch_first:
            # (sample, timesteps, output_size)
            output = output.contiguous().view(observation.size(0), -1, output.size(-1))
        else:
            # (timestep, samples, output_size)
            output = output.contiguous().view(-1, observation.size(1), output.size(-1))

        return output


class MultiEmbedding(nn.Module):
    """Embedding layer for categorical variables including groups of categorical variables."""

    def __init__(
        self,
        embedding_sizes: Dict[str, Dict[str, int]],
        cat_var: List[str],
        cat_var_ordering: Dict[str, int],
        multi_cat_var: Union[Dict[str, List[str]], None] = None,
    ) -> None:
        super().__init__()
        self.embedding_sizes = embedding_sizes
        self.cat_var = cat_var
        self.cat_var_ordering = cat_var_ordering
        self.multi_cat_var = multi_cat_var

        self.init_embeddings()

    def init_embeddings(self) -> None:
        self.embeddings = nn.ModuleDict()
        for name in self.embedding_sizes.keys():
            num_classes = self.embedding_sizes[name]["num_classes"]
            embedding_size = self.embedding_sizes[name]["emb_size"]

            if name in self.multi_cat_var:
                self.embeddings[name] = TemporalEmbeddingBag(
                    num_classes, embedding_size, mode="sum", batch_first=True
                )
            else:
                self.embeddings[name] = nn.Embedding(num_classes, embedding_size)

    def forward(self, observation: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Precompute indices to avoid repetitive computation
        indices = {
            name: self.cat_var_ordering.index(name) for name in self.embeddings.keys()
        }

        outputs = {}
        for name, emb in self.embeddings.items():
            if name in self.multi_cat_var:
                multi_cat_indices = [
                    indices[cat_name] for cat_name in self.multi_cat_var[name]
                ]
                outputs[name] = emb(observation[..., multi_cat_indices])
            else:
                outputs[name] = emb(observation[..., indices[name]])

        return outputs
