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
    """Convert the categorical variables to continuous domain"""

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
    """Embedding layer for categorical variables including groups of categorical variables.

    Enabled for static and dynamic categories (i.e. 3 dimensions for batch x time x categories).

    Args:
        embedding_sizes (Union[Dict[str, Tuple[int, int]], Dict[str, int], List[int],
            List[Tuple[int, int]]]):

            * dictionary of embedding sizes, e.g. ``{'cat1': (10, 3)}``
              indicates that the first categorical variable has 10 unique values which are mapped to 3 embedding
              dimensions. Use :py:func:`~pytorch_forecasting.utils.get_embedding_size` to automatically obtain
              reasonable embedding sizes depending on the number of categories.
            * dictionary of categorical sizes, e.g. ``{'cat1': 10}`` where embedding sizes are inferred by
              :py:func:`~pytorch_forecasting.utils.get_embedding_size`.
            * list of embedding and categorical sizes, e.g. ``[(10, 3), (20, 2)]`` (requires ``x_categoricals`` to
              be empty)
            * list of categorical sizes where embedding sizes are inferred by
              :py:func:`~pytorch_forecasting.utils.get_embedding_size` (requires ``x_categoricals`` to be empty).

            If input is provided as list, output will be a single tensor of shape batch x (optional) time x
            sum(embedding_sizes). Otherwise, output is a dictionary of embedding tensors.
        x_categoricals (List[str]): list of categorical variables that are used as input.
        categorical_groups (Dict[str, List[str]]): dictionary of categories that should be summed up in an
            embedding bag, e.g. ``{'cat1': ['cat2', 'cat3']}`` indicates that a new categorical variable ``'cat1'``
            is mapped to an embedding bag containing the second and third categorical variables.
            Defaults to empty dictionary.
        embedding_paddings (List[str]): list of categorical variables for which the value 0 is mapped to a zero
            embedding vector. Defaults to empty list.
        max_embedding_size (int, optional): if embedding size defined by ``embedding_sizes`` is larger than
            ``max_embedding_size``, it will be constrained. Defaults to None.
    """

    def __init__(
        self,
        embedding_sizes: Union[
            Dict[str, Tuple[int, int]], Dict[str, int], List[int], List[Tuple[int, int]]
        ],
        x_categoricals: List[str] = None,
        categorical_groups: Dict[str, List[str]] = None,
        embedding_paddings: List[str] = None,
        max_embedding_size: int = None,
    ) -> None:
        super().__init__()

        if isinstance(embedding_sizes, dict):
            self.concat_input = False
            assert x_categoricals is not None, "x_categoricals must be provided"
            categorical_group_var = [
                name for names in categorical_groups.values for name in names
            ]
            if len(categorical_groups) > 0:
                assert all(
                    name in embedding_sizes for name in categorical_groups
                ), "categorical_groups must be in embedding sizes"
                assert all(
                    name in x_categoricals for name in categorical_group_var
                ), "group variables in categorical_groups must be in x_categoricals."
            assert all(
                name in embedding_sizes
                for name in embedding_sizes
                if name not in categorical_group_var
            ), (
                "all variables in embedding_sizes must be in x_categoricals - but only if"
                "not already in categorical_groups."
            )

        else:
            assert (
                x_categoricals is None and len(categorical_groups) == 0
            ), "If embedding_sizes is not a dictionary, categorical_groups and x_categoricals must be empty."
            # number embeddings based on order
            embedding_sizes = {
                str(name): size for name, size in enumerate(embedding_sizes)
            }
            x_categoricals = list(embedding_sizes.keys())
            self.concat_output = True

        self.embedding_sizes = {
            name: (size, get_embedding_size(size)) if isinstance(size, int) else size
            for name, size in embedding_sizes.items()
        }
        self.categorical_groups = categorical_groups
        self.embedding_paddings = embedding_paddings
        self.max_embedding_size = max_embedding_size
        self.x_categoricals = x_categoricals

        self.init_embeddings()

    def init_embeddings(self) -> None:
        self.embeddings = nn.ModuleDict()
        for name in self.embedding_sizes.keys():
            embedding_size = self.embedding_sizes[name][1]
            if self.max_embedding_size is not None:
                embedding_size = min(embedding_size, self.max_embedding_size)
            self.embedding_sizes[name] = list(self.embedding_sizes[name])
            self.embedding_sizes[name][1] = embedding_size

            if name in self.categorical_groups:
                self.embeddings[name] = TemporalEmbeddingBag(
                    self.embedding_sizes[name][0],
                    embedding_size,
                    mode="sum",
                    batch_first=True,
                )
            else:
                if name in self.embedding_paddings:
                    padding_idx = 0
                else:
                    padding_idx = None
                self.embeddings[name] = nn.Embedding(
                    self.embedding_sizes[name][0], embedding_size, padding_idx
                )

    def names(self):
        return list(self.keys())

    def items(self):
        return self.embeddings.items()

    def keys(self):
        return self.embeddings.keys()

    def values(self):
        return self.embeddings.values()

    def __getitem__(self, name: str):
        return self.embeddings[name]

    @property
    def input_size(self) -> int:
        return len(self.x_categoricals)

    @property
    def output_size(self) -> Union[Dict[str, int], int]:
        if self.concat_output:
            return sum([s[1] for s in self.embedding_sizes.values()])
        else:
            return {name: s[1] for name, s in self.embedding_sizes.items()}

    def forward(
        self, observation: torch.Tensor
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Args:
            x (torch.Tensor): input tensor of shape batch x (optional) time x
                categoricals in the order of ``x_categoricals``.

        Returns:
            Union[Dict[str, torch.Tensor], torch.Tensor]: dictionary of category
                names to embeddings of shape batch x (optional) time x embedding_size if
                ``embedding_size`` is given as dictionary. Otherwise, returns the embedding
                of shape batch x (optional) time x sum(embedding_sizes). Query attribute
                ``output_size`` to get the size of the output(s).
        """

        input_vectors = {}
        for name, emb in self.embeddings.items():
            if name in self.categorical_groups:
                input_vectors[name] = emb(
                    observation[
                        ...,
                        [
                            self.x_categoricals.index(cat_name)
                            for cat_name in self.categorical_groups[name]
                        ],
                    ]
                )
            else:
                input_vectors[name] = emb(
                    observation[..., self.x_categoricals.index(name)]
                )
        if self.concat_output:  # concatenate output
            return torch.cat(list(input_vectors.values()), dim=-1)
        else:
            return input_vectors
