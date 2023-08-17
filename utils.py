from typing import List

import numpy as np
import pandas as pd
import torch

from data_preprocessor import TimeseriesEncoderNormalizer


def denormalize_target(
    target: torch.Tensor, target_var: List[str], normalizer: TimeseriesEncoderNormalizer
) -> torch.Tensor:
    """Denormalize the target data"""
    target_numpy = target.detach().cpu().numpy()
    data_frame = pd.DataFrame(
        {name: target_numpy[..., i].flatten() for i, name in enumerate(target_var)}
    )
    batch_size, seq_len, _ = target.shape

    denorm_df = normalizer.denormalize(data_frame)
    denorm_targets = [
        torch.tensor(denorm_df[name].to_numpy(np.float32), dtype=torch.float32).reshape(
            batch_size, seq_len, 1
        )
        for name in target_var
    ]

    denorm_targets = []
    for target_name in target_var:
        tmp = torch.tensor(
            denorm_df[target_name].to_numpy(np.float32), dtype=torch.float32
        ).reshape(batch_size, seq_len, 1)

        denorm_targets.append(tmp)

    return torch.cat(denorm_targets, dim=2)
