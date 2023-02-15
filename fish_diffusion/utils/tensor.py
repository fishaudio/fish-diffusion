from typing import Union

import numpy as np
import torch


def repeat_expand(
    content: Union[torch.Tensor, np.ndarray], target_len: int, mode: str = "nearest"
):
    """Repeat content to target length.
    This is a wrapper of torch.nn.functional.interpolate.

    Args:
        content (torch.Tensor): tensor
        target_len (int): target length
        mode (str, optional): interpolation mode. Defaults to "nearest".

    Returns:
        torch.Tensor: tensor
    """

    ndim = content.ndim

    if content.ndim == 1:
        content = content[None, None]
    elif content.ndim == 2:
        content = content[None]

    assert content.ndim == 3

    is_np = isinstance(content, np.ndarray)
    if is_np:
        content = torch.from_numpy(content)

    results = torch.nn.functional.interpolate(content, size=target_len, mode=mode)

    if is_np:
        results = results.numpy()

    if ndim == 1:
        return results[0, 0]
    elif ndim == 2:
        return results[0]
