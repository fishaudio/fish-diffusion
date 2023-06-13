from typing import Optional, Union

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


def interpolate(
    x: torch.Tensor,
    xp: torch.Tensor,
    fp: torch.Tensor,
    left: Optional[torch.Tensor] = None,
    right: Optional[torch.Tensor] = None,
):
    """Interpolate a 1-D function.

    Args:
        x (torch.Tensor): The x-coordinates at which to evaluate the interpolated values.
        xp (torch.Tensor): A 1-D array of monotonically increasing real values.
        fp (torch.Tensor): A 1-D array of real values, same length as xp.
        left (torch.Tensor, optional): Value to return for x < xp[0], default is fp[0].
        right (torch.Tensor, optional): Value to return for x > xp[-1], default is fp[-1].

    Returns:
        torch.Tensor: The interpolated values, same shape as x.
    """

    # Ref: https://github.com/pytorch/pytorch/issues/1552#issuecomment-979998307
    i = torch.clip(torch.searchsorted(xp, x, right=True), 1, len(xp) - 1)
    interped = (fp[i - 1] * (xp[i] - x) + fp[i] * (x - xp[i - 1])) / (xp[i] - xp[i - 1])

    if left is None:
        left = fp[0]

    interped = torch.where(x < xp[0], left, interped)

    if right is None:
        right = fp[-1]

    interped = torch.where(x > xp[-1], right, interped)

    return interped
