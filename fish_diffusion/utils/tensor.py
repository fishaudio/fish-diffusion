import torch


def repeat_expand_2d(content: torch.Tensor, target_len: int):
    """Repeat 2D content to target length.
    This is a wrapper of torch.nn.functional.interpolate.

    Args:
        content (torch.Tensor): 2D tensor (n_frames, n_features)
        target_len (int): target length

    Returns:
        torch.Tensor: 2D tensor (target_len, n_features)
    """

    return torch.nn.functional.interpolate(
        content[None], size=target_len, mode="nearest"
    )[0]
