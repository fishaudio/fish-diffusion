import torch

from fish_diffusion.archs.diffsinger.diffsinger import DiffSingerLightning


def load_checkpoint(config, checkpoint, device="cuda", model_cls=DiffSingerLightning):
    """Load checkpoint from path

    Args:
        config: config
        checkpoint: checkpoint path
        device: device

    Returns:
        model
    """

    model = model_cls(config)
    state_dict = torch.load(checkpoint, map_location="cpu")

    if "state_dict" in state_dict:  # Checkpoint is saved by pl
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    return model
