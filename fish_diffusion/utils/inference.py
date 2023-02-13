import torch

from train import FishDiffusion


def load_checkpoint(config, checkpoint, device="cuda") -> FishDiffusion:
    """Load checkpoint from path

    Args:
        config: config
        checkpoint: checkpoint path
        device: device

    Returns:
        FishDiffusion: model
    """

    model = FishDiffusion(config)
    state_dict = torch.load(checkpoint, map_location="cpu")

    if "state_dict" in state_dict:  # Checkpoint is saved by pl
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model
