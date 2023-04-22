import torch
from diffusers import UNet2DConditionModel

from .builder import DENOISERS


@DENOISERS.register_module()
class UNetDenoiserV2(UNet2DConditionModel):
    def forward(
        self, x: torch.Tensor, diffusion_step: torch.Tensor, conditioner: torch.Tensor
    ):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `diffusion_step` has shape `[batch_size]`
        * `conditioner` has shape `[batch_size, in_channels, height, width]`
        """

        from_3d = False
        if x.ndim == 3:
            from_3d = True
            x = x.unsqueeze(1)

        if conditioner.ndim == 4:
            conditioner = conditioner.squeeze(1)

        conditioner = conditioner.transpose(1, 2)

        x = (
            super()
            .forward(
                sample=x,
                timestep=diffusion_step,
                encoder_hidden_states=conditioner,
            )
            .sample
        )

        if from_3d:
            x = x.squeeze(1)

        return x


if __name__ == "__main__":
    n = 5
    # Create a random batch of images
    x = torch.rand(n, 1, 128, int(1024 * 1.5)).cuda()
    # Create a random batch of time-steps
    t = torch.randint(0, 100, (n,)).cuda()
    # Create a random batch of conditioners
    c = torch.rand(n, 1, 256, int(1024 * 1.5)).cuda()

    # Create the model
    model = UNetDenoiserV2(
        in_channels=1,
        out_channels=1,
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        encoder_hid_dim=256,
        block_out_channels=(32, 64, 128, 512),
        attention_head_dim=4,
    ).cuda()
    # Forward pass
    y = model(x, t, c)
    # Print the shape of the output
    print(y.shape)

    # Count the number of parameters
    print(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, "M")
    # Get memory usage
    print(torch.cuda.memory_allocated() / 1e6, "M")
