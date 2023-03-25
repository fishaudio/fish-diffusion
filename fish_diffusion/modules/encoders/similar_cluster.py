import torch
from torch import nn

from .builder import ENCODERS


@ENCODERS.register_module()
class SimilarClusterEncoder(nn.Module):
    def __init__(
        self, n_clusters: int = 100, input_size: int = 256, output_size: int = 256
    ):
        """
        This encoder is a simple wrapper around a set of cluster centers.
        It finds the closest cluster center to each input vector and projects it to the output size.

        Args:
            n_clusters (int, optional): Number of clusters. Defaults to 100.
            input_size (int, optional): Input size. Defaults to 256.
            output_size (int, optional): Output size. Defaults to 256.
        """
        super().__init__()

        self.cluster_centers = nn.Parameter(
            torch.randn(n_clusters, input_size), requires_grad=True
        )

        if input_size != output_size:
            self.proj = nn.Linear(input_size, output_size)
        else:
            # No need to project
            self.proj = nn.Identity()

    def forward(self, x, src_masks=None):
        distances = torch.cdist(x, self.cluster_centers)
        selected = torch.argmin(distances, dim=2)

        # We still have gradients flowing through the cluster centers
        x = self.cluster_centers[selected]
        x = self.proj(x)

        if src_masks is not None:
            x = x * src_masks.unsqueeze(-1)

        return x
