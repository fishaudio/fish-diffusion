import numpy as np
import torch
from loguru import logger
from torch import nn

from .builder import ENCODERS


@ENCODERS.register_module()
class SimilarClusterEncoder(nn.Module):
    def __init__(
        self,
        n_clusters: int = 128,
        input_size: int = 256,
        output_size: int = 256,
        restore_path: str = None,
    ):
        """
        This encoder is a simple wrapper around a set of cluster centers.
        It finds the closest cluster center to each input vector and projects it to the output size.

        Args:
            n_clusters (int, optional): Number of clusters. Defaults to 128.
            input_size (int, optional): Input size. Defaults to 256.
            output_size (int, optional): Output size. Defaults to 256.
            restore_path (str, optional): Path to the cluster centers. Defaults to None.
        """
        super().__init__()

        self.cluster_centers = nn.Parameter(
            torch.rand(n_clusters, input_size), requires_grad=True
        )

        if restore_path is not None:
            self.cluster_centers.data = torch.from_numpy(np.load(restore_path))
            logger.info(f"Restored cluster centers from {restore_path}")

        self.proj = nn.Linear(input_size, output_size)

    def forward(self, x, src_masks=None):
        # Normalize the input vectors, so that the cluster centers are not
        # biased towards a particular direction

        distances = torch.cdist(x, self.cluster_centers)
        selected = torch.argmin(distances, dim=2)

        # We still have gradients flowing through the cluster centers
        x = self.cluster_centers[selected]
        x = self.proj(x)

        if src_masks is not None:
            x = x * (~src_masks).unsqueeze(-1)

        return x
