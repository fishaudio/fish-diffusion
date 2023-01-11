import torch

from .base import BaseFeatureExtractor
from fairseq import checkpoint_utils


class ChineseHubertFairSeq(BaseFeatureExtractor):
    def __init__(
        self, path="checkpoints/hubert_chinese/chinese-hubert-base-fairseq-ckpt.pt"
    ):
        super().__init__()

        models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
            [path],
            suffix="",
        )

        self.model = models[0]
        self.model.eval()

        checkpoint = torch.hub.load_state_dict_from_url(
            "https://github.com/bshall/hubert/releases/download/v0.1/kmeans100-50f36a95.pt",
            progress=True,
        )
        self.cluster_centers = checkpoint["cluster_centers_"]

    def forward(self, path_or_audio, sampling_rate=None):
        audio = self.preprocess(path_or_audio, sampling_rate)

        feats = audio.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False)

        inputs = {
            "source": feats.to(self.device),
            "padding_mask": padding_mask.to(self.device),
            "output_layer": 9,  # layer 9
        }
        with torch.no_grad():
            logits = self.model.extract_features(**inputs)
            feats = self.model.final_proj(logits[0])

        # (B, D, T) -> (B, T, D)
        feats = feats.transpose(1, 2)

        # Get distance to all centers for
        # each feature vector

        print(feats.shape, self.cluster_centers.shape)
        dists = torch.cdist(feats, self.cluster_centers)

        print(dists.shape)

        exit()
