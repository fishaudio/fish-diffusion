import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from fish_diffusion.encoders import ENCODERS
from mmengine import Config
from fish_diffusion.moessdiffusion import MOESSDIFFUSIONS


def denorm_f0(f0, pitch_padding=None):
    rf0 = 2 ** f0
    rf0[pitch_padding] = 0
    return rf0


def add_pitch(f0, mel2ph):
    pitch_padding = (mel2ph == 0)
    f0_denorm = denorm_f0(f0, pitch_padding=pitch_padding)
    return f0_denorm


class DiffSvc(nn.Module):
    def __init__(self, model_config):
        super(DiffSvc, self).__init__()
        self.text_encoder = ENCODERS.build(model_config.text_encoder)
        self.diffusion = MOESSDIFFUSIONS.build(model_config.diffusion)
        self.speaker_encoder = ENCODERS.build(model_config.speaker_encoder)
        self.pitch_encoder = ENCODERS.build(model_config.pitch_encoder)

    def forward(self, hubert, mel2ph, spk_embed, f0):
        decoder_inp = F.pad(hubert, [0, 0, 1, 0])
        mel2ph_ = mel2ph.unsqueeze(2).repeat([1, 1, hubert.shape[-1]])
        decoder_inp = torch.gather(decoder_inp, 1, mel2ph_)  # [B, T, H]

        f0_denorm = add_pitch(f0, mel2ph)

        max_src_len = decoder_inp.shape[1]
        features = self.text_encoder(decoder_inp, None)
        speaker_embed = (
            self.speaker_encoder(spk_embed).unsqueeze(1).expand(-1, max_src_len, -1)
        )
        features += speaker_embed
        features += self.pitch_encoder(f0_denorm)
        return features.transpose(1, 2), f0_denorm


class FishDiffusion(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.model = DiffSvc(config.model)
        self.config = config


def main(project_name):
    device = "cpu"
    config = Config.fromfile("configs/svc_hubert_soft_multi_speakers.py")
    model = FishDiffusion(config)
    state_dict = torch.load(
        "epoch=619-step=140000-valid_loss=0.22.ckpt",
        map_location=device,
    )["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    model = model.model

    hubert = torch.randn(1, 300, 256)
    mel2ph = torch.arange(0, 300, dtype=torch.int64).unsqueeze(0)
    f0 = torch.randn(1, 300)
    spk_embed = torch.LongTensor([0])
    print(hubert.shape, mel2ph.shape, spk_embed.shape, f0.shape)
    torch.onnx.export(
        model,
        (hubert, mel2ph, spk_embed, f0),
        f"{project_name}_encoder.onnx",
        input_names=["hubert", "mel2ph", "spk_embed", "f0"],
        output_names=["mel_pred", "f0_pred"],
        dynamic_axes={
            "hubert": [1],
            "f0": [1],
            "mel2ph": [1]
        },
        opset_version=16
    )

    print("exporting Diffusion")
    model.diffusion.MoeSSOnnxExport(project_name, device)
    print("Diffusion exported")


if __name__ == "__main__":
    main(project_name="MyModel")
