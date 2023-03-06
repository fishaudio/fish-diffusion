from pathlib import Path

import click
import onnxruntime as ort
import torch
from export import export_feature_extractor
from loguru import logger
from mmengine import Config
from torch.nn import functional as F

from fish_diffusion.archs.diffsinger import DiffSinger, DiffSingerLightning
from fish_diffusion.utils.inference import load_checkpoint


def denorm_f0(f0, pitch_padding=None):
    rf0 = 2**f0
    rf0[pitch_padding] = 0
    return rf0


def add_pitch(f0, mel2ph):
    pitch_padding = mel2ph == 0
    f0_denorm = denorm_f0(f0, pitch_padding=pitch_padding)
    return f0_denorm


class FeatureEmbeddingWrapper(torch.nn.Module):
    def __init__(self, model: DiffSinger):
        super().__init__()

        self.model = model

    def forward(self, hubert, mel2ph, spk_embed, f0):
        decoder_inp = F.pad(hubert, [0, 0, 1, 0])
        mel2ph_ = mel2ph.unsqueeze(2).repeat([1, 1, hubert.shape[-1]])
        decoder_inp = torch.gather(decoder_inp, 1, mel2ph_)  # [B, T, H]

        f0_denorm = add_pitch(f0, mel2ph)

        max_src_len = decoder_inp.shape[1]
        features = self.model.text_encoder(decoder_inp, None)
        speaker_embed = (
            self.model.speaker_encoder(spk_embed)
            .unsqueeze(1)
            .expand(-1, max_src_len, -1)
        )
        features += speaker_embed
        features += self.model.pitch_encoder(f0_denorm)

        return features.transpose(1, 2), f0_denorm


def export_feature_embedding(model, device):
    model = FeatureEmbeddingWrapper(model).to(device)

    n_frames = 100
    speakers = torch.tensor([0], dtype=torch.long, device=device)
    text_features = torch.randn((1, n_frames, 256), device=device)
    mel2ph = torch.arange(0, n_frames, dtype=torch.int64)[None]
    pitches = torch.rand((1, n_frames), device=device)

    torch.onnx.export(
        model,
        (text_features, mel2ph, speakers, pitches),
        "./exported/moess_encoder.onnx",
        input_names=["hubert", "mel2ph", "spk_embed", "f0"],
        output_names=["mel_pred", "f0_pred"],
        dynamic_axes={
            "hubert": {1: "n_frames"},
            "mel2ph": {1: "n_frames"},
            "f0": {1: "n_frames"},
            "mel_pred": {1: "n_frames"},
            "f0_pred": {1: "n_frames"},
        },
        verbose=False,
        opset_version=16,
    )

    logger.info("ONNX Feature Encoder exported.")

    n_frames = 110
    speakers = torch.tensor([0], dtype=torch.long, device=device)
    text_features = torch.randn((1, n_frames, 256), device=device)
    mel2ph = torch.arange(0, n_frames, dtype=torch.int64)[None]
    pitches = torch.rand((1, n_frames), device=device)

    # Verify the exported model
    ort_session = ort.InferenceSession("./exported/moess_encoder.onnx")
    ort_inputs = {
        "hubert": text_features.cpu().numpy(),
        "mel2ph": mel2ph.cpu().numpy(),
        "spk_embed": speakers.cpu().numpy(),
        "f0": pitches.cpu().numpy(),
    }

    ort_outs = ort_session.run(None, ort_inputs)
    torch_outs = model(text_features, mel2ph, speakers, pitches)

    assert torch.allclose(
        torch_outs[0], torch.from_numpy(ort_outs[0][0]), atol=1e-4
    ), "ONNX model output does not match PyTorch model output."

    assert torch.allclose(
        torch_outs[1], torch.from_numpy(ort_outs[1][0]), atol=1e-4
    ), "ONNX model output does not match PyTorch model output."

    logger.info("ONNX Feature Encoder verified.")


class AfterDiffusion(torch.nn.Module):
    # A special module for MoeSS

    def __init__(self, spec_min, spec_max):
        super().__init__()

        self.spec_min = spec_min
        self.spec_max = spec_max

    def forward(self, x):
        x = x.squeeze(1).permute(0, 2, 1)
        d = (self.spec_max - self.spec_min) / 2
        m = (self.spec_max + self.spec_min) / 2
        mel_out = x * d + m
        mel_out = mel_out * 2.30259

        return mel_out.transpose(2, 1)


def export_moess_diffusion(config, model, device):
    # Trace the denoiser
    n_frames = 10
    x = torch.randn((1, 1, config.mel_channels, n_frames), device=device)
    step = torch.randint(
        0, config.model.diffusion.timesteps, (1,), device=device, dtype=torch.long
    )
    cond = torch.randn((1, config.hidden_size, n_frames), device=device)

    model.diffusion.denoise_fn = torch.jit.trace(
        model.diffusion.denoise_fn, (x, step, cond), check_trace=True
    )

    logger.info("denoiser traced.")

    torch.onnx.export(
        model.diffusion.denoise_fn,
        (x, step, cond),
        "exported/moess_denoiser.onnx",
        input_names=["noise", "step", "condition"],
        output_names=["noise_pred"],
        dynamic_axes={
            "noise": {
                3: "n_frames",
            },
            "condition": {
                2: "n_frames",
            },
        },
        opset_version=16,
    )

    logger.info("denoiser exported.")

    # Verify the exported model
    n_frames = 100
    x = torch.randn((1, 1, config.mel_channels, n_frames), device=device)
    step = torch.randint(
        0, config.model.diffusion.timesteps, (1,), device=device, dtype=torch.long
    )
    cond = torch.randn((1, config.hidden_size, n_frames), device=device)

    ort_session = ort.InferenceSession("exported/moess_denoiser.onnx")
    ort_inputs = {
        "noise": x.cpu().numpy(),
        "step": step.cpu().numpy(),
        "condition": cond.cpu().numpy(),
    }

    ort_outs = ort_session.run(None, ort_inputs)[0]

    assert torch.allclose(
        model.diffusion.denoise_fn(x, step, cond),
        torch.from_numpy(ort_outs),
        atol=1e-4,
    ), "ONNX model output does not match PyTorch model output."

    logger.info("ONNX denoiser verified.")

    # Export plms noise predictor
    t = torch.randint(10, 20, (1,), device=device, dtype=torch.long)
    t_prev = torch.randint(0, 10, (1,), device=device, dtype=torch.long)

    torch.onnx.export(
        model.diffusion.plms_noise_predictor,
        (x, x, t, t_prev),
        "exported/moess_plms_noise_predictor.onnx",
        input_names=["noise", "noise_pred", "time", "time_prev"],
        output_names=["noise_pred_o"],
        dynamic_axes={
            "noise": {
                3: "n_frames",
            },
            "noise_pred": {
                3: "n_frames",
            },
        },
        opset_version=16,
    )

    logger.info("PLMS noise predictor exported.")

    # Verify the exported model
    ort_session = ort.InferenceSession("exported/moess_plms_noise_predictor.onnx")
    ort_inputs = {
        "noise": x.cpu().numpy(),
        "noise_pred": x.cpu().numpy(),
        "time": t.cpu().numpy(),
        "time_prev": t_prev.cpu().numpy(),
    }

    ort_outs = ort_session.run(None, ort_inputs)[0]

    assert torch.allclose(
        model.diffusion.plms_noise_predictor(x, x, t, t_prev),
        torch.from_numpy(ort_outs),
        atol=1e-4,
    ), "ONNX model output does not match PyTorch model output."

    logger.info("ONNX PLMS noise predictor verified.")

    # Export after diffusion
    after_diffusion = AfterDiffusion(model.diffusion.spec_max, model.diffusion.spec_min)

    torch.onnx.export(
        after_diffusion,
        x,
        "exported/moess_after_diffusion.onnx",
        input_names=["x"],
        output_names=["mel_out"],
        dynamic_axes={
            "x": {
                3: "n_frames",
            },
            "mel_out": {
                2: "n_frames",
            },
        },
        opset_version=16,
    )

    logger.info("After diffusion exported.")

    # Verify the exported model
    ort_session = ort.InferenceSession("exported/moess_after_diffusion.onnx")
    ort_inputs = {
        "x": x.cpu().numpy(),
    }

    ort_outs = ort_session.run(None, ort_inputs)[0]

    assert torch.allclose(
        after_diffusion(x),
        torch.from_numpy(ort_outs),
        atol=1e-4,
    ), "ONNX model output does not match PyTorch model output."

    logger.info("ONNX After diffusion verified.")


@click.command()
@click.option("--config", default="configs/exp_cn_hubert_soft_finetune.py")
@click.option(
    "--checkpoint",
    default="logs/DiffSVC/oonzyobz/checkpoints/epoch=1249-step=5000-valid_loss=0.31.ckpt",
)
def main(config: str, checkpoint: str):
    Path("exported").mkdir(exist_ok=True)

    device = "cpu"
    config = Config.fromfile(config)
    model = load_checkpoint(config, checkpoint, device, model_cls=DiffSingerLightning)

    # Ignore vocoder
    model = model.model

    logger.info("Model loaded.")

    # Export feature extractor
    export_feature_extractor(config, device)

    # Export feature embedding
    export_feature_embedding(model, device)

    # Export diffusion
    export_moess_diffusion(config, model, device)


if __name__ == "__main__":
    main()
