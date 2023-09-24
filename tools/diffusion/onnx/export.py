from pathlib import Path

import click
import numpy as np
import onnxruntime as ort
import torch
from loguru import logger
from mmengine import Config

from fish_diffusion.archs.diffsinger import DiffSinger, DiffSingerLightning
from fish_diffusion.modules.feature_extractors import FEATURE_EXTRACTORS
from fish_diffusion.utils.inference import load_checkpoint


class FeatureEmbeddingWrapper(torch.nn.Module):
    def __init__(self, model: DiffSinger):
        super().__init__()

        self.model = model

    def forward(self, speakers, text_features, pitches, pitch_shift=None):
        # Speakers: [n_frames, 1]/[1], text_features: [n_frames, 256], pitches: [n_frames], pitch_shift: [n_frames, 1]/[1]

        pitches = pitches.unsqueeze(0)
        text_features = text_features.unsqueeze(0)

        features = self.model.forward_features(
            speakers=speakers,
            contents=text_features,
            contents_lens=None,
            contents_max_len=text_features.shape[1],
            mel_lens=None,
            pitches=pitches,
            pitch_shift=pitch_shift,
        )

        return features["features"]


def export_feature_embedding(model, device):
    has_pitch_shift = hasattr(model, "pitch_shift_encoder")
    logger.info(f"The model has pitch shift encoder: {has_pitch_shift}")

    model = FeatureEmbeddingWrapper(model).to(device)

    n_frames = 10
    speakers = torch.tensor([0], dtype=torch.long, device=device)
    text_features = torch.randn(
        (n_frames, model.model.text_encoder.input_size), device=device
    )
    pitches = torch.rand((n_frames,), device=device)
    pitch_shift = None

    # Handle pitch shift enabled model
    inputs = (speakers, text_features, pitches)
    input_names = ["speakers", "text_features", "pitches"]

    if has_pitch_shift:
        pitch_shift = torch.tensor([1], dtype=torch.float, device=device)
        inputs += (pitch_shift,)
        input_names.append("pitch_shift")

    torch.onnx.export(
        model,
        inputs,
        "./exported/feature_embedding.onnx",
        opset_version=16,
        input_names=input_names,
        output_names=["features"],
        dynamic_axes={
            "text_features": {0: "n_frames"},
            "pitches": {0: "n_frames"},
            "features": {0: "n_frames"},
        },
        verbose=False,
    )

    logger.info("ONNX Feature Extractor exported.")

    # Verify the exported model
    ort_session = ort.InferenceSession("./exported/feature_embedding.onnx")
    ort_inputs = {
        "speakers": speakers.cpu().numpy(),
        "text_features": text_features.cpu().numpy(),
        "pitches": pitches.cpu().numpy(),
    }

    if has_pitch_shift:
        ort_inputs["pitch_shift"] = pitch_shift.cpu().numpy()

    ort_outs = ort_session.run(None, ort_inputs)
    torch_outs = model(speakers, text_features, pitches, pitch_shift)

    assert torch.allclose(
        torch_outs, torch.from_numpy(ort_outs[0]), atol=1e-4
    ), "ONNX model output does not match PyTorch model output."

    logger.info("ONNX Feature Extractor verified.")


def export_diffusion(config, model, device):
    # Trace the denoiser
    n_frames = 10
    x = torch.randn((1, config.mel_channels, n_frames), device=device)
    step = torch.randint(
        0, config.model.diffusion.timesteps, (1,), device=device, dtype=torch.long
    )
    cond = torch.randn(
        (1, config.model.diffusion.denoiser.condition_dim, n_frames), device=device
    )

    model.diffusion.denoise_fn = torch.jit.trace(
        model.diffusion.denoise_fn, (x, step, cond), check_trace=True
    )

    logger.info("denoiser traced.")

    # Trace naive noise predictor, since there is a randn in it, we need to verify the trace manually
    torch.manual_seed(0)
    _temp = model.diffusion.naive_noise_predictor(x, step, x)

    model.diffusion.naive_noise_predictor = torch.jit.trace(
        model.diffusion.naive_noise_predictor, (x, step, x), check_trace=False
    )

    torch.manual_seed(0)
    assert torch.allclose(_temp, model.diffusion.naive_noise_predictor(x, step, x))

    logger.info("Naive noise predictor traced.")

    # Trace the plms noise predictor
    step_prev = torch.maximum(
        step - 10, torch.tensor(0, dtype=torch.long, device=device)
    )
    noise_list = torch.randn((3, *x.shape), device=device)

    model.diffusion.plms_noise_predictor = torch.jit.trace_module(
        model.diffusion.plms_noise_predictor,
        {
            "forward": (x, x, step, step_prev),
            "predict_stage0": (x, x),
            "predict_stage1": (x, noise_list),
            "predict_stage2": (x, noise_list),
            "predict_stage3": (x, noise_list),
        },
        check_trace=True,
    )

    logger.info("PLMS noise predictor traced.")

    condition = torch.rand(
        (1, 20, config.model.diffusion.denoiser.condition_dim), device=device
    )
    sampler_interval = torch.tensor(100, dtype=torch.long, device=device)

    torch.manual_seed(0)
    _temp = model.diffusion(condition, sampler_interval)

    model.diffusion = torch.jit.script(model.diffusion)
    model.diffusion = torch.jit.optimize_for_inference(model.diffusion)

    torch.manual_seed(0)
    assert torch.allclose(
        _temp, model.diffusion(condition, sampler_interval), atol=1e-3
    )

    logger.info("Diffusion traced.")

    # As of 2023-02-14, there is an known bug in the onnx export of torchscript module
    # Issue: https://github.com/pytorch/pytorch/issues/81085
    # You need to comment out _C._jit_pass_onnx_autograd_function_process(graph)
    # in torch/onnx/utils.py to make it work

    torch.onnx.export(
        model.diffusion,
        (condition, sampler_interval, False),
        "exported/diffusion.onnx",
        opset_version=16,
        input_names=["condition", "sampler_interval", "progress"],
        output_names=["mel"],
        dynamic_axes={
            "condition": {1: "n_frames"},
            "mel": {1: "n_frames"},
        },
        verbose=False,
    )

    logger.info("Diffusion exported.")

    # Verify the exported model
    ort_session = ort.InferenceSession("exported/diffusion.onnx")
    ort_inputs = {
        "condition": condition.cpu().numpy(),
        "sampler_interval": sampler_interval.cpu().numpy(),
        "progress": np.array([False]),
    }

    ort_outs = ort_session.run(None, ort_inputs)[0]

    # We can't set seed for the onnxruntime, the output of the diffusion is not deterministic
    _temp = _temp.detach().cpu().numpy()
    assert (
        ort_outs.shape == _temp.shape
    ), "ONNX model output shape does not match PyTorch model output shape."

    error = np.mean(np.abs(ort_outs - _temp))
    logger.info(f"ONNX Diffusion shape verified, average absolute error: {error:.4f}.")


class FeatureExtractorWrapper(torch.nn.Module):
    def __init__(self, feature_extractor):
        super().__init__()
        self.feature_extractor = feature_extractor

    def forward(self, x):
        x = self.feature_extractor._forward(x)

        return x.transpose(1, 2)


def export_feature_extractor(config, device):
    # if config.preprocessing.text_features_extractor.type == "ContentVec":
    #     logger.warning("ContentVec is not supported in ONNX. Skip exporting.")
    #     return

    feature_extractor = FEATURE_EXTRACTORS.build(
        config.preprocessing.text_features_extractor
    )
    feature_extractor = FeatureExtractorWrapper(feature_extractor)
    feature_extractor.eval()
    feature_extractor.to(device)

    # Fake audio
    x = torch.randn((1, 44100), device=device)

    torch.onnx.export(
        feature_extractor,
        (x,),
        "exported/feature_extractor.onnx",
        opset_version=16,
        input_names=["waveform"],
        output_names=["features"],
        dynamic_axes={
            "waveform": {1: "n_samples"},
            "features": {2: "n_frames"},
        },
        verbose=False,
    )

    logger.info("Feature extractor exported.")

    # Verify the exported model
    ort_session = ort.InferenceSession("exported/feature_extractor.onnx")
    ort_inputs = {
        "waveform": x.cpu().numpy(),
    }

    ort_outs = ort_session.run(None, ort_inputs)[0]

    assert torch.allclose(
        feature_extractor(x), torch.from_numpy(ort_outs), atol=1e-4
    ), "ONNX model output does not match PyTorch model output."

    logger.info("ONNX Feature extractor verified.")


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
    if config.model.type == "DiffSinger":
        logger.info(
            "DiffSingeer model does not have feature extractor. Skip exporting."
        )
    else:
        export_feature_extractor(config, device)

    # Export feature embedding
    export_feature_embedding(model, device)

    # Export diffusion
    export_diffusion(config, model, device)


if __name__ == "__main__":
    main()
