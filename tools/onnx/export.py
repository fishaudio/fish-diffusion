import torch
from loguru import logger
from mmengine import Config

from train import FishDiffusion


def main():
    device = "cpu"

    config = Config.fromfile("configs/exp_lengyue_mix.py")
    model = FishDiffusion(config)
    state_dict = torch.load(
        "logs/DiffSVC/8r0ci7id/checkpoints/epoch=153-step=260000-valid_loss=0.17.ckpt",
        map_location=device,
    )["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)

    logger.info("Model loaded.")

    # Ignore vocoder
    model = model.model

    # Trace the denosier
    n_frames = 10
    x = torch.randn((1, config.mel_channels, n_frames), device=device)
    step = torch.randint(
        0, config.model.diffusion.timesteps, (1,), device=device, dtype=torch.long
    )
    cond = torch.randn((1, config.hidden_size, n_frames), device=device)

    model.diffusion.denoise_fn = torch.jit.trace(
        model.diffusion.denoise_fn, (x, step, cond), check_trace=True
    )

    logger.info("Denosier traced.")

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

    condition = torch.rand((1, 20, config.hidden_size), device=device)
    sampler_interval = torch.tensor(100, dtype=torch.long, device=device)

    torch.manual_seed(0)
    _temp = model.diffusion(condition, sampler_interval)

    model.diffusion = torch.jit.script(model.diffusion)
    model.diffusion = torch.jit.optimize_for_inference(model.diffusion)

    torch.manual_seed(0)
    assert torch.allclose(
        _temp, model.diffusion(condition, sampler_interval), atol=1e-4
    )

    logger.info("Diffusion traced.")

    # As of 2023-02-14, there is an known bug in the onnx export of torchscript module
    # Issue: https://github.com/pytorch/pytorch/issues/81085
    # You need to comment out _C._jit_pass_onnx_autograd_function_process(graph)
    # in torch/onnx/utils.py to make it work

    torch.onnx.export(
        model.diffusion,
        (condition, sampler_interval, False),
        "diffusion.onnx",
        opset_version=16,
        input_names=["condition", "sampler_interval", "progress"],
        output_names=["mel"],
        dynamic_axes={
            "condition": {1: "n_frames"},
            "mel": {1: "n_frames"},
        },
        verbose=True,
    )

    logger.info("Diffusion exported.")

    # print(model.diffusion.graph)


if __name__ == "__main__":
    main()
