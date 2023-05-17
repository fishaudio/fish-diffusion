from sched import scheduler
from omegaconf import DictConfig, OmegaConf
from loguru import logger
import torch
import sys
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as default


def generate_config(model, dataset, scheduler, output_name):
    config = OmegaConf.create()

    # Determine which parts of the configuration to include
    if model == "diff_svc_v2":
        config.model = OmegaConf.load("configs/model/diff_svc_v2.yaml")

    if dataset == "naive_svc":
        config.dataset = OmegaConf.load("configs/dataset/naive_svc.yaml")
        config.dataloader = OmegaConf.load("configs/dataloader/naive_svc.yaml")

    if scheduler == "warmup_cosine":
        config.scheduler = OmegaConf.load("configs/scheduler/warmup_cosine.yaml")
        config.optimizer = OmegaConf.load("configs/optimizer/warmup_cosine.yaml")

    config.trainer = OmegaConf.load("configs/trainer/base.yaml")

    # Save the resulting configuration to a file
    # OmegaConf.save(final_cfg, f"{output_name}.yaml")
    logger.debug(OmegaConf.to_yaml(config))

    OmegaConf.save(config, f"configs/{output_name}.yaml", resolve=True)


def create_ddp_strategy():
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        return {
            "find_unused_parameters": True,
            "process_group_backend": "nccl" if sys.platform != "win32" else "gloo",
            "gradient_as_bucket_view": True,
            "ddp_comm_hook": default.fp16_compress_hook,
        }
    else:
        return None


if __name__ == "__main__":
    # Register custom resolvers for configuration variables
    OmegaConf.register_new_resolver("mel_channels", lambda: 128)
    OmegaConf.register_new_resolver("sampling_rate", lambda: 44100)
    OmegaConf.register_new_resolver("hidden_size", lambda: 256)
    OmegaConf.register_new_resolver("n_fft", lambda: 2048)
    OmegaConf.register_new_resolver("hop_length", lambda: 256)
    OmegaConf.register_new_resolver("win_length", lambda: 2048)
    OmegaConf.register_new_resolver("create_ddp_strategy", create_ddp_strategy)
    model = "diff_svc_v2"
    dataset = "naive_svc"
    scheduler = "warmup_cosine"
    output_name = "test"
    generate_config(model, dataset, scheduler, output_name)
