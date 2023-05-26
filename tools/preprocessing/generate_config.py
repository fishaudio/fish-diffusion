import click
from omegaconf import OmegaConf
from loguru import logger
from pathlib import Path
import omegaconf
import torch
import sys
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as default
from typing import Dict
from hydra.utils import get_original_cwd

from hydra.experimental import compose, initialize


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


def build_hifi_svc_datasets(
    train_speaker_ids: Dict[str, int],
    val_speaker_ids: Dict[str, int],
    datasetConf: OmegaConf,
) -> Dict[str, Dict]:
    train_datasets = {
        "_target_": "fish_diffusion.datasets.ConcatDataset",
        "datasets": [
            {
                "_target_": f"{datasetConf.train._target_}",
                "path": f"dataset/train/{speaker}",
                "segment_size": datasetConf.train.segment_size,
                "speaker_id": train_speaker_ids[speaker],
            }
            for speaker in train_speaker_ids.keys()
        ],
        "collate_fn": f"{datasetConf.train._target_}.collate_fn",
    }

    valid_datasets = {
        "_target_": "fish_diffusion.datasets.ConcatDataset",
        "datasets": [
            {
                "_target_": f"{datasetConf.valid._target_}",
                "path": f"dataset/valid/{speaker}",
                "segment_size": datasetConf.valid.segment_size,
                "speaker_id": train_speaker_ids.get(speaker, val_speaker_ids[speaker]),
            }
            for speaker in val_speaker_ids.keys()
        ],
        "collate_fn": f"{datasetConf.valid._target_}.collate_fn",
    }

    return {"train": train_datasets, "valid": valid_datasets}


def build_naive_svc_datasets(
    train_speaker_ids: Dict[str, int],
    val_speaker_ids: Dict[str, int],
    datasetConf: OmegaConf,
) -> Dict[str, Dict]:
    train_datasets = {
        "_target_": "fish_diffusion.datasets.ConcatDataset",
        "datasets": [
            {
                "_target_": f"{datasetConf.train._target_}",
                "path": f"dataset/train/{speaker}",
                "speaker_id": train_speaker_ids[speaker],
            }
            for speaker in train_speaker_ids.keys()
        ],
        "collate_fn": f"{datasetConf.train._target_}.collate_fn",
    }

    valid_datasets = {
        "_target_": "fish_diffusion.datasets.ConcatDataset",
        "datasets": [
            {
                "_target_": f"{datasetConf.valid._target_}",
                "path": f"dataset/valid/{speaker}",
                "speaker_id": train_speaker_ids.get(speaker, val_speaker_ids[speaker]),
            }
            for speaker in val_speaker_ids.keys()
        ],
        "collate_fn": f"{datasetConf.valid._target_}.collate_fn",
    }

    return {"train": train_datasets, "valid": valid_datasets}


def generate_config(model, dataset, scheduler, output_name, is_multi_speaker, trainer):
    config = OmegaConf.create()

    config.text_features_extractor_type = "HubertSoft"
    config.pitch_extractor_type = "ParselMouthPitchExtractor"
    config.pretrained = None
    config.resume = None
    config.tensorboard = False
    config.resume_id = None
    config.entity = None
    config.name = None
    config.only_train_speaker_embeddings = False
    config.path = "dataset"
    config.clean = False
    config.num_workers = 8
    config.no_augmentation = True

    # Determine which parts of the configuration to include
    try:
        config.model = OmegaConf.load(f"configs/model/{model}.yaml")
        config.preprocessing = OmegaConf.load(f"configs/preprocessing/{model}.yaml")
    except FileNotFoundError:
        logger.error(f"Could not find model {model}, exiting.")
        raise click.Abort()
    config.model_type = config.model.type

    try:
        datasetConf = OmegaConf.load(f"configs/dataset/{dataset}.yaml")
        if not is_multi_speaker:
            config.dataset = datasetConf
        else:
            # Get speaker ids
            train_speaker_ids = {}
            for i, folder in enumerate((Path(config.path) / "train").iterdir()):
                if folder.is_dir():
                    train_speaker_ids[folder.name] = i
            val_speaker_ids = {}
            for i, folder in enumerate((Path(config.path) / "valid").iterdir()):
                if folder.is_dir():
                    val_speaker_ids[folder.name] = i

            # Create datasets for each speaker
            if model == "hifi_svc":
                config.dataset = OmegaConf.create(
                    build_hifi_svc_datasets(
                        train_speaker_ids, val_speaker_ids, datasetConf
                    )
                )
            else:
                config.dataset = OmegaConf.create(
                    build_naive_svc_datasets(
                        train_speaker_ids, val_speaker_ids, datasetConf
                    )
                )

            # change the input size of the speaker encoder in the model
            config.model.speaker_encoder.input_size = len(train_speaker_ids.keys())
        config.dataloader = OmegaConf.load(f"configs/dataloader/base.yaml")

    except FileNotFoundError as e:
        if is_multi_speaker:
            logger.error(f"error: {e}")
        else:
            logger.error(f"Could not find dataset {dataset}, exiting.")
        raise click.Abort()

    try:
        config.scheduler = OmegaConf.load(f"configs/scheduler/{scheduler}.yaml")
        config.optimizer = OmegaConf.load(f"configs/optimizer/{scheduler}.yaml")
    except FileNotFoundError as e:
        logger.error(f"Could not find scheduler {scheduler}, exiting.")
        raise click.Abort()

    # Add trainer configuration
    with initialize(config_path="../../configs"):
        trainer_conf = compose(config_name=f"trainer/{trainer}")
        config = OmegaConf.merge(config, trainer_conf)

    # Save the resulting configuration to a file
    OmegaConf.save(config, f"configs/{output_name}.yaml", resolve=True)
    logger.info(f"Saved configuration to configs/{output_name}.yaml")


@click.command()
@click.option("--model", default="diff_svc_v2", help="Model to use")
@click.option("--dataset", default="naive_svc", help="Dataset to use")
@click.option("--scheduler", default="warmup_cosine", help="Scheduler to use")
@click.option("--output", default="svc_hubert_soft", help="Name of the output file")
@click.option("--trainer", default="base", help="Name of the trainer file")
@click.option(
    "--is_multi_speaker", default=True, help="Whether to use multi-speaker dataset"
)
def main(model, dataset, scheduler, output, is_multi_speaker, trainer):
    generate_config(model, dataset, scheduler, output, is_multi_speaker, trainer)


if __name__ == "__main__":
    # Register custom resolvers for configuration variables
    OmegaConf.register_new_resolver("mel_channels", lambda: 128)
    OmegaConf.register_new_resolver("sampling_rate", lambda: 44100)
    OmegaConf.register_new_resolver("hidden_size", lambda: 256)
    # for hifi
    OmegaConf.register_new_resolver("n_fft", lambda: 2048)
    OmegaConf.register_new_resolver("hop_length", lambda: 256)
    OmegaConf.register_new_resolver("win_length", lambda: 2048)
    # for ddp
    OmegaConf.register_new_resolver("create_ddp_strategy", create_ddp_strategy)
    main()
