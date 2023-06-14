import os
import click
from matplotlib.dates import SA
from networkx import project
from omegaconf import OmegaConf
from loguru import logger
from pathlib import Path
from sympy import N
import torch
import sys
from typing import Dict

from hydra.experimental import compose, initialize

MEL_CHANNELS = 128
SAMPLING_RATE = 44100
HIDDEN_SIZE = 256
N_FFT = 2048
HOP_LENGTH = 256
WIN_LENGTH = 2048


def create_ddp_strategy():
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        return {
            "_target_": "pytorch_lightning.strategies.DDPStrategy",
            "find_unused_parameters": True,
            "process_group_backend": "nccl" if sys.platform != "win32" else "gloo",
            "gradient_as_bucket_view": True,
            "ddp_comm_hook": "torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook",
        }
    else:
        return None


def build_datasets(
    train_speaker_ids: Dict[str, int],
    val_speaker_ids: Dict[str, int],
    datasetConf: OmegaConf,
    model: str,
) -> Dict[str, Dict]:
    if len(train_speaker_ids.keys()) == 0:
        train_datasets = {
            "_target_": f"{datasetConf.train._target_}",
            "path": f"dataset/train",
            # "segment_size": datasetConf.train.segment_size,
            "speaker_id": 0,
        }
        speaker_mapping = {
            "placeholder": 0,
        }
    else:
        train_datasets = {
            "_target_": "fish_diffusion.datasets.concat.ConcatDataset",
            "datasets": [
                {
                    "_target_": f"{datasetConf.train._target_}",
                    "path": f"dataset/train/{speaker}",
                    # "segment_size": datasetConf.train.segment_size,
                    "speaker_id": train_speaker_ids[speaker],
                }
                for speaker in train_speaker_ids.keys()
            ],
            "collate_fn": f"{datasetConf.train._target_}.collate_fn",
        }
        speaker_mapping = {
            speaker: train_speaker_ids[speaker] for speaker in train_speaker_ids.keys()
        }

    if len(val_speaker_ids.keys()) == 0:
        valid_datasets = {
            "_target_": f"{datasetConf.valid._target_}",
            "path": f"dataset/valid",
            "speaker_id": 0,
        }
    else:
        valid_datasets = {
            "_target_": "fish_diffusion.datasets.concat.ConcatDataset",
            "datasets": [
                {
                    "_target_": f"{datasetConf.valid._target_}",
                    "path": f"dataset/valid/{speaker}",
                    "speaker_id": train_speaker_ids.get(
                        speaker, val_speaker_ids[speaker]
                    ),
                }
                for speaker in val_speaker_ids.keys()
            ],
            "collate_fn": f"{datasetConf.valid._target_}.collate_fn",
        }

    # valid_datasets = [
    #     {
    #         "_target_": f"{datasetConf.valid._target_}",
    #         "path": f"dataset/valid",
    #         "speaker_id": 0,
    #     }
    # ]

    # add segment_size for hifi models
    if "hifi" in model.lower():
        if len(train_speaker_ids.keys()) == 0:
            train_datasets["segment_size"] = datasetConf.train.segment_size
        else:
            for dataset in train_datasets["datasets"]:
                dataset["segment_size"] = datasetConf.train.segment_size

    return {
        "train": train_datasets,
        "valid": valid_datasets,
        "speaker_mapping": speaker_mapping,
    }


# def build_naive_svc_datasets(
#     train_speaker_ids: Dict[str, int],
#     val_speaker_ids: Dict[str, int],
#     datasetConf: OmegaConf,
# ) -> Dict[str, Dict]:
#     if len(train_speaker_ids.keys()) == 0:
#         train_datasets["datasets"] = [
#             {
#                 "_target_": f"{datasetConf.train._target_}",
#                 "path": f"dataset/train",
#             }
#         ]
#         speaker_mapping = {"placeholder": 0}
#     else:
#         train_datasets = {
#             "_target_": "fish_diffusion.datasets.concat.ConcatDataset",
#             "datasets": [
#                 {
#                     "_target_": f"{datasetConf.train._target_}",
#                     "path": f"dataset/train/{speaker}",
#                     "speaker_id": train_speaker_ids[speaker],
#                 }
#                 for speaker in train_speaker_ids.keys()
#             ],
#             "collate_fn": f"{datasetConf.train._target_}.collate_fn",
#         }
#         speaker_mapping = {
#             speaker: train_speaker_ids[speaker] for speaker in train_speaker_ids.keys()
#         }

#     if len(val_speaker_ids.keys()) == 0:
#         valid_datasets["datasets"] = [
#             {
#                 "_target_": f"{datasetConf.valid._target_}",
#                 "path": f"dataset/valid",
#             }
#         ]
#     else:
#         valid_datasets = {
#             "_target_": "fish_diffusion.datasets.concat.ConcatDataset",
#             "datasets": [
#                 {
#                     "_target_": f"{datasetConf.valid._target_}",
#                     "path": f"dataset/valid/{speaker}",
#                     "speaker_id": train_speaker_ids.get(
#                         speaker, val_speaker_ids[speaker]
#                     ),
#                 }
#                 for speaker in val_speaker_ids.keys()
#             ],
#             "collate_fn": f"{datasetConf.valid._target_}.collate_fn",
#         }

#     return {
#         "train": train_datasets,
#         "valid": valid_datasets,
#         "speaker_mapping": speaker_mapping,
#     }


def generate_config(
    model, dataset, scheduler, output_name, is_multi_speaker, trainer, output_dir
):
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
    config.project_root = os.getcwd()
    config.sampling_rate = SAMPLING_RATE

    OmegaConf.register_new_resolver("project_root", lambda: os.getcwd())

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
        # if not is_multi_speaker:
        #     config.dataset = datasetConf
        # else:
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
        config.dataset = OmegaConf.create(
            build_datasets(train_speaker_ids, val_speaker_ids, datasetConf, model)
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
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    OmegaConf.save(config, f"{output_dir}/{output_name}.yaml", resolve=True)
    logger.info(f"Saved configuration to {output_dir}/{output_name}.yaml")


@click.command()
@click.option("--model", "-m", default="diff_svc_v2", help="Model to use")
@click.option("--dataset", "-d", default="naive_svc", help="Dataset to use")
@click.option("--scheduler", "-s", default="warmup_cosine", help="Scheduler to use")
@click.option(
    "--output", "-o", default="svc_hubert_soft", help="Name of the output file"
)
@click.option("--trainer", "-t", default="base", help="Name of the trainer file")
@click.option(
    "--is_multi_speaker",
    "-m",
    is_flag=True,
    help="Whether to use multi-speaker dataset",
)
@click.option(
    "--dir-name", "-n", default="./configs", help="Name of the output directory"
)
def main(model, dataset, scheduler, output, is_multi_speaker, trainer, dir_name):
    generate_config(
        model, dataset, scheduler, output, is_multi_speaker, trainer, dir_name
    )


if __name__ == "__main__":
    # Register custom resolvers for configuration variables
    OmegaConf.register_new_resolver("mel_channels", lambda: MEL_CHANNELS)
    OmegaConf.register_new_resolver("sampling_rate", lambda: SAMPLING_RATE)
    OmegaConf.register_new_resolver("hidden_size", lambda: HIDDEN_SIZE)
    # for hifi
    OmegaConf.register_new_resolver("n_fft", lambda: N_FFT)
    OmegaConf.register_new_resolver("hop_length", lambda: HOP_LENGTH)
    OmegaConf.register_new_resolver("win_length", lambda: WIN_LENGTH)

    # for speaker encoder speaker_embedding_size
    OmegaConf.register_new_resolver("speaker_embedding_size", lambda: 1)
    # for ddp
    OmegaConf.register_new_resolver("create_ddp_strategy", create_ddp_strategy)
    main()
