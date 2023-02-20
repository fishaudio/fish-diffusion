from functools import partial

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from fish_diffusion.utils.pitch import pitch_to_scale

trainer = dict(
    accelerator="gpu",
    devices=-1,
    gradient_clip_val=0.5,
    # log_every_n_steps=10,
    # val_check_interval=5000,
    # check_val_every_n_epoch=None,
    max_steps=-1,
    # Warning: If you are training the model with fs2 (and see nan), you should either use bf16 or fp32
    precision=16,
    callbacks=[
        ModelCheckpoint(
            filename="{epoch}-{step}-{valid_loss:.2f}",
            # every_n_train_steps=10000,
            every_n_epochs=1,
            save_top_k=-1,
        ),
        LearningRateMonitor(logging_interval="step"),
    ],
    strategy=DDPStrategy(find_unused_parameters=True, process_group_backend="nccl"),
)

_pitch_to_scale = partial(pitch_to_scale, f0_min=40.0, f0_max=1600.0)

model = dict(
    type="NSF-HiFiGAN",
    config="checkpoints/nsf_hifigan_finetune/config.json",
    generator_checkpoint="checkpoints/nsf_hifigan_finetune/g_01809000",
    discriminator_checkpoint="checkpoints/nsf_hifigan_finetune/do_01806000",
    vocoder=dict(
        type="NsfHifiGAN",
        checkpoint_path="checkpoints/nsf_hifigan/model",
        sampling_rate=44100,
        mel_channels=128,
        use_natural_log=True,
    ),
)

dataloader = dict(
    train=dict(
        batch_size=20,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    ),
    valid=dict(
        batch_size=20,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    ),
)

preprocessing = dict(
    text_features_extractor=None,
    pitch_extractor=dict(
        type="ParselMouthPitchExtractor",
        keep_zeros=False,
        f0_min=40.0,
        f0_max=1600.0,
    ),
)
