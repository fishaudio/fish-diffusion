import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

trainer = dict(
    accelerator="gpu",
    devices=-1,
    strategy=DDPStrategy(find_unused_parameters=True),
    gradient_clip_val=0.5,
    log_every_n_steps=10,
    val_check_interval=5000,
    check_val_every_n_epoch=None,
    max_steps=250000,
    # Note: bf16 is not supported on GPUs older than 30 series
    # Warning: If you are training the model with fs2, you should either use bf16 or fp32
    precision="bf16" if torch.cuda.is_bf16_supported() else 16,
    callbacks=[
        ModelCheckpoint(
            filename="diff-svc-{epoch:02d}-{valid_loss:.2f}",
            every_n_train_steps=10000,
        ),
        LearningRateMonitor(logging_interval="step"),
    ],
)
