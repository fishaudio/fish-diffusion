from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

trainer = dict(
    accelerator="gpu",
    devices=-1,
    strategy=DDPStrategy(find_unused_parameters=False),
    gradient_clip_val=0.5,
    log_every_n_steps=10,
    val_check_interval=1000,
    check_val_every_n_epoch=None,
    max_steps=300000,
    precision=16,
    callbacks=[
        ModelCheckpoint(
            filename="diff-svc-{epoch:02d}-{valid_loss:.2f}",
            every_n_train_steps=1000,
        ),
        LearningRateMonitor(logging_interval="step")
    ]
)
